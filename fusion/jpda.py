"""
JPDA 关联 - 联合概率数据关联
替换/并列贪心 N-to-1 匈牙利匹配

核心思想：
- 多个检测可能来自同一个目标（ clutter / 共享检测 ）
- JPDA 计算每个 (检测, 轨迹) 对的边缘关联概率 β_ij
- 轨迹更新用所有有效检测的加权平均（概率加权）

参考文献：
- Bar-Shalom & Fortmann, "Tracking and Data Association", 1988
- Blackman & Popoli, "Design and Analysis of Modern Tracking Systems", 1999
"""
import numpy as np
from scipy.special import gamma as gamma_func


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def _mahal_distance(det_pos, trk_x, trk_P, dim=3):
    """马氏距离（检测到轨迹预测位置的distance）"""
    z = det_pos - trk_x[:dim]
    try:
        inv_S = np.linalg.inv(trk_P[:dim, :dim] + 1e-6 * np.eye(dim))
    except np.linalg.LinAlgError:
        return float(np.linalg.norm(z))
    d2 = float(z @ inv_S @ z)
    return np.sqrt(d2)


def _chi2_cdf(x2, df):
    """Chi-square CDF（用不完全 gamma 近似）"""
    return _gammainc(df / 2, x2 / 2)


def _gammainc(a, x):
    """不完全 gamma 函数 CDF（scipy.special.gammainc 的简化版）"""
    from scipy.special import gammainc
    return gammainc(a, x)


def _gate_volume(trk_P, gate_threshold, dim=3):
    """validation gate 体积（用于 P_D * 体积 计算期望检测数）"""
    det_P = trk_P[:dim, :dim]
    try:
        chol = np.linalg.cholesky(det_P + 1e-9 * np.eye(dim))
        vol = (np.pi ** (dim / 2) / gamma_func(dim / 2 + 1)) \
            * (gate_threshold ** dim) \
            * (2 ** dim) * np.prod(np.diag(chol))
    except np.linalg.LinAlgError:
        vol = 1e6
    return max(vol, 1e-9)


# ──────────────────────────────────────────────────────────────
# JPDA 关联核心
# ──────────────────────────────────────────────────────────────

def jpda_associate(detections: list,
                   tracks: list,
                   gate_threshold: float = 5.0,
                   PD: float = 0.9,          # 检测概率
                   clut_density: float = 1e-5,  # 杂波密度（空间密度）
                   track_existence_prior: float = 0.95,
                   use_mahalanobis: bool = True,
                   debug: bool = False) -> tuple:
    """
    JPDA 关联

    Args:
        detections: list of Detection
        tracks: list of EKFTrack / UKFTrack（有 .kf.x / .kf.P）
        gate_threshold: 门限（Mahalanobis distance 单位，或米）
        PD: 检测概率 (0~1)
        clut_density: 杂波空间密度（用于 clutter 概率计算）
        track_existence_prior: 轨迹存在先验概率
        use_mahalanobis: True=马氏距离 gate，False=欧氏距离
        debug: 打印调试信息

    Returns:
        (association_weights, unmatched_detections, unmatched_tracks)

        association_weights:
            dict track_idx -> list of (det_idx, beta) weights
            β_ij = 边缘关联概率（检测 i → 轨迹 j 的概率）
            β_0j = 1 - sum_i β_ij = clutter 概率（本帧无更新）

        unmatched_detections: list of det_idx（未与任何轨迹关联，进入 clutter）
        unmatched_tracks: list of track_idx（无有效检测，进入 miss）
    """
    n_det = len(detections)
    n_trk = len(tracks)
    if n_det == 0:
        return {}, list(range(n_det)), list(range(n_trk))
    if n_trk == 0:
        return {}, list(range(n_det)), []

    # ── Step 1: 计算 Mahalanobis 距离矩阵 ──────────────────────
    # GATE[i,j] = True 如果检测 i 在轨迹 j 的 validation gate 内
    mahal = np.full((n_det, n_trk), np.inf)
    gate = np.zeros((n_det, n_trk), dtype=bool)

    for i, det in enumerate(detections):
        for j, trk in enumerate(tracks):
            if use_mahalanobis:
                d = _mahal_distance(det.position, trk.kf.x, trk.kf.P)
            else:
                d = float(np.linalg.norm(det.position - trk.kf.x[:3]))
            mahal[i, j] = d
            # 使用 chi2 gate（dim=3, Mahalanobis 距离平方服从 chi2_3）
            # gate_threshold 作为 chi2 阈值（ squared distance < gate_threshold^2 ）
            gate[i, j] = (d <= gate_threshold)
            # 备选：如果用欧氏距离，直接用 gate_threshold 作为米
            if not use_mahalanobis:
                gate[i, j] = (d <= gate_threshold * 3)  # 3σ 等效

    # ── Step 2: 计算每个 (det, track) 的似然 ──────────────────
    # L_ij = 检测 i 来自 track j 的似然（高斯分布）
    L = np.zeros((n_det, n_trk))
    for i in range(n_det):
        for j in range(n_trk):
            if not gate[i, j]:
                continue
            d2 = mahal[i, j] ** 2
            # |S| = P 的行列式（观测噪声协方差）
            S = max(np.linalg.det(tracks[j].kf.P[:3, :3]), 1e-9)
            L[i, j] = np.exp(-0.5 * d2) / (np.sqrt((2 * np.pi) ** 3 * S) + 1e-9)

    # beta 数组: beta[0,j]=clutter概率; beta[i,j] for i>=1 = P(det_i → trk_j)
    beta = np.zeros((n_det + 1, n_trk))

    # ── Step 3: 标准 marginal JPDA 公式（按 track 独立归一）──
    # β_ij = e_ij / (b_j + Σ_i e_ij)
    # β_0j = b_j  / (b_j + Σ_i e_ij)
    # 其中 e_ij = PD * L_ij
    #       b_j  = 杂波先验（小常量，默认 0.05，表示 "该 track 检测不到的先验概率"）
    #
    # 这是 Bar-Shalom 原始公式（1988），对每个 track 独立归一。
    # b_j 使用固定小量代替 clut_density*gate_volume（后者在低 P / 离群点下
    # 会变极大导致 β_0j 接近 1，误判所有检测为 clutter）。
    valid_dets_per_track = [np.where(gate[:, j])[0] for j in range(n_trk)]
    for j in range(n_trk):
        v = valid_dets_per_track[j]
        if len(v) == 0:
            beta[0, j] = 1.0
            continue
        # 固定小 b_j（可调，默认 0.05 等价于 "本帧检测 5% 概率是 clutter"）
        b_j = clut_density if clut_density < 0.1 else 0.05
        # 逐个检测的 e_ij = PD * L_ij
        e = np.array([PD * L[i, j] * track_existence_prior for i in v])
        # 归一化分母
        denom = b_j + np.sum(e) + 1e-12
        for k, di in enumerate(v):
            beta[di + 1, j] = e[k] / denom
        beta[0, j] = b_j / denom

    # ── Step 3: 找出无有效检测的 track → miss ──────────────────
    unmatched_trks = [j for j in range(n_trk) if len(valid_dets_per_track[j]) == 0]
    # 检查是否有 clutter-only 的 track（beta[0,j] 接近 1）
    for j in range(n_trk):
        if beta[0, j] > 0.95 and len(valid_dets_per_track[j]) > 0:
            # 检测有效但都被 clutter 淹没 → miss
            unmatched_trks.append(j)

    unmatched_trks = list(set(unmatched_trks))  # 去重

    # ── Step 4: 构建 association_weights ────────────────────────
    # 每个 track 的 β_ij 加权和（用于 update）
    association_weights = {}
    for j in range(n_trk):
        if j in unmatched_trks:
            continue
        weights = []  # list of (det_idx, beta)
        for i in range(1, n_det + 1):
            if beta[i, j] > 1e-6:
                det_idx = i - 1
                weights.append((det_idx, float(beta[i, j])))
        if weights:
            association_weights[j] = weights
        else:
            if j not in unmatched_trks:
                unmatched_trks.append(j)

    # 检测未与任何 track 关联 → clutter
    matched_det_indices = set()
    for j, wlist in association_weights.items():
        for di, _ in wlist:
            matched_det_indices.add(di)
    unmatched_dets = [i for i in range(n_det) if i not in matched_det_indices]

    if debug:
        print(f'  JPDA: {n_det}d x {n_trk}t  '
              f'matched={len(association_weights)}t  '
              f'miss={len(unmatched_trks)}t  clutter={len(unmatched_dets)}d')
        for j, wlist in association_weights.items():
            betas_str = " + ".join([f"D{d}+{b:.3f}" for d, b in wlist])
            print(f"    T{j}: {betas_str} (clutter={beta[0,j]:.3f})")

    return association_weights, unmatched_dets, unmatched_trks


def jpda_associate_enumeration(detections: list,
                                tracks: list,
                                gate_threshold: float = 3.0,
                                PD: float = 0.9,
                                clut_density: float = 1e-5,
                                use_mahalanobis: bool = True,
                                max_events: int = 5000,
                                debug: bool = False) -> tuple:
    """
    JPDA 关联（穷举联合事件版本，适用于低检测数场景）
    枚举所有可行的联合关联事件，计算精确的边缘概率

    适用于: n_det ≤ 8, n_trk ≤ 6
    对于 Dense Highway (24车)，用上面的简化 jpda_associate 更实用
    """
    n_det = len(detections)
    n_trk = len(tracks)
    if n_det == 0 or n_trk == 0:
        return {}, list(range(n_det)), list(range(n_trk))

    # 验证门限（Mahalanobis）
    mahal = np.full((n_det, n_trk), np.inf)
    gate = np.zeros((n_det, n_trk), dtype=bool)
    for i, det in enumerate(detections):
        for j, trk in enumerate(tracks):
            d = _mahal_distance(det.position, trk.kf.x, trk.kf.P) if use_mahalanobis \
                else float(np.linalg.norm(det.position - trk.kf.x[:3]))
            mahal[i, j] = d
            gate[i, j] = d <= gate_threshold

    # 归一化 Mahalanobis → 似然
    L = np.zeros((n_det, n_trk))
    for i in range(n_det):
        for j in range(n_trk):
            if not gate[i, j]:
                continue
            d2 = mahal[i, j] ** 2
            S = max(np.linalg.det(tracks[j].kf.P[:3, :3]), 1e-9)
            L[i, j] = np.exp(-0.5 * d2) / (np.sqrt((2 * np.pi) ** 3 * S) + 1e-9)

    # 穷举联合事件（检测分配给track 或 clutter）
    # 每个检测: 分配给某个 track(0..n_trk) 或 clutter(n_trk+1)
    # track 约束: 每个 track 至多接收 1 个检测
    beta = np.zeros((n_det + 1, n_trk))  # beta[0,j]=clutter; beta[i,j]=P(det_i->trk_j)
    event_count = 0

    def enumerate_assignments(d_idx, assign, track_assigned):
        nonlocal event_count, beta
        if event_count > max_events:
            return
        if d_idx == n_det:
            # 评估这个 assign
            # 计算事件似然
            event_L = 1.0
            for di, a in enumerate(assign):
                if a <= n_trk:  # 分配给某个 track
                    tj = a - 1
                    if not gate[di, tj]:
                        event_L = 0.0
                        break
                    event_L *= L[di, tj] * PD
                else:
                    event_L *= clut_density
            if event_L <= 0:
                return
            # 累积 beta（归一化在最后）
            for di, a in enumerate(assign):
                if 1 <= a <= n_trk:
                    beta[di + 1, a - 1] += event_L
            event_count += 1
            return

        # 枚举检测 di 的分配
        # 尝试分配给每个 track（如果该 track 还未被分配且在 gate 内）
        for tj in range(n_trk):
            if not track_assigned[tj] and gate[d_idx, tj]:
                assign[d_idx] = tj + 1
                track_assigned[tj] = True
                enumerate_assignments(d_idx + 1, assign, track_assigned)
                track_assigned[tj] = False
                assign[d_idx] = -1
        # 或分配给 clutter
        assign[d_idx] = n_trk + 1
        enumerate_assignments(d_idx + 1, assign, track_assigned)
        assign[d_idx] = -1

    assign = [-1] * n_det
    track_assigned = [False] * n_trk
    enumerate_assignments(0, assign, track_assigned)

    # 归一化
    total = np.sum(beta) + 1e-12
    beta /= total
    for j in range(n_trk):
        beta[0, j] = 1.0 - np.sum(beta[1:, j])

    # 构建 association_weights
    association_weights = {}
    valid_dets_per_track = [np.where(gate[:, j])[0] for j in range(n_trk)]
    unmatched_trks = [j for j in range(n_trk) if len(valid_dets_per_track[j]) == 0 or beta[0, j] > 0.95]

    for j in range(n_trk):
        if j in unmatched_trks:
            continue
        weights = [(i, float(beta[i + 1, j])) for i in range(n_det) if beta[i + 1, j] > 1e-6]
        if weights:
            association_weights[j] = weights

    matched_det_indices = set()
    for j, wlist in association_weights.items():
        for di, _ in wlist:
            matched_det_indices.add(di)
    unmatched_dets = [i for i in range(n_det) if i not in matched_det_indices]

    if debug:
        print(f'  JPDA-enum: {n_det}d x {n_trk}t  events={event_count}  '
              f'matched={len(association_weights)}t  miss={len(unmatched_trks)}t')

    return association_weights, unmatched_dets, list(set(unmatched_trks))


def auto_jpda(detections: list, tracks: list,
              gate_threshold: float = 5.0,
              use_mahalanobis: bool = True,
              debug: bool = False) -> tuple:
    """
    JPDA 关联 - 统一调用 marginal JPDA 公式
    （枚举版有归一化 bug，现已跳走）
    """
    return jpda_associate(
        detections, tracks,
        gate_threshold=gate_threshold,
        use_mahalanobis=use_mahalanobis,
        debug=debug)
