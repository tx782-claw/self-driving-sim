"""
数据关联 - 多检测匹配（贪心 N-to-1）
关键修复: 多个检测可以更新同一个 track（多传感器融合的核心）

P3-B 增强 (v0.4):
- 新增 _mahal_distance() — Mahalanobis 距离 (考虑 track 协方差)
- 新增 _chi2_gate() — Chi-square 门限 (3D 默认 7.815 = 95% 分位)
- 新增 _build_mahal_cost_matrix() — 马氏距离成本矩阵 (含 chi2 门限过滤)
- hungarian_associate() 新增 use_mahalanobis=True 支持
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2


# Chi-square 门限常量 (P3-B)
# 3D: 自由度=3
# - 95% 分位: 7.815
# - 99% 分位: 11.345
# - 99.9% 分位: 16.266
CHI2_3D_95 = 7.815
CHI2_3D_99 = 11.345
CHI2_3D_999 = 16.266


def _mahal_distance(det_pos: np.ndarray, trk_x: np.ndarray, trk_P: np.ndarray,
                    dim: int = 3) -> float:
    """
    Mahalanobis 距离平方根

    Args:
        det_pos: 检测位置 (3,)
        trk_x: track 状态向量 (6,) 或更长
        trk_P: track 协方差 (6x6) 或更大
        dim: 使用前 dim 维 (默认 3)

    Returns:
        Mahalanobis 距离 sqrt(d^T S^-1 d),S = P[:dim,:dim]
    """
    z = det_pos[:dim] - trk_x[:dim]
    try:
        S = trk_P[:dim, :dim]
        # 加微小子阵避免奇异
        inv_S = np.linalg.inv(S + 1e-6 * np.eye(dim))
        d2 = float(z @ inv_S @ z)
        return float(np.sqrt(max(d2, 0.0)))
    except np.linalg.LinAlgError:
        # fallback 欧氏距离
        return float(np.linalg.norm(z))


def _chi2_gate(distance: float, df: int = 3, confidence: float = 0.95) -> bool:
    """
    Chi-square 门限验证

    Args:
        distance: Mahalanobis 距离
        df: 自由度 (3D = 3)
        confidence: 置信水平 (0.95 = 95%)
    """
    threshold = chi2.ppf(confidence, df)
    return distance ** 2 <= threshold


def _chi2_threshold(df: int = 3, confidence: float = 0.95) -> float:
    """获取 chi2 门限值"""
    return chi2.ppf(confidence, df)


def _build_mahal_cost_matrix(detections: list, tracks: list,
                              gate_chi2: float = CHI2_3D_95,
                              confidence_weighted: bool = True) -> np.ndarray:
    """
    构建 Mahalanobis 距离成本矩阵 (未过滤 = 大数)

    Args:
        detections: list of Detection
        tracks: list of EKFTrack/UKFTrack (有 .kf.x / .kf.P)
        gate_chi2: chi-square 门限 (默认 7.815 = 95%)
        confidence_weighted: 是否除以 confidence 加权

    Returns:
        cost matrix (n_det x n_trk),不通过门限的设为 inf
    """
    n_det = len(detections)
    n_trk = len(tracks)
    cost = np.full((n_det, n_trk), np.inf)
    for i, det in enumerate(detections):
        for j, trk in enumerate(tracks):
            d = _mahal_distance(det.position, trk.kf.x, trk.kf.P)
            # chi-square 门限过滤
            if d ** 2 > gate_chi2:
                continue
            if confidence_weighted:
                conf = max(0.1, getattr(det, 'confidence', 0.5))
                cost[i, j] = d / conf
            else:
                cost[i, j] = d
    return cost


def hungarian_associate(detections: list, tracks: list,
                        gate_threshold: float = 5.0,
                        use_mahalanobis: bool = False,
                        confidence_weighted: bool = True,
                        debug: bool = False,
                        greedy_multi: bool = True,
                        gate_chi2: float = CHI2_3D_95) -> tuple:
    """
    多检测匹配关联
    Args:
        detections: list of Detection
        tracks: list of EKFTrack
        gate_threshold: 门限距离（米）
        use_mahalanobis: 是否使用马氏距离
        confidence_weighted: 置信度加权
        greedy_multi: True=贪心多检测匹配 (一个 track 可被多个 detection 更新)
                       False=匈牙利 1-to-1
        gate_chi2: chi-square 门限 (use_mahalanobis=True 时生效,默认 7.815 = 95%)
    Returns:
        (matched_pairs, unmatched_detections, unmatched_tracks)
        matched_pairs: list of (det_idx, track_idx) - 多个 det 可匹配同一 track
    """
    """
    多检测匹配关联
    Args:
        detections: list of Detection
        tracks: list of EKFTrack
        gate_threshold: 门限距离（米）
        use_mahalanobis: 是否使用马氏距离
        confidence_weighted: 置信度加权
        greedy_multi: True=贪心多检测匹配 (一个 track 可被多个 detection 更新)
                       False=匈牙利 1-to-1
    Returns:
        (matched_pairs, unmatched_detections, unmatched_tracks)
        matched_pairs: list of (det_idx, track_idx) - 多个 det 可匹配同一 track
    """
    n_det = len(detections)
    n_trk = len(tracks)
    if n_det == 0 or n_trk == 0:
        return [], list(range(n_det)), list(range(n_trk))

    # 计算距离矩阵
    distances = np.full((n_det, n_trk), np.inf)
    for i, det in enumerate(detections):
        for j, trk in enumerate(tracks):
            if use_mahalanobis:
                pos = trk.kf.x[:3]
                cov = trk.kf.P[:3, :3]
                try:
                    inv_cov = np.linalg.inv(cov)
                    diff = det.position - pos
                    d = float(np.sqrt(diff @ inv_cov @ diff))
                except np.linalg.LinAlgError:
                    d = float(np.linalg.norm(det.position - pos))
            else:
                d = float(np.linalg.norm(det.position - trk.kf.x[:3]))

            if confidence_weighted:
                conf = max(0.1, getattr(det, 'confidence', 0.5))
                d = d / conf

            distances[i, j] = d

    matched = []
    used_dets = set()
    used_trks = set()

    if greedy_multi:
        # 贪心多检测匹配: 按距离排序，每个 det 分配给最近 track
        # 关键: det 只用一次，但 track 可被多个 det 选中
        candidates = []
        for i in range(n_det):
            for j in range(n_trk):
                if distances[i, j] <= gate_threshold:
                    candidates.append((distances[i, j], i, j))
        candidates.sort()
        for dist, i, j in candidates:
            if i in used_dets:
                continue
            matched.append((i, j))
            used_dets.add(i)
    else:
        # 标准匈牙利 1-to-1
        cost = np.full((n_det, n_trk), gate_threshold + 1.0)
        for i in range(n_det):
            for j in range(n_trk):
                if distances[i, j] <= gate_threshold:
                    cost[i, j] = distances[i, j]
        det_idx, trk_idx = linear_sum_assignment(cost)
        for d, t in zip(det_idx, trk_idx):
            if cost[d, t] <= gate_threshold:
                matched.append((int(d), int(t)))
                used_dets.add(int(d))
                used_trks.add(int(t))

        # P3-B: use_mahalanobis + 1-to-1 模式 - 用 chi2 门限过滤
        if use_mahalanobis:
            cost = _build_mahal_cost_matrix(detections, tracks,
                                              gate_chi2=gate_chi2,
                                              confidence_weighted=confidence_weighted)
            # 过滤 inf
            valid = cost < np.inf
            if not valid.any():
                return [], list(range(n_det)), list(range(n_trk))
            det_idx, trk_idx = linear_sum_assignment(cost)
            for d, t in zip(det_idx, trk_idx):
                if cost[d, t] < np.inf:
                    matched.append((int(d), int(t)))
                    used_dets.add(int(d))
                    used_trks.add(int(t))

    unmatched_dets = [i for i in range(n_det) if i not in used_dets]
    unmatched_trks = [j for j in range(n_trk) if j not in used_trks]

    if debug:
        print(f'  associate: {n_det}d x {n_trk}t  matched={len(matched)}  unmatched_dets={len(unmatched_dets)}')

    return matched, unmatched_dets, unmatched_trks
