"""
数据关联 - 多检测匹配（贪心 N-to-1）
关键修复: 多个检测可以更新同一个 track（多传感器融合的核心）
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_associate(detections: list, tracks: list,
                        gate_threshold: float = 5.0,
                        use_mahalanobis: bool = False,
                        confidence_weighted: bool = True,
                        debug: bool = False,
                        greedy_multi: bool = True) -> tuple:
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

    unmatched_dets = [i for i in range(n_det) if i not in used_dets]
    unmatched_trks = [j for j in range(n_trk) if j not in used_trks]

    if debug:
        print(f'  associate: {n_det}d x {n_trk}t  matched={len(matched)}  unmatched_dets={len(unmatched_dets)}')

    return matched, unmatched_dets, unmatched_trks
