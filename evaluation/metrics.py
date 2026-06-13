"""
评估指标 - 优化版
"""
import numpy as np
from typing import List
from core.data_types import SimFrame


def compute_rmse(frames: List[SimFrame]) -> dict:
    """位置/速度 RMSE"""
    pos_errors = []
    vel_errors = []

    for frame in frames:
        gt_objs = frame.ground_truth
        tracks = frame.tracks
        if not gt_objs or not tracks:
            continue

        # 构建距离矩阵，贪心匹配
        gt_to_track = _greedy_match(gt_objs, tracks, max_dist=5.0)
        for gt_id, trk in gt_to_track.items():
            gt = next((g for g in gt_objs if g.object_id == gt_id), None)
            if gt is not None:
                pos_errors.append(np.linalg.norm(trk.position - gt.position))
                vel_errors.append(np.linalg.norm(trk.velocity - gt.velocity))

    result = {}
    if pos_errors:
        result['position_rmse_m'] = float(np.sqrt(np.mean(np.array(pos_errors) ** 2)))
        result['position_mae_m'] = float(np.mean(pos_errors))
        result['position_max_m'] = float(np.max(pos_errors))
    if vel_errors:
        result['velocity_rmse_mps'] = float(np.sqrt(np.mean(np.array(vel_errors) ** 2)))
        result['velocity_mae_mps'] = float(np.mean(vel_errors))
    result['n_samples'] = len(pos_errors)
    return result


def _greedy_match(gt_objs, tracks, max_dist=5.0):
    """贪心最近匹配，返回 {gt_id: track}"""
    pairs = []
    for i, gt in enumerate(gt_objs):
        for j, trk in enumerate(tracks):
            d = np.linalg.norm(trk.position - gt.position)
            if d < max_dist:
                pairs.append((d, i, j))
    pairs.sort(key=lambda x: x[0])
    used_gt = set()
    used_trk = set()
    result = {}
    for d, i, j in pairs:
        if i in used_gt or j in used_trk:
            continue
        result[gt_objs[i].object_id] = tracks[j]
        used_gt.add(i)
        used_trk.add(j)
    return result


def compute_tracking_stats(frames: List[SimFrame]) -> dict:
    """跟踪精度/召回/ID 切换"""
    total_gt = 0
    total_matched = 0
    total_tracks = 0
    total_false_tracks = 0
    id_switches = 0

    # 跟踪每个 GT 在过去帧的 track_id 分配
    prev_assignments = {}

    for frame in frames:
        gt_objs = frame.ground_truth
        tracks = frame.tracks
        total_gt += len(gt_objs)
        total_tracks += len(tracks)

        cur_assignments = {}
        gt_to_trk = _greedy_match(gt_objs, tracks, max_dist=3.0)
        for gt_id, trk in gt_to_trk.items():
            cur_assignments[gt_id] = trk.track_id
            total_matched += 1
            if gt_id in prev_assignments and prev_assignments[gt_id] != trk.track_id:
                id_switches += 1

        # false tracks
        matched_track_ids = set(cur_assignments.values())
        total_false_tracks += sum(1 for t in tracks if t.track_id not in matched_track_ids)
        prev_assignments = cur_assignments

    return {
        'tracking_precision': total_matched / max(1, total_tracks),
        'tracking_recall': total_matched / max(1, total_gt),
        'false_track_rate': total_false_tracks / max(1, total_tracks),
        'id_switches': id_switches,
        'total_gt': total_gt,
        'total_tracks': total_tracks,
    }


def compute_latency(frames: List[SimFrame]) -> dict:
    """时延统计"""
    times = []
    for f in frames:
        if hasattr(f, 'processing_ms'):
            times.append(f.processing_ms)
    if not times:
        return {}
    return {
        'avg_latency_ms': float(np.mean(times)),
        'p95_latency_ms': float(np.percentile(times, 95)),
        'max_latency_ms': float(np.max(times)),
    }


def compute_mot_metrics(frames: List[SimFrame]) -> dict:
    """MOTA / MOTP 标准 MOT 指标"""
    total_gt = 0
    total_fp = 0  # false positives
    total_fn = 0  # false negatives
    total_idsw = 0
    total_dist = 0
    total_matches = 0

    prev_assignments = {}

    for frame in frames:
        gt_objs = frame.ground_truth
        tracks = frame.tracks
        total_gt += len(gt_objs)

        cur_assignments = {}
        gt_to_trk = _greedy_match(gt_objs, tracks, max_dist=3.0)
        for gt_id, trk in gt_to_trk.items():
            cur_assignments[gt_id] = trk.track_id
            total_matches += 1
            gt = next((g for g in gt_objs if g.object_id == gt_id), None)
            if gt is not None:
                total_dist += np.linalg.norm(trk.position - gt.position)
            if gt_id in prev_assignments and prev_assignments[gt_id] != trk.track_id:
                total_idsw += 1
        # false positives
        matched_track_ids = set(cur_assignments.values())
        total_fp += sum(1 for t in tracks if t.track_id not in matched_track_ids)
        # false negatives
        total_fn += len(gt_objs) - len(gt_to_trk)
        prev_assignments = cur_assignments

    if total_gt == 0:
        return {}
    mota = 1.0 - (total_fp + total_fn + total_idsw) / total_gt
    motp = total_dist / max(1, total_matches)
    return {
        'MOTA': float(mota),
        'MOTP_m': float(motp),
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_idsw': total_idsw,
    }


def evaluate(frames: List[SimFrame]) -> dict:
    """综合评估"""
    rmse = compute_rmse(frames)
    stats = compute_tracking_stats(frames)
    mot = compute_mot_metrics(frames)
    lat = compute_latency(frames)
    return {**rmse, **stats, **mot, **lat}


def compute_per_frame_metrics(frames: List[SimFrame]) -> dict:
    """
    逐帧评估指标 - 用于时序图
    Returns:
        {
            'timestamps': [...],
            'pos_rmse': [...],   # 帧内平均位置误差
            'pos_max': [...],    # 帧内最大位置误差
            'vel_rmse': [...],
            'n_tracks': [...],   # 帧内跟踪数
            'n_gt': [...],       # 帧内 GT 数
            'n_matched': [...],  # 帧内成功匹配数
        }
    """
    timestamps = []
    pos_rmse_per_frame = []
    pos_max_per_frame = []
    vel_rmse_per_frame = []
    n_tracks_list = []
    n_gt_list = []
    n_matched_list = []

    for frame in frames:
        timestamps.append(frame.timestamp)
        n_tracks_list.append(len(frame.tracks))
        n_gt_list.append(len(frame.ground_truth))

        gt_objs = frame.ground_truth
        tracks = frame.tracks
        if not gt_objs or not tracks:
            pos_rmse_per_frame.append(0.0)
            pos_max_per_frame.append(0.0)
            vel_rmse_per_frame.append(0.0)
            n_matched_list.append(0)
            continue

        gt_to_trk = _greedy_match(gt_objs, tracks, max_dist=5.0)
        pos_errs = []
        vel_errs = []
        for gt_id, trk in gt_to_trk.items():
            gt = next((g for g in gt_objs if g.object_id == gt_id), None)
            if gt is not None:
                pos_errs.append(np.linalg.norm(trk.position - gt.position))
                vel_errs.append(np.linalg.norm(trk.velocity - gt.velocity))
        n_matched_list.append(len(pos_errs))
        if pos_errs:
            pos_rmse_per_frame.append(float(np.sqrt(np.mean(np.array(pos_errs) ** 2))))
            pos_max_per_frame.append(float(np.max(pos_errs)))
        else:
            pos_rmse_per_frame.append(0.0)
            pos_max_per_frame.append(0.0)
        if vel_errs:
            vel_rmse_per_frame.append(float(np.sqrt(np.mean(np.array(vel_errs) ** 2))))
        else:
            vel_rmse_per_frame.append(0.0)

    return {
        'timestamps': timestamps,
        'pos_rmse': pos_rmse_per_frame,
        'pos_max': pos_max_per_frame,
        'vel_rmse': vel_rmse_per_frame,
        'n_tracks': n_tracks_list,
        'n_gt': n_gt_list,
        'n_matched': n_matched_list,
    }
