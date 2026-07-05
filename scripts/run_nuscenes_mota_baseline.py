"""nuScenes mini MOTA baseline.

跑现有 MultiObjectTracker 在真实 nuScenes mini 数据上, 算 MOTA / MOTP / IDsw.
无噪声 ('gt' mode) baseline, 评估 sensor fusion + tracker pipeline.

使用:
    python3 scripts/run_nuscenes_mota_baseline.py
    python3 scripts/run_nuscenes_mota_baseline.py --scene scene-0061
    python3 scripts/run_nuscenes_mota_baseline.py --scenes 3 --gate 2.0
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# 路径: 把项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.adapters.nuscenes_adapter import NuScenesAdapter  # noqa: E402
from fusion import MultiObjectTracker  # noqa: E402


def evaluate_scene(adapter, scene_name, tracker, gate_m=2.0, max_frames=None):
    """跑一个 scene, 返回该 scene 的 MOTA 累积量.

    Returns:
        dict: {
            'num_frames': int,
            'gt_total': int,           # 该 scene GT 总数 (跨帧)
            'fn': int,                 # False Negatives
            'fp': int,                 # False Positives
            'idsw': int,               # Identity Switches
            'motp': float,             # Mean Position Error (matched GT)
            'match_count': int,        # matched GT-pred 对数
            'gt_traj': Dict[gt_id, pos_list],
            'pred_traj': Dict[track_id, pos_list],
        }
    """
    fn = 0
    fp = 0
    idsw = 0
    matches = []  # (gt_id, track_id) per frame
    last_seen = {}  # gt_id -> track_id (上一帧谁)

    gt_traj = defaultdict(list)
    pred_traj = defaultdict(list)
    num_frames = 0

    for frame in adapter.load_scene(scene_name):
        num_frames += 1
        if max_frames and num_frames > max_frames:
            num_frames -= 1
            break

        gt_for_frame = frame.ground_truth  # List[GroundTruthObj]
        # Run tracker
        tracks = tracker.update(frame.detections_by_sensor, frame.timestamp)

        # Record trajectories (for IDsw detection + post-hoc analysis)
        gt_for_eval = []
        for gt in gt_for_frame:
            gt_for_eval.append({
                "id": gt.object_id,
                "position": np.asarray(gt.position),
            })
            gt_traj[gt.object_id].append(np.asarray(gt.position))

        pred_for_eval = []
        for tr in tracks:
            pred_for_eval.append({
                "id": tr.track_id,
                "position": np.asarray(tr.position),
            })
            pred_traj[tr.track_id].append(np.asarray(tr.position))

        n_gt = len(gt_for_eval)
        n_pred = len(pred_for_eval)

        if n_gt == 0 and n_pred == 0:
            continue

        # Hungarian assignment: distance matrix
        if n_gt == 0:
            fp += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            continue

        # Cost = euclidean distance (m)
        cost = np.zeros((n_gt, n_pred))
        for i, g in enumerate(gt_for_eval):
            for j, p in enumerate(pred_for_eval):
                d = np.linalg.norm(g["position"] - p["position"])
                cost[i, j] = d if d <= gate_m else 1e6

        row, col = linear_sum_assignment(cost)

        # Count matched / unmatched
        matched_gt = set()
        matched_pred = set()
        frame_matches = []
        for r, c in zip(row, col):
            if cost[r, c] <= gate_m:
                gt_id = gt_for_eval[r]["id"]
                tr_id = pred_for_eval[c]["id"]
                matched_gt.add(r)
                matched_pred.add(c)
                frame_matches.append((gt_id, tr_id))
                # Distance for MOTP
                matches.append(cost[r, c])
                # IDsw: same GT, different track_id than previous frame
                if gt_id in last_seen and last_seen[gt_id] != tr_id:
                    idsw += 1
                last_seen[gt_id] = tr_id

        fn += n_gt - len(matched_gt)
        fp += n_pred - len(matched_pred)

    motp = float(np.mean(matches)) if matches else 0.0

    # GT total = unique gt_ids (across frames) count of "appears in any frame"
    # Actually for CLEAR MOT: GT total = sum over frames of GT count
    gt_total = sum(len(v) for v in gt_traj.values())

    return {
        "scene": scene_name,
        "num_frames": num_frames,
        "gt_total": gt_total,
        "fn": fn,
        "fp": fp,
        "idsw": idsw,
        "motp": motp,
        "match_count": len(matches),
        "n_unique_gt": len(gt_traj),
        "n_unique_pred": len(pred_traj),
    }


def aggregate(results):
    """聚合多个 scene 的 MOTA."""
    total_gt = sum(r["gt_total"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_idsw = sum(r["idsw"] for r in results)
    mota = 1.0 - (total_fn + total_fp + total_idsw) / max(total_gt, 1)

    # MOTP weighted by matches
    weighted_motp = sum(r["motp"] * r["match_count"] for r in results)
    total_matches = sum(r["match_count"] for r in results)
    motp = weighted_motp / max(total_matches, 1)

    return {
        "total_gt": total_gt,
        "total_fn": total_fn,
        "total_fp": total_fp,
        "total_idsw": total_idsw,
        "MOTA": mota,
        "MOTP_m": motp,
        "scenes": len(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot",
        default="/Users/mac/.openclaw/workspace/self-driving-sim/data/nuscenes")
    parser.add_argument("--scenes", type=int, default=3,
        help="评估前 N 个 scene (mini 有 10)")
    parser.add_argument("--max-frames", type=int, default=None,
        help="每个 scene 最多跑 N 帧 (默认全跑)")
    parser.add_argument("--gate", type=float, default=2.0,
        help="匹配距离门限 (m)")
    parser.add_argument("--save", default=None,
        help="结果保存到 JSON 文件 (默认打印到 stdout)")
    parser.add_argument("--no-dedup", action="store_true",
        help="Adapter 不去重: 12 传感器通道 × GT (默认: dedup 关, 1 det/GT)")
    parser.add_argument("--tracker-dt", type=float, default=0.5,
        help="Tracker dt (s), 默认 0.5 (适配 nuScenes 2Hz keyframe)")
    parser.add_argument("--gate-threshold", type=float, default=10.0,
        help="Tracker Mahalanobis gate chi-square (默认 10.0)")
    parser.add_argument("--min-hits", type=int, default=3,
        help="Tracker min_hits_to_confirm (默认 3)")
    parser.add_argument("--max-miss", type=int, default=1,
        help="Tracker max_miss_streak (默认 1, IMM 多传感器下最准确)")
    parser.add_argument("--association-mode", default="hungarian",
        choices=["hungarian", "jpda"],
        help="关联算法 (默认 hungarian; jpda 密集场景更稳健)")
    parser.add_argument("--use-imm", action="store_true", default=True,
        help="IMM (CV+CA 交互多模型) 代替单一 EKF (默认开)")
    args = parser.parse_args()

    print(f"=== nuScenes mini MOTA Baseline ===")
    print(f"dataroot = {args.dataroot}")
    print(f"scenes  = {args.scenes}")
    print(f"gate    = {args.gate} m")
    print(f"max_frames per scene = {args.max_frames or 'all'}")
    print(f"deduplicate = {not args.no_dedup}")
    print(f"tracker dt = {args.tracker_dt}s")
    print(f"tracker gate_threshold = {args.gate_threshold}")
    print()

    # gt mode = 无噪声, 每个 GT → 完美 detection
    adapter = NuScenesAdapter(args.dataroot, version="v1.0-mini", mode="gt",
        verbose=False, deduplicate=not args.no_dedup)

    scene_names = adapter.list_scenes()[:args.scenes]
    print(f"selected scenes: {scene_names}")
    print()

    tracker_kwargs = dict(
        dt=args.tracker_dt,
        gate_threshold=args.gate_threshold,
        min_hits_to_confirm=args.min_hits,
        max_miss_streak=args.max_miss,
        use_confidence_weighted=True,
        association_mode=args.association_mode,
        use_imm=args.use_imm,
    )

    results = []
    t_start = time.time()
    for i, scene in enumerate(scene_names):
        # 每个 scene 用全新 tracker (避免 cross-scene ID 污染)
        tracker = MultiObjectTracker(**tracker_kwargs)
        # Reset tracker 内部 history (just in case)
        try:
            tracker.reset()
        except AttributeError:
            pass

        print(f"[{i+1}/{len(scene_names)}] {scene} ...", end=" ", flush=True)
        r = evaluate_scene(adapter, scene, tracker, gate_m=args.gate,
                            max_frames=args.max_frames)
        results.append(r)
        print(f"frames={r['num_frames']}, GT={r['gt_total']}, "
              f"FN={r['fn']}, FP={r['fp']}, IDsw={r['idsw']}, "
              f"MOTP={r['motp']:.2f}m")

    agg = aggregate(results)
    elapsed = time.time() - t_start

    print()
    print(f"=== 聚合 ({len(results)} 个 scene, {elapsed:.1f}s) ===")
    print(f"  GT total: {agg['total_gt']}")
    print(f"  FN:       {agg['total_fn']}")
    print(f"  FP:       {agg['total_fp']}")
    print(f"  IDsw:     {agg['total_idsw']}")
    print(f"  MOTA:     {agg['MOTA']:.4f}  ({(1 - agg['MOTA']) * 100:.1f}% 错误)")
    print(f"  MOTP:     {agg['MOTP_m']:.4f} m")

    if args.save:
        out = {
            "config": vars(args),
            "tracker_kwargs": tracker_kwargs,
            "scenes": results,
            "aggregate": agg,
            "elapsed_s": elapsed,
        }
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
