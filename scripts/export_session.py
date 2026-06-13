"""
数据回放与导出
- 单帧 JSON 导出
- 全场 JSON 导出 (所有帧 + 元数据)
- CSV 格式 (适合 Excel/分析)
- 从 JSON 重新加载回放
"""
import os
import json
import csv
from typing import List
from core.data_types import SimFrame


def frame_to_dict(frame: SimFrame) -> dict:
    """单帧转 dict"""
    return {
        'timestamp': float(frame.timestamp),
        'ego': {
            'position': frame.ego_state.position.tolist(),
            'velocity': frame.ego_state.velocity.tolist(),
            'heading': float(frame.ego_state.heading),
        },
        'ground_truth': [
            {
                'id': int(gt.object_id),
                'type': gt.obj_type,
                'position': gt.position.tolist(),
                'velocity': gt.velocity.tolist(),
                'size': gt.size.tolist(),
                'color': gt.color,
            }
            for gt in frame.ground_truth
        ],
        'tracks': [
            {
                'id': int(t.track_id),
                'position': t.position.tolist(),
                'velocity': t.velocity.tolist(),
                'age': t.age,
                'hits': t.hits,
                'sources': sorted(list(t.source_sensors)),
            }
            for t in frame.tracks
        ],
        'lidar_points': len(frame.lidar_data.points) if frame.lidar_data else 0,
    }


def export_session_json(frames: List[SimFrame], out_path: str,
                        metadata: dict = None) -> str:
    """
    导出整个仿真会话为单个 JSON 文件
    Args:
        frames: 所有帧
        out_path: 输出路径
        metadata: 场景/参数等元数据
    Returns:
        输出路径
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    data = {
        'metadata': metadata or {},
        'frame_count': len(frames),
        'duration_s': float(frames[-1].timestamp) if frames else 0.0,
        'frames': [frame_to_dict(f) for f in frames],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out_path


def export_tracks_csv(frames: List[SimFrame], out_path: str) -> str:
    """
    导出 track 数据为 CSV (适合 Excel/Pandas 分析)
    列: timestamp, track_id, x, y, z, vx, vy, vz, hits, age, sources
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'track_id', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                         'hits', 'age', 'sources'])
        for frame in frames:
            for t in frame.tracks:
                p = t.position
                v = t.velocity
                writer.writerow([
                    f'{frame.timestamp:.3f}',
                    t.track_id,
                    f'{p[0]:.3f}', f'{p[1]:.3f}', f'{p[2]:.3f}',
                    f'{v[0]:.3f}', f'{v[1]:.3f}', f'{v[2]:.3f}',
                    t.hits, t.age, '|'.join(sorted(t.source_sensors)),
                ])
    return out_path


def export_groundtruth_csv(frames: List[SimFrame], out_path: str) -> str:
    """导出 ground truth 为 CSV"""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'gt_id', 'type', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        for frame in frames:
            for gt in frame.ground_truth:
                p = gt.position
                v = gt.velocity
                writer.writerow([
                    f'{frame.timestamp:.3f}',
                    gt.object_id, gt.obj_type,
                    f'{p[0]:.3f}', f'{p[1]:.3f}', f'{p[2]:.3f}',
                    f'{v[0]:.3f}', f'{v[1]:.3f}', f'{v[2]:.3f}',
                ])
    return out_path


def load_session_json(path: str) -> dict:
    """
    加载之前导出的 session JSON
    Returns:
        {
            'metadata': {...},
            'frame_count': N,
            'frames': [{...}, ...]
        }
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
