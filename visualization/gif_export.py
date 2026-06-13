"""
动画导出模块
- GIF (用 PIL/Pillow)
- MP4 (用 imageio-ffmpeg 或 OpenCV)
"""
import os
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from typing import List, Optional
from core.data_types import SimFrame


def render_frame_to_png(frame: SimFrame, ego_pos=None, gt=None, tracks=None,
                        lidar_points=None, title="", out_path=None,
                        ego_centered: bool = True) -> Optional[bytes]:
    """
    渲染单帧为 PNG，返回 bytes 或保存到文件
    Args:
        ego_centered: True=以自车为中心 (推荐) / False=世界坐标
    """
    if ego_pos is None:
        ego_pos = frame.ego_state.position
    if gt is None:
        gt = frame.ground_truth
    if tracks is None:
        tracks = frame.tracks
    if lidar_points is None and frame.lidar_data is not None:
        lidar_points = frame.lidar_data.points

    # 是否变换到自车中心坐标系
    if ego_centered:
        ego_render = np.array([0.0, 0.0, 0.0])
        gt_render = [g.position - ego_pos for g in gt]
        trk_render = [t.position - ego_pos for t in tracks]
        ego_heading = frame.ego_state.heading
        ego_yaw = ego_heading  # 简化: 假设朝向 +X
        cos_y, sin_y = np.cos(-ego_yaw), np.sin(-ego_yaw)
        def rotate_xy(v):
            x, y, z = v
            return np.array([cos_y * x - sin_y * y, sin_y * x + cos_y * y, z])
        gt_render = [rotate_xy(v) for v in gt_render]
        trk_render = [rotate_xy(v) for v in trk_render]
        if lidar_points is not None and len(lidar_points) > 0:
            lidar_points = np.array([rotate_xy(p - ego_pos) for p in lidar_points])
    else:
        ego_render = ego_pos
        gt_render = [g.position for g in gt]
        trk_render = [t.position for t in tracks]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 自车 (大、青色)
    ax.scatter(*ego_render, c='cyan', s=300, marker='D', label='Ego', zorder=10, edgecolors='blue', linewidth=2)
    # 自车标签
    ax.text(0, 0, 3, 'EGO', color='blue', fontsize=12, fontweight='bold', ha='center')

    # GT 红色方块 + 标签
    for i, g in enumerate(gt_render):
        gt_obj = gt[i]
        speed = np.linalg.norm(gt_obj.velocity)
        dist_from_ego = np.linalg.norm(g)  # 相对位置即是距离
        ax.scatter(*g, c='red', s=100, marker='s', alpha=0.7, edgecolors='darkred', linewidth=1.5, zorder=4)
        # GT 标签: GT#5  18m/s  car
        ax.text(g[0], g[1], g[2] + 2.0,
                f"GT#{gt_obj.object_id}",
                color='darkred', fontsize=8, fontweight='bold', ha='center', zorder=11)
        # 距离
        ax.text(g[0], g[1], g[2] - 2.0,
                f"{dist_from_ego:.0f}m",
                color='red', fontsize=7, ha='center', alpha=0.8, zorder=11)

    # Track 绿色圆点 + 标签
    for i, t in enumerate(trk_render):
        trk_obj = tracks[i]
        speed = np.linalg.norm(trk_obj.velocity)
        dist_from_ego = np.linalg.norm(t)
        ax.scatter(*t, c='lime', s=140, marker='o', edgecolors='black', linewidth=1.5, zorder=5)
        # Track 标签: T#1  18m/s
        ax.text(t[0], t[1], t[2] + 2.0,
                f"T#{trk_obj.track_id}",
                color='darkgreen', fontsize=9, fontweight='bold', ha='center', zorder=12)
        # 速度
        ax.text(t[0], t[1], t[2] - 2.0,
                f"{speed:.0f}m/s",
                color='green', fontsize=7, ha='center', alpha=0.8, zorder=12)

    if lidar_points is not None and len(lidar_points) > 0:
        if len(lidar_points) > 300:
            idx = np.random.choice(len(lidar_points), 300, replace=False)
            lidar_points = lidar_points[idx]
        ax.scatter(lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2],
                   c='lightgray', s=0.5, alpha=0.4)

    # 坐标轴标签
    if ego_centered:
        ax.set_xlabel('X (Front) m'); ax.set_ylabel('Y (Left) m'); ax.set_zlabel('Z (Up) m')
    else:
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    # 如果是自车中心，限制视野范围
    if ego_centered:
        # 固定视野范围 ±80m
        ax.set_xlim(-80, 100)  # 前方多一点
        ax.set_ylim(-30, 30)
        ax.set_zlim(-2, 8)

    if out_path:
        plt.savefig(out_path, dpi=70, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        import io
        buf = io.BytesIO()
        plt.savefig(buf, dpi=70, bbox_inches='tight', format='png')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()


def export_gif(frames: List[SimFrame], out_path: str,
               duration_ms: int = 80, max_frames: int = 200,
               progress_callback=None) -> int:
    """
    导出 GIF 动画
    Args:
        frames: SimFrame 列表
        out_path: 输出 .gif 路径
        duration_ms: 每帧时长 (ms)
        max_frames: 最大帧数 (避免太大)
        progress_callback: 可选 callback(进度 0-1)
    Returns:
        实际帧数
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow not installed. Run: pip install Pillow")

    # 降采样
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        sampled = [frames[i] for i in idx]
    else:
        sampled = frames

    images = []
    total = len(sampled)
    for i, frame in enumerate(sampled):
        title = f"t={frame.timestamp:.2f}s | Tracks: {len(frame.tracks)} | GT: {len(frame.ground_truth)}"
        png_bytes = render_frame_to_png(frame, title=title, out_path=None, ego_centered=True)
        img = Image.open(io.BytesIO(png_bytes))
        # 转为 RGB（GIF 不支持 RGBA）
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        images.append(img)
        if progress_callback:
            progress_callback((i + 1) / total * 0.9)  # 90% 渲染，10% 保存

    if not images:
        return 0

    # 保存 GIF
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )

    if progress_callback:
        progress_callback(1.0)
    return len(images)


def export_mp4(frames: List[SimFrame], out_path: str,
               fps: int = 20, max_frames: int = 300) -> int:
    """
    导出 MP4 (用 OpenCV)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python not installed")

    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        sampled = [frames[i] for i in idx]
    else:
        sampled = frames

    if not sampled:
        return 0

    # 用第一帧确定尺寸
    first_png = render_frame_to_png(sampled[0], out_path=None)
    import io
    from PIL import Image
    first_img = Image.open(io.BytesIO(first_png))
    w, h = first_img.size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for frame in sampled:
        title = f"t={frame.timestamp:.2f}s | Tracks: {len(frame.tracks)} | GT: {len(frame.ground_truth)}"
        png_bytes = render_frame_to_png(frame, title=title, out_path=None)
        # PIL → OpenCV (BGR)
        img = Image.open(io.BytesIO(png_bytes)).convert('RGB')
        arr = np.array(img)  # RGB
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    return len(sampled)
