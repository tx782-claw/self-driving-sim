"""
可视化 - 优先 Plotly (Web), 备选 matplotlib (CLI/无 GUI)
"""
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

try:
    import matplotlib
    matplotlib.use('Agg')  # 非 GUI 后端
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ============================================================
# Open3D 工具（高级，可选）
# ============================================================
def build_pcd(points: np.ndarray, color: list = [0.5, 0.5, 0.5]):
    if not HAS_OPEN3D:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd


def build_box(center: np.ndarray, size: np.ndarray, color: list = [1, 0, 0]):
    if not HAS_OPEN3D:
        return None
    l, w, h = size
    cx, cy, cz = center
    pts = [
        [cx - l/2, cy - w/2, cz - h/2], [cx + l/2, cy - w/2, cz - h/2],
        [cx + l/2, cy + w/2, cz - h/2], [cx - l/2, cy + w/2, cz - h/2],
        [cx - l/2, cy - w/2, cz + h/2], [cx + l/2, cy - w/2, cz + h/2],
        [cx + l/2, cy + w/2, cz + h/2], [cx - l/2, cy + w/2, cz + h/2],
    ]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.LineVector(lines)
    ls.paint_uniform_color(color)
    return ls


def build_coordinate_frame(size: float = 2.0):
    if not HAS_OPEN3D:
        return []
    frames = []
    for axis, color in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        endpoints = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        endpoints[1][axis] = size
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(endpoints)
        ls.lines = o3d.utility.LineVector([[0, 1]])
        ls.paint_uniform_color(color)
        frames.append(ls)
    return frames


# ============================================================
# 3D 场景渲染 - 自动选择最佳后端
# ============================================================
def render_frame(ego_pos, ground_truth, tracks, lidar_points=None, title="Frame", out_path=None):
    """
    统一接口: 返回 figure 对象或保存到文件
    优先 Plotly → Matplotlib
    """
    if _HAS_PLOTLY:
        return _render_frame_plotly(ego_pos, ground_truth, tracks, lidar_points, title, out_path)
    elif _HAS_MPL:
        return _render_frame_mpl(ego_pos, ground_truth, tracks, lidar_points, title, out_path)
    else:
        raise RuntimeError("No plotting library available (need plotly or matplotlib)")


def _render_frame_plotly(ego_pos, ground_truth, tracks, lidar_points, title, out_path, ego_centered=True):
    """Plotly 3D 渲染 (默认自车中心)"""
    import plotly.graph_objects as go
    fig = go.Figure()

    if ego_centered:
        # 自车坐标系变换
        ego_render = np.array([0.0, 0.0, 0.0])
        gt_render = [g.position - ego_pos for g in ground_truth]
        trk_render = [t.position - ego_pos for t in tracks]
        if lidar_points is not None and len(lidar_points) > 0:
            lidar_points = np.array([p - ego_pos for p in lidar_points])
        view_range = (-80, 100)
    else:
        ego_render = ego_pos
        gt_render = [g.position for g in ground_truth]
        trk_render = [t.position for t in tracks]
        view_range = None

    fig.add_trace(go.Scatter3d(
        x=[ego_render[0]], y=[ego_render[1]], z=[ego_render[2]],
        mode='markers+text', marker=dict(size=15, color='cyan', symbol='diamond',
                                          line=dict(color='blue', width=3)),
        text=['EGO'], textfont=dict(size=14, color='blue'),
        name='Ego Vehicle', textposition="top center"
    ))

    # GT 红色方块 + 标签
    for i, gt in enumerate(ground_truth):
        g = gt_render[i]
        speed = float(np.linalg.norm(gt.velocity))
        fig.add_trace(go.Scatter3d(
            x=[g[0]], y=[g[1]], z=[g[2]],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='square',
                        line=dict(color='darkred', width=2), opacity=0.7),
            text=[f"GT#{gt.object_id}<br>{speed:.0f}m/s"],
            textfont=dict(size=9, color='darkred'),
            textposition='top center',
            showlegend=False,
            name=f"GT #{gt.object_id}",
            hovertemplate=f"GT #{gt.object_id}<br>Type: {gt.obj_type}<br>Speed: {speed:.1f} m/s<br>pos: {gt.position.tolist()}"
        ))

    # Track 绿色圆点 + 标签
    for i, trk in enumerate(tracks):
        t = trk_render[i]
        speed = float(np.linalg.norm(trk.velocity))
        fig.add_trace(go.Scatter3d(
            x=[t[0]], y=[t[1]], z=[t[2]],
            mode='markers+text',
            marker=dict(size=12, color='lime', symbol='circle',
                        line=dict(color='black', width=2)),
            text=[f"T#{trk.track_id}<br>{speed:.0f}m/s"],
            textfont=dict(size=10, color='darkgreen'),
            textposition='top center',
            name=f'Track #{trk.track_id}',
            hovertemplate=f"Track #{trk.track_id}<br>Age: {trk.age}<br>Hits: {trk.hits}<br>pos: {trk.position.tolist()}"
        ))
    if lidar_points is not None and len(lidar_points) > 0:
        if len(lidar_points) > 2000:
            idx = np.random.choice(len(lidar_points), 2000, replace=False)
            lidar_points = lidar_points[idx]
        fig.add_trace(go.Scatter3d(
            x=lidar_points[:, 0], y=lidar_points[:, 1], z=lidar_points[:, 2],
            mode='markers', marker=dict(size=1.5, color='lightgray', opacity=0.5),
            name='LiDAR', hoverinfo='skip', showlegend=False
        ))

    scene_dict = dict(
        xaxis_title='X (Front) m' if ego_centered else 'X (m)',
        yaxis_title='Y (Left) m' if ego_centered else 'Y (m)',
        zaxis_title='Z (Up) m' if ego_centered else 'Z (m)',
        aspectmode='data',
        camera=dict(eye=dict(x=20, y=-30, z=15))
    )
    if view_range is not None:
        scene_dict['xaxis_range'] = [view_range[0], view_range[1]]
        scene_dict['yaxis_range'] = [-30, 30]
        scene_dict['zaxis_range'] = [-2, 8]

    fig.update_layout(
        title=title,
        scene=scene_dict,
        height=600, margin=dict(l=0, r=0, t=40, b=0),
    )
    if out_path:
        if out_path.endswith('.html'):
            fig.write_html(out_path)
        else:
            fig.write_image(out_path)
    return fig


def _render_frame_mpl(ego_pos, ground_truth, tracks, lidar_points, title, out_path, ego_centered=True):
    """Matplotlib 3D 渲染（fallback，默认自车中心）"""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    if ego_centered:
        ego_render = np.array([0.0, 0.0, 0.0])
        gt_render = [g.position - ego_pos for g in ground_truth]
        trk_render = [t.position - ego_pos for t in tracks]
        if lidar_points is not None and len(lidar_points) > 0:
            lidar_points = np.array([p - ego_pos for p in lidar_points])
    else:
        ego_render = ego_pos
        gt_render = [g.position for g in ground_truth]
        trk_render = [t.position for t in tracks]

    ax.scatter(*ego_render, c='cyan', s=300, marker='D', label='Ego', zorder=10,
               edgecolors='blue', linewidth=2)
    ax.text(0, 0, 3, 'EGO', color='blue', fontsize=12, fontweight='bold', ha='center')

    for i, g in enumerate(gt_render):
        gt_obj = ground_truth[i]
        speed = np.linalg.norm(gt_obj.velocity)
        ax.scatter(*g, c='red', s=100, marker='s', alpha=0.7,
                   edgecolors='darkred', linewidth=1.5, zorder=4)
        ax.text(g[0], g[1], g[2] + 2.0, f"GT#{gt_obj.object_id}",
                color='darkred', fontsize=8, fontweight='bold', ha='center', zorder=11)
        ax.text(g[0], g[1], g[2] - 2.0, f"{np.linalg.norm(g):.0f}m",
                color='red', fontsize=7, ha='center', alpha=0.8, zorder=11)

    for i, t in enumerate(trk_render):
        trk_obj = tracks[i]
        speed = np.linalg.norm(trk_obj.velocity)
        ax.scatter(*t, c='lime', s=140, marker='o', edgecolors='black', linewidth=1.5, zorder=5)
        ax.text(t[0], t[1], t[2] + 2.0, f"T#{trk_obj.track_id}",
                color='darkgreen', fontsize=9, fontweight='bold', ha='center', zorder=12)
        ax.text(t[0], t[1], t[2] - 2.0, f"{speed:.0f}m/s",
                color='green', fontsize=7, ha='center', alpha=0.8, zorder=12)
    if lidar_points is not None and len(lidar_points) > 0:
        if len(lidar_points) > 500:
            idx = np.random.choice(len(lidar_points), 500, replace=False)
            lidar_points = lidar_points[idx]
        ax.scatter(lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2],
                   c='lightgray', s=0.5, alpha=0.4)
    if ego_centered:
        ax.set_xlabel('X (Front) m'); ax.set_ylabel('Y (Left) m'); ax.set_zlabel('Z (Up) m')
        ax.set_xlim(-80, 100); ax.set_ylim(-30, 30); ax.set_zlim(-2, 8)
    else:
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(title); ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=80, bbox_inches='tight')
    return fig


# ============================================================
# 2D 俯视图
# ============================================================
def render_topdown(ego_pos, ground_truth, tracks, range_m=80.0, out_path=None, title='Top-Down View'):
    """统一接口"""
    if _HAS_PLOTLY:
        return _render_topdown_plotly(ego_pos, ground_truth, tracks, range_m, out_path, title)
    elif _HAS_MPL:
        return _render_topdown_mpl(ego_pos, ground_truth, tracks, range_m, out_path, title)
    else:
        raise RuntimeError("No plotting library")


def _render_topdown_plotly(ego_pos, ground_truth, tracks, range_m, out_path, title):
    import plotly.graph_objects as go
    fig = go.Figure()
    cx, cy = ego_pos[0], ego_pos[1]
    # 自车
    fig.add_trace(go.Scatter(x=[0], y=[0],
        mode='markers+text',
        marker=dict(size=20, color='cyan', symbol='diamond',
                    line=dict(color='blue', width=2)),
        text=['EGO'], textposition='top center',
        textfont=dict(size=14, color='blue'),
        name='Ego'))
    for gt in ground_truth:
        rx, ry = gt.position[0] - cx, gt.position[1] - cy
        if abs(rx) > range_m or abs(ry) > range_m:
            continue
        color = {'car': 'red', 'truck': 'darkred', 'pedestrian': 'orange'}.get(gt.obj_type, 'red')
        speed = float(np.linalg.norm(gt.velocity))
        fig.add_trace(go.Scatter(x=[rx], y=[ry],
            mode='markers+text',
            marker=dict(size=12, color=color, symbol='square',
                        line=dict(color='darkred', width=2)),
            text=[f"GT#{gt.object_id}<br>{speed:.0f}m/s"],
            textposition='top center',
            textfont=dict(size=9, color='darkred'),
            name=f'GT #{gt.object_id}',
            hovertemplate=f"GT#{gt.object_id}<br>{gt.obj_type}<br>{speed:.1f} m/s"))
    for trk in tracks:
        rx, ry = trk.position[0] - cx, trk.position[1] - cy
        if abs(rx) > range_m or abs(ry) > range_m:
            continue
        speed = float(np.linalg.norm(trk.velocity))
        fig.add_trace(go.Scatter(x=[rx], y=[ry],
            mode='markers+text',
            marker=dict(size=14, color='lime', symbol='circle',
                        line=dict(color='black', width=2)),
            text=[f"T#{trk.track_id}<br>{speed:.0f}m/s"],
            textposition='top center',
            textfont=dict(size=10, color='darkgreen'),
            name=f'Track #{trk.track_id}',
            hovertemplate=f"Track #{trk.track_id}<br>Age: {trk.age}<br>Hits: {trk.hits}"))
    fig.update_layout(
        title=title + ' [Ego-centered]',
        xaxis_title='X (Front) m', yaxis_title='Y (Left) m',
        xaxis=dict(range=[-range_m, range_m], zeroline=False),
        yaxis=dict(range=[-range_m/2, range_m/2], zeroline=False),
        height=600, width=900, yaxis_scaleanchor='x',
    )
    if out_path:
        if out_path.endswith('.html'):
            fig.write_html(out_path)
        else:
            fig.write_image(out_path)
    return fig


def _render_topdown_mpl(ego_pos, ground_truth, tracks, range_m, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 7))
    # 计算自车中心坐标
    cx, cy = ego_pos[0], ego_pos[1]
    ax.scatter([0], [0], c='cyan', s=300, marker='D', label='Ego', zorder=10,
               edgecolors='blue', linewidth=2)
    ax.annotate('EGO', (0, 0), textcoords='offset points', xytext=(0, 12),
                fontsize=11, fontweight='bold', color='blue', ha='center')
    for gt in ground_truth:
        rx, ry = gt.position[0] - cx, gt.position[1] - cy
        if abs(rx) > range_m or abs(ry) > range_m:
            continue
        color = {'car': 'red', 'truck': 'darkred', 'pedestrian': 'orange'}.get(gt.obj_type, 'red')
        speed = np.linalg.norm(gt.velocity)
        ax.scatter([rx], [ry], c=color, s=120, marker='s', alpha=0.7,
                   edgecolors='darkred', linewidth=1.5, zorder=4)
        # GT 标签
        ax.annotate(f"GT#{gt.object_id}", (rx, ry), textcoords='offset points',
                    xytext=(0, 12), fontsize=9, fontweight='bold', color='darkred', ha='center')
        ax.annotate(f"{speed:.0f}m/s", (rx, ry), textcoords='offset points',
                    xytext=(0, -15), fontsize=7, color='red', ha='center')
    for trk in tracks:
        rx, ry = trk.position[0] - cx, trk.position[1] - cy
        if abs(rx) > range_m or abs(ry) > range_m:
            continue
        speed = np.linalg.norm(trk.velocity)
        ax.scatter([rx], [ry], c='lime', s=150, marker='o', edgecolors='black', linewidth=1.5, zorder=5)
        # Track 标签
        ax.annotate(f"T#{trk.track_id}", (rx, ry), textcoords='offset points',
                    xytext=(0, 12), fontsize=10, fontweight='bold', color='darkgreen', ha='center')
        ax.annotate(f"{speed:.0f}m/s", (rx, ry), textcoords='offset points',
                    xytext=(0, -15), fontsize=7, color='green', ha='center')
    ax.set_xlabel('X (Front) m'); ax.set_ylabel('Y (Left) m')
    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m/2, range_m/2)
    ax.set_title(title + ' [Ego-centered]')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    # 加十字线标出自车位置
    ax.axhline(y=0, color='cyan', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='cyan', linestyle='--', alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=80, bbox_inches='tight')
    return fig


# 兼容旧 API
def render_frame_plotly(ego_pos, ground_truth, tracks, lidar_points=None, title="Frame"):
    return _render_frame_plotly(ego_pos, ground_truth, tracks, lidar_points, title, None)


def render_topdown_plotly(ego_pos, ground_truth, tracks, range_m=80.0):
    return _render_topdown_plotly(ego_pos, ground_truth, tracks, range_m, None, "Top-Down View")
