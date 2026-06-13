"""
BEV 鸟瞰融合可视化 (v0.2.2 优化E)
=====================================
单图叠加多传感器信息：
- 灰度背景：LiDAR 强度 (Birds Eye View)
- 蓝色等高线：Radar RCS（雷达散射截面）
- 绿色框：Camera 2D bbox 投影
- 红色轨迹：融合 track + 协方差椭圆
- 自车：青色六边形
- GT：黄色叉

设计目标：一张图展示 fusion 价值，替代多个独立 panel
"""
import numpy as np
import plotly.graph_objects as go


def _ego_centric(points, ego_pos):
    """points 转到 ego 中心坐标系"""
    if len(points) == 0:
        return points
    return points - ego_pos


def _ego_centric_3d(positions, ego_pos):
    """list of (x,y,z) → ego-centered array"""
    if not positions:
        return np.zeros((0, 3))
    arr = np.array([p if hasattr(p, 'shape') else np.array(p) for p in positions])
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr - ego_pos


def _cov_ellipse(pos, cov_2d, n_std=2.0, n_points=40):
    """
    协方差椭圆 (2D)
    pos: (x, y) 中心
    cov_2d: (2, 2) 协方差
    """
    if not np.all(np.isfinite(cov_2d)):
        return None
    try:
        eigvals, eigvecs = np.linalg.eigh(cov_2d)
        if np.any(eigvals <= 0):
            return None
        n_std * np.sqrt(eigvals)  # 2σ 半径
    except np.linalg.LinAlgError:
        return None

    theta = np.linspace(0, 2 * np.pi, n_points)
    ell = np.array([np.cos(theta), np.sin(theta)])
    ell = n_std * eigvecs @ np.diag(np.sqrt(eigvals)) @ ell
    return pos[:2, None] + ell  # (2, n_points)


def render_bev_fusion(ego_pos: np.ndarray,
                      ground_truth: list,
                      tracks: list,
                      lidar_points: np.ndarray = None,
                      lidar_intensity: np.ndarray = None,
                      radar_detections: list = None,
                      camera_detections: list = None,
                      range_m: float = 60.0,
                      title: str = "BEV Fusion",
                      out_path: str = None) -> 'go.Figure':
    """
    BEV 鸟瞰融合视图 (Plotly)

    Args:
        ego_pos: 自车世界坐标
        ground_truth: list of GroundTruthObj
        tracks: list of TrackedObject
        lidar_points: (N, 3) LiDAR 点云
        lidar_intensity: (N,) LiDAR 强度
        radar_detections: list of Detection (Radar)
        camera_detections: list of Detection (Camera, 含 bbox_2d)
        range_m: 显示范围 (前/后/左/右)
    """
    fig = go.Figure()

    # 1. LiDAR 灰度 BEV
    if lidar_points is not None and len(lidar_points) > 0:
        pts = _ego_centric(lidar_points[:, :2], ego_pos[:2])
        # 限 range
        mask = (np.abs(pts[:, 0]) <= range_m) & (np.abs(pts[:, 1]) <= range_m)
        pts_f = pts[mask]
        if lidar_intensity is not None:
            intens = lidar_intensity[mask]
        else:
            intens = np.ones(len(pts_f)) * 0.5
        # 高度作为 z 用 scatter 大小表示
        fig.add_trace(go.Scattergl(
            x=pts_f[:, 0], y=pts_f[:, 1],
            mode='markers',
            marker=dict(
                size=1.5,
                color=intens,
                colorscale='Greys',
                cmin=0.0, cmax=1.0,
                showscale=False,
                opacity=0.5,
            ),
            name='LiDAR',
            hoverinfo='skip',
        ))

    # 2. Radar RCS 等高线（多个 det → 多组小等高线）
    if radar_detections:
        for det in radar_detections:
            pos_ego = det.position[:2] - ego_pos[:2]
            rcs_db = det.attributes.get('rcs_dbsm', 0.0)
            rcs_norm = max(0, min(1, (rcs_db + 20) / 40))  # -20~20 dBsm → 0~1
            # RCS 越大 → 等高线越大（线性 1~3m）
            radius = 1.0 + rcs_norm * 2.0
            theta = np.linspace(0, 2 * np.pi, 30)
            xs = pos_ego[0] + radius * np.cos(theta)
            ys = pos_ego[1] + radius * np.sin(theta)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines',
                line=dict(color='cyan', width=1.5),
                opacity=0.7,
                name='Radar',
                hoverinfo='skip',
                showlegend=False,
            ))

    # 3. Camera bbox 投影
    if camera_detections:
        for det in camera_detections:
            pos_ego = det.position[:2] - ego_pos[:2]
            # 简化：bbox 显示成 2m × 4m 的矩形 (FOV 方向)
            heading = 0.0  # 简化：固定朝前
            box_size = det.attributes.get('size', np.array([4.5, 1.8]))
            if hasattr(box_size, '__len__') and len(box_size) >= 2:
                length, width = box_size[0], box_size[1]
            else:
                length, width = 4.5, 1.8
            # 4 个角
            cx, cy = pos_ego[0], pos_ego[1]
            box = np.array([
                [cx - length/2, cy - width/2],
                [cx + length/2, cy - width/2],
                [cx + length/2, cy + width/2],
                [cx - length/2, cy + width/2],
                [cx - length/2, cy - width/2],
            ])
            fig.add_trace(go.Scatter(
                x=box[:, 0], y=box[:, 1], mode='lines',
                line=dict(color='lime', width=2),
                opacity=0.6,
                name='Camera',
                hoverinfo='skip',
                showlegend=False,
            ))

    # 4. GT (黄色叉)
    for gt in ground_truth:
        pos_ego = gt.position[:2] - ego_pos[:2]
        fig.add_trace(go.Scatter(
            x=[pos_ego[0]], y=[pos_ego[1]],
            mode='markers',
            marker=dict(symbol='x', color='yellow', size=14, line=dict(width=2)),
            name='GT',
            text=[f"GT #{gt.object_id}"],
            hoverinfo='text',
            showlegend=False,
        ))

    # 5. Track (红色圆 + 协方差椭圆)
    for trk in tracks:
        pos_ego = trk.position[:2] - ego_pos[:2]
        fig.add_trace(go.Scatter(
            x=[pos_ego[0]], y=[pos_ego[1]],
            mode='markers',
            marker=dict(symbol='circle', color='red', size=12,
                        line=dict(color='darkred', width=2)),
            name='Track',
            text=[f"Track #{trk.track_id}"],
            hoverinfo='text',
            showlegend=False,
        ))
        # 协方差椭圆
        cov = trk.covariance[:2, :2] if hasattr(trk, 'covariance') and trk.covariance is not None else None
        if cov is not None:
            ell = _cov_ellipse(pos_ego.reshape(2, 1), cov, n_std=2.0)
            if ell is not None:
                fig.add_trace(go.Scatter(
                    x=ell[0], y=ell[1], mode='lines',
                    line=dict(color='red', width=1, dash='dot'),
                    opacity=0.6,
                    hoverinfo='skip',
                    showlegend=False,
                ))
        # ID 标签
        fig.add_annotation(
            x=pos_ego[0], y=pos_ego[1] + 2.5,
            text=f"#{trk.track_id}",
            showarrow=False,
            font=dict(color='darkred', size=10, family='Arial Black'),
        )

    # 6. 自车 (青色菱形)
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(symbol='diamond', color='cyan', size=20,
                    line=dict(color='blue', width=2)),
        text=['EGO'],
        textposition='top center',
        textfont=dict(color='blue', size=12, family='Arial Black'),
        name='Ego',
    ))

    # ── 图例 (手动加一个 trace 当图例) ──
    for label, color, symbol in [
        ('LiDAR 点云', 'gray', 'square'),
        ('Radar RCS', 'cyan', 'circle'),
        ('Camera bbox', 'lime', 'square'),
        ('GT', 'yellow', 'x'),
        ('Track', 'red', 'circle'),
        ('自车', 'cyan', 'diamond'),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(symbol=symbol, color=color, size=10),
            name=label,
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(title='X (前) m', range=[-range_m*0.5, range_m],
                   scaleanchor='y', scaleratio=1, showgrid=True),
        yaxis=dict(title='Y (左) m', range=[-range_m, range_m],
                   showgrid=True),
        width=900, height=700,
        plot_bgcolor='black',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(x=1.02, y=0.5, xanchor='left'),
        hovermode='closest',
    )
    fig.update_xaxes(zeroline=True, zerolinecolor='white', zerolinewidth=1)
    fig.update_yaxes(zeroline=True, zerolinecolor='white', zerolinewidth=1)

    if out_path:
        fig.write_html(out_path)

    return fig
