"""
传感器独立面板 - 显示每个传感器的原始数据
- LiDAR: 3D 点云 (俯视 + 3D)
- Radar: 极坐标距离-多普勒 + 俯视检测点
- Camera: 2D 图像 + 2D bbox
- IMU: 加速度/角速度时序
- GPS: 位置误差
"""
import numpy as np
import plotly.graph_objects as go
from core.data_types import SimFrame


def render_lidar_panel(frame: SimFrame, sensor_id: str = "lidar_top",
                      view: str = "bev", data_in_ego_centric: bool = False) -> go.Figure:
    """
    LiDAR 独立面板

    Args:
        frame: SimFrame
        sensor_id: LiDAR 传感器 ID
        view: "bev" 俯视 / "3d" 三维
        data_in_ego_centric: True=points 已是 lidar frame (ego-centric) 坐标 (nuScenes adapter),
                             不再做 ego_pos 减法
                             False=global 坐标 (仿真), 做 points - ego_pos 转 ego-centric
    """
    lidars = frame.get_lidars()
    if sensor_id not in lidars:
        # 用第一个可用的
        if not lidars:
            return _empty_panel("No LiDAR data")
        sensor_id = list(lidars.keys())[0]
    scan = lidars[sensor_id]
    points = scan.points
    intensity = scan.intensity

    # 转到自车中心
    if points is None or len(points) == 0:
        return _empty_panel(f"{sensor_id}: no points")

    if data_in_ego_centric:
        # nuScenes adapter: lidar points 已是 lidar frame (在车上 1.7m) ≈ ego frame
        # xy 平面一致, 不需要减 ego_pos (否则会双减, 跑到 1000m 外)
        rel_pts = points
    else:
        # 仿真: points 是 global, 转 ego-centric
        ego_pos = frame.ego_state.position
        rel_pts = points - ego_pos
    dets = scan.detections

    if view == "bev":
        fig = go.Figure()
        # 点云 (灰)
        if len(rel_pts) > 0:
            if len(rel_pts) > 5000:
                idx = np.random.choice(len(rel_pts), 5000, replace=False)
                rel_pts = rel_pts[idx]
                if intensity is not None:
                    intensity = intensity[idx]
            # 强度着色 (size 4 让 34752 点云清晰可辨)
            if intensity is not None:
                fig.add_trace(go.Scatter(
                    x=rel_pts[:, 0], y=rel_pts[:, 1],
                    mode='markers',
                    marker=dict(size=4, color=intensity, colorscale='Greys',
                                showscale=True, colorbar=dict(title='Intensity', x=1.0)),
                    name='Points',
                    hovertemplate='x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{customdata[0]:.1f}',
                    customdata=rel_pts[:, 2:3] if rel_pts.shape[1] >= 3 else None,
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=rel_pts[:, 0], y=rel_pts[:, 1],
                    mode='markers',
                    marker=dict(size=4, color='lightgray'),
                    name='Points',
                ))
        # 检测目标
        for i, det in enumerate(dets):
            if data_in_ego_centric:
                pos = det.position
            else:
                pos = det.position - ego_pos
            fig.add_trace(go.Scatter(
                x=[pos[0]], y=[pos[1]], mode='markers+text',
                marker=dict(size=12, color='red', symbol='square', line=dict(color='darkred', width=2)),
                text=[f"L{i+1}"], textposition='top center',
                textfont=dict(size=10, color='darkred'),
                name='LiDAR Detection',
                hovertemplate=f"LiDAR Det {i+1}<br>Range: {det.attributes.get('range_m', '?')}m<br>Points: {det.attributes.get('num_points', '?')}",
            ))
        # 自车
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text',
            marker=dict(size=18, color='cyan', symbol='diamond', line=dict(color='blue', width=2)),
            text=['EGO'], textfont=dict(size=12, color='blue'),
            textposition='top center', name='Ego'))
        # 距离圈
        for r in [25, 50, 75]:
            theta = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(
                x=r*np.cos(theta), y=r*np.sin(theta),
                mode='lines', line=dict(color='gray', width=0.5, dash='dot'),
                showlegend=False, hoverinfo='skip',
            ))
        fig.update_layout(
            title=f"{sensor_id} | {len(points)} points | {len(dets)} detections | Ego-centered BEV",
            xaxis=dict(title='X (Front) m', range=[-50, 80], scaleanchor='y'),
            yaxis=dict(title='Y (Left) m', range=[-50, 50]),
            height=400, width=600,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    elif view == "3d":
        fig = go.Figure()
        if len(rel_pts) > 2000:
            idx = np.random.choice(len(rel_pts), 2000, replace=False)
            rel_pts = rel_pts[idx]
        fig.add_trace(go.Scatter3d(
            x=rel_pts[:, 0], y=rel_pts[:, 1], z=rel_pts[:, 2] if rel_pts.shape[1] > 2 else np.zeros(len(rel_pts)),
            mode='markers', marker=dict(size=3, color='lightgray', opacity=0.5),
            name='Points', hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode='markers+text',
            marker=dict(size=12, color='cyan', symbol='diamond'),
            text=['EGO'], textposition='top center',
            name='Ego',
        ))
        fig.update_layout(
            title=f"{sensor_id} 3D | {len(points)} points",
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                       aspectmode='data'),
            height=400, margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig
    else:
        return _empty_panel(f"Unknown view: {view}")


def render_radar_panel(frame: SimFrame, sensor_id: str = "radar_front",
                      data_in_ego_centric: bool = False) -> go.Figure:
    """
    Radar 独立面板 - 距离-多普勒 + 方位角

    Args:
        data_in_ego_centric: True=detection 已是 ego-centric (nuScenes), 不减 ego_pos
    """
    radars = frame.get_radars()
    if sensor_id not in radars:
        if not radars:
            return _empty_panel("No Radar data")
        sensor_id = list(radars.keys())[0]
    track = radars[sensor_id]
    dets = track.detections

    if not dets:
        return _empty_panel(f"{sensor_id}: no detections")

    fig = go.Figure()

    # 自车
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers+text',
        marker=dict(size=18, color='cyan', symbol='diamond', line=dict(color='blue', width=2)),
        text=['EGO'], textfont=dict(size=12, color='blue'),
        textposition='top center', name='Ego',
    ))

    # 雷达检测 (极坐标 → 笛卡尔, 距离 vs 多普勒)
    ranges = []
    dopplers = []
    positions_x = []
    positions_y = []
    for i, det in enumerate(dets):
        rel = det.position
        r = det.attributes.get('range_m', float(np.linalg.norm(rel)))
        d = det.attributes.get('doppler_mps', 0.0)
        az = det.attributes.get('azimuth_deg', 0.0)
        # 笛卡尔位置
        positions_x.append(rel[0])
        positions_y.append(rel[1])
        ranges.append(r)
        dopplers.append(d)

    # 散点 (按距离半径着色)
    fig.add_trace(go.Scatter(
        x=positions_x, y=positions_y,
        mode='markers+text',
        marker=dict(
            size=12, color=ranges, colorscale='Reds',
            showscale=True, colorbar=dict(title='Range (m)', x=1.0),
            symbol='circle', line=dict(color='darkred', width=2),
        ),
        text=[f"R{i+1}<br>{r:.0f}m<br>{d:.1f}m/s" for i, (r, d) in enumerate(zip(ranges, dopplers))],
        textposition='top center',
        textfont=dict(size=9, color='darkred'),
        name='Radar Detection',
        hovertemplate='Range: %{marker.color:.0f}m<br>Doppler: %{customdata:.1f}m/s<br>Az: %{text}<extra></extra>',
        customdata=dopplers,
    ))

    # 距离圈
    for r in [50, 100, 150]:
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=r*np.cos(theta), y=r*np.sin(theta),
            mode='lines', line=dict(color='orange', width=0.5, dash='dot'),
            showlegend=False, hoverinfo='skip',
        ))

    # FOV 扇区指示 (假设 ±45°)
    fov_half = 45
    theta_fov = np.linspace(-np.deg2rad(fov_half), np.deg2rad(fov_half), 50)
    r_max = max(ranges) * 1.2 if ranges else 150
    fig.add_trace(go.Scatter(
        x=r_max*np.cos(theta_fov), y=r_max*np.sin(theta_fov),
        mode='lines', line=dict(color='orange', width=2, dash='dash'),
        name='Radar FOV', hoverinfo='skip',
    ))

    fig.update_layout(
        title=f"{sensor_id} | {len(dets)} detections (colored by range)",
        xaxis=dict(title='X (Front) m', range=[-50, 200], scaleanchor='y'),
        yaxis=dict(title='Y (Left) m', range=[-100, 100]),
        height=400, width=600,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def render_camera_panel(frame: SimFrame, sensor_id: str = "camera_front") -> go.Figure:
    """Camera 2D 图像 + bbox"""
    cameras = frame.get_cameras()
    if sensor_id not in cameras:
        if not cameras:
            return _empty_panel("No Camera data")
        sensor_id = list(cameras.keys())[0]
    cam = cameras[sensor_id]
    # 直接显示图像
    fig = go.Figure()
    fig.add_trace(go.Image(z=cam.image))
    # 画 bbox
    for det in cam.detections_2d:
        x1, y1, x2, y2 = det['bbox']
        fig.add_shape(
            type='rect', x0=x1, y0=y1, x1=x2, y1=y2,
            line=dict(color='red', width=2),
        )
        if 'class' in det:
            fig.add_annotation(
                x=x1, y=y1-5, text=det['class'],
                showarrow=False, font=dict(color='red', size=10),
            )
    fig.update_layout(
        title=f"{sensor_id} | {len(cam.detections_2d)} 2D detections",
        xaxis=dict(title='px', range=[0, cam.image.shape[1]]),
        yaxis=dict(title='px', range=[cam.image.shape[0], 0], scaleanchor='x'),
        height=400, width=600,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def render_imu_panel(frame: SimFrame, sensor_id: str = "imu") -> go.Figure:
    """IMU 加速度 + 角速度"""
    imus = frame.get_imus()
    if sensor_id not in imus:
        if not imus:
            return _empty_panel("No IMU data")
        sensor_id = list(imus.keys())[0]
    imu = imus[sensor_id]
    fig = go.Figure()
    labels = ['x', 'y', 'z']
    accel = imu.attributes.get('accel', [0, 0, 0])
    gyro = imu.attributes.get('gyro', [0, 0, 0])
    fig.add_trace(go.Bar(name='Accel (m/s²)', x=labels, y=accel, marker_color='blue'))
    fig.add_trace(go.Bar(name='Gyro (rad/s)', x=labels, y=gyro, marker_color='green'))
    fig.update_layout(
        title=f"{sensor_id} | Accel/Gyro at t={frame.timestamp:.2f}s",
        barmode='group', height=300,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def render_gps_panel(frame: SimFrame, sensor_id: str = "gps") -> go.Figure:
    """GPS 位置误差"""
    gpses = frame.get_gpses()
    if sensor_id not in gpses:
        if not gpses:
            return _empty_panel("No GPS data")
        sensor_id = list(gpses.keys())[0]
    gps = gpses[sensor_id]
    ego_pos = frame.ego_state.position
    err = np.linalg.norm(gps.position - ego_pos)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['X', 'Y', 'Z', 'Total'],
        y=[abs(gps.position[i] - ego_pos[i]) for i in range(3)] + [err],
        marker_color=['blue', 'green', 'red', 'orange'],
        text=[f"{abs(gps.position[i] - ego_pos[i]):.2f}m" for i in range(3)] + [f"{err:.2f}m"],
        textposition='outside',
    ))
    fig.update_layout(
        title=f"{sensor_id} | Position error at t={frame.timestamp:.2f}s (total: {err:.2f}m)",
        yaxis_title='Error (m)', height=300,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _empty_panel(msg: str) -> go.Figure:
    """空面板占位"""
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, xref='paper', yref='paper',
        text=msg, showarrow=False,
        font=dict(size=14, color='gray'),
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=200, plot_bgcolor='#fafafa',
    )
    return fig
