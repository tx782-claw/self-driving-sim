"""
Streamlit Web UI - 自驾驶仿真平台
启动: streamlit run app/webui.py
"""
import sys
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# 路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import Simulator, __version__ as APP_VERSION
from scenarios import HighwayScenario, UrbanScenario, DenseHighwayScenario, JunctionScenario, StopAndGoScenario
from sensors import LidarSensor, RadarSensor, CameraSensor, IMUSensor, GPSSensor
from fusion import MultiObjectTracker
from evaluation import evaluate, compute_per_frame_metrics
from visualization import render_frame_plotly, render_topdown_plotly, export_gif
from scenarios.weather import PRESETS as WEATHER_PRESETS, get_weather_by_name

st.set_page_config(
    page_title="🚗 Self-Driving Sim",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Self-Driving Sensor Simulation & Fusion Platform")
st.caption(f"v{APP_VERSION} · 本地仿真 · 多传感器融合 · 3D 可视化 · 时间回放")

# ===== 侧边栏 =====
with st.sidebar:
    st.header("⚙️ 参数配置")

    # 数据源选择 (v0.3 新增: 真实数据集回归)
    data_source = st.radio(
        "📂 数据源",
        ["仿真 (Synthetic)", "真实数据集 (nuScenes mini)"],
        index=0,
        help="仿真: 手写 Highway/Urban/Junction 场景\n真实: nuScenes mini 10 场景 (v0.3+ 回归)",
    )

    if data_source.startswith("真实"):
        # nuScenes mini 配置
        NUSCENES_DATAROOT = "/Users/mac/.openclaw/workspace/self-driving-sim/data/nuscenes"
        import os
        if os.path.isdir(NUSCENES_DATAROOT):
            from scripts.adapters.nuscenes_adapter import NuScenesAdapter
            _ns_adapter = NuScenesAdapter(NUSCENES_DATAROOT, mode="gt", verbose=False)
            ns_scenes = _ns_adapter.list_scenes()
            ns_scene = st.selectbox("场景 (10 个可用)", ns_scenes, index=0)
            del _ns_adapter
        else:
            st.error(f"找不到 {NUSCENES_DATAROOT}, 请先跑 nuScenes mini 集成 setup")
            ns_scene = "scene-0061"
        ns_mode = st.radio("模式", ["gt (无噪声)", "noisy (RangeNoiseModel)"], index=0,
                            horizontal=False)
        # Track config (v0.3 优化后是 max_miss=1 + IMM)
        c1, c2 = st.columns(2)
        with c1:
            ns_gate = st.slider("关联门限", 5.0, 30.0, 10.0, key="ns_gate")
            ns_min_hits = st.slider("确认命中", 1, 5, 3, key="ns_mh")
        with c2:
            ns_max_miss = st.slider("max_miss", 1, 15, 1, key="ns_mm")
            ns_use_imm = st.checkbox("IMM (CV+CA)", True, key="ns_imm",
                                       help="CV+CA 交互多模型. 默认开 (MOTA +0.012)")
        ns_max_frames = st.slider("最大帧数", 5, 50, 20, key="ns_mf",
                                  help="每个 scene 最多跑 N 帧 (防 UI 卡顿)")
        st.divider()
        st.info("📊 真实数据 MOTA baseline = **+0.209** "
                "(10 scene × 404 frames in 1.8s)")
    else:
        ns_scene = None
        ns_mode = None
        ns_gate = 10.0
        ns_min_hits = 2
        ns_max_miss = 5
        ns_use_imm = False
        ns_max_frames = 20

    scenario_type = st.selectbox("场景", [
        "Highway (高速)",
        "Urban (城市)",
        "Dense Highway (密集)",
        "Junction (路口)",
        "Stop & Go (拥堵)",
    ])
    num_vehicles = st.slider("车辆数", 2, 8, 5)
    duration = st.slider("时长 (秒)", 5, 30, 15)
    ego_speed = st.slider("自车速度 (m/s)", 5, 40, 25)

    st.divider()
    st.subheader("传感器开关")
    use_lidar = st.checkbox("LiDAR", True)
    use_radar = st.checkbox("Radar", True)
    use_camera = st.checkbox("Camera", True)
    use_imu = st.checkbox("IMU", True)
    use_gps = st.checkbox("GPS", True)

    st.divider()
    st.subheader("融合参数")
    gate_threshold = st.slider("关联门限 (m)", 3.0, 20.0, 10.0)
    min_hits = st.slider("确认所需命中数", 1, 5, 2)

    st.divider()
    st.subheader("🌤️ 天气/光照")
    weather_name = st.selectbox("天气条件", list(WEATHER_PRESETS.keys()),
                                index=0)

    st.divider()
    st.subheader("🔗 融合算法")
    association_mode = st.selectbox("关联模式", ["hungarian", "jpda"], index=0,
                                    help="Hungarian=贪心匹配（推荐）；JPDA=联合概率数据关联（实验）")
    use_ukf = st.checkbox("使用 UKF（默认 EKF）", value=False,
                          help="UKF=无迹卡尔曼滤波（Beta）")

    st.divider()
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_button = st.button("▶️ 启动", type="primary", width="stretch")
    with col_btn2:
        reset_button = st.button("🔄 重置", width="stretch")


# ===== Session state =====
if 'frames' not in st.session_state:
    st.session_state.frames = None
    st.session_state.metrics = None
    st.session_state.per_frame = None
    st.session_state.last_run_t = None
    st.session_state.current_frame_idx = None  # 滑块位置

if reset_button:
    st.session_state.frames = None
    st.session_state.metrics = None
    st.session_state.per_frame = None
    st.session_state.current_frame_idx = None
    # 清除预渲染的 3D 图缓存
    for key in ['cached_3d', 'cached_top']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


# ===== 运行仿真 =====
def run_simulation(use_l, use_r, use_c, use_i, use_g, gate, min_h,
                   scenario_type, num_vehicles, duration, ego_speed, weather_name,
                   association_mode='hungarian', use_ukf=False):
    """执行仿真（带进度条）"""
    if scenario_type == "Highway (高速)":
        scenario = HighwayScenario(num_vehicles=num_vehicles, duration=duration, dt=0.05,
                                   ego_speed_mps=ego_speed)
    elif scenario_type == "Urban (城市)":
        scenario = UrbanScenario(num_vehicles=num_vehicles, duration=duration, dt=0.05,
                                 ego_speed_mps=ego_speed)
    elif scenario_type == "Dense Highway (密集)":
        scenario = DenseHighwayScenario(num_lanes=3, cars_per_lane=num_vehicles//3 + 1,
                                        duration=duration, dt=0.05, ego_speed_mps=ego_speed)
    elif scenario_type == "Junction (路口)":
        scenario = JunctionScenario(num_vehicles=12, duration=duration, dt=0.05)
    elif scenario_type == "Stop & Go (拥堵)":
        scenario = StopAndGoScenario(num_vehicles=min(num_vehicles, 8), duration=duration, dt=0.05)
    else:
        scenario = HighwayScenario(num_vehicles=num_vehicles, duration=duration, dt=0.05,
                                   ego_speed_mps=ego_speed)

    weather = get_weather_by_name(weather_name)

    sensors = {}
    if use_l:
        sensors['lidar_top'] = LidarSensor(position=np.array([0, 0, 1.7]),
                                            rate_hz=10.0, num_lines=32, max_range_m=80.0)
    if use_r:
        sensors['radar_front'] = RadarSensor(position=np.array([2.0, 0, 0.5]),
                                              rate_hz=20.0, max_range_m=200.0)
    if use_c:
        sensors['camera_front'] = CameraSensor(position=np.array([2.0, 0, 1.5]),
                                                rate_hz=10.0, max_range_m=100.0)
    if use_i:
        sensors['imu'] = IMUSensor(position=np.zeros(3), rate_hz=100.0)
    if use_g:
        sensors['gps'] = GPSSensor(position=np.zeros(3), rate_hz=1.0)

    fusion = MultiObjectTracker(
        dt=0.05,
        gate_threshold=gate,
        min_hits_to_confirm=min_h,
        association_mode=association_mode,
        use_ukf=use_ukf,
    )
    sim = Simulator(scenario, sensors, fusion, dt=0.05, weather=weather)

    n_frames = int(duration * 20)
    sim.reset()
    sim.clock.start()
    frames = []

    # 进度条
    progress = st.progress(0.0, text=f"🏃 仿真中... 0/{n_frames}")
    import time as _time
    t0 = _time.time()
    for i in range(n_frames):
        frames.append(sim.step())
        if (i + 1) % 20 == 0 or i == n_frames - 1:
            progress.progress((i + 1) / n_frames, text=f"🏃 仿真中... {i+1}/{n_frames} ({100*(i+1)/n_frames:.0f}%)")
    elapsed = _time.time() - t0
    progress.empty()
    return frames, elapsed


def run_nuscenes(scene_name, mode='gt', gate=10.0, min_hits=3,
                 max_miss=1, use_imm=True, max_frames=20,
                 dataroot='/Users/mac/.openclaw/workspace/self-driving-sim/data/nuscenes'):
    """在真实 nuScenes mini 数据上跑 tracker, 返回 SimFrame 列表 (同 simulation 格式)."""
    from scripts.adapters.nuscenes_adapter import NuScenesAdapter
    from fusion import MultiObjectTracker
    noise_model = None
    if mode == 'noisy':
        from sensors.range_model import RangeNoiseModel
        noise_model = RangeNoiseModel(max_range_m=80.0)
    adapter = NuScenesAdapter(dataroot, version='v1.0-mini',
                               mode=mode, noise_model=noise_model,
                               verbose=False, deduplicate=False,
                               load_sensor_data=True)

    tracker = MultiObjectTracker(
        dt=0.5,
        gate_threshold=gate,
        min_hits_to_confirm=min_hits,
        max_miss_streak=max_miss,
        use_confidence_weighted=True,
        use_imm=use_imm,
    )

    frames = []
    progress = st.progress(0.0, text=f"🎥 加载 nuScenes scene={scene_name} ...")
    import time as _time
    t0 = _time.time()
    # 先预估帧数 (仅供进度条; 实际不等那么多)
    try:
        scene_records = [s for s in adapter.nuscenes.scene if s['name'] == scene_name]
        total = min(scene_records[0]['nbr_samples'], max_frames) if scene_records else max_frames
    except Exception:
        total = max_frames
    for i, frame in enumerate(adapter.load_scene(scene_name)):
        if i >= max_frames:
            break
        tracks = tracker.update(frame.detections_by_sensor, frame.timestamp)
        frame.tracks = tracks
        frames.append(frame)
        if (i + 1) % 5 == 0 or i == 0:
            progress.progress(min((i + 1) / max(total, 1), 1.0),
                              text=f"🎥 {scene_name} frame {i+1}/{total} "
                                   f"(GT={len(frame.ground_truth)}, tracks={len(tracks)})")
    elapsed = _time.time() - t0
    progress.progress(1.0, text=f"✅ {len(frames)} 帧 in {elapsed:.1f}s")
    progress.empty()
    return frames, elapsed


# ===== 主区域 =====
if run_button:
    if data_source.startswith("真实"):
        with st.spinner(f"🎥 加载 nuScenes {ns_scene} ..."):
            frames, elapsed = run_nuscenes(
                ns_scene, mode='gt' if ns_mode.startswith('gt') else 'noisy',
                gate=ns_gate, min_hits=ns_min_hits, max_miss=ns_max_miss,
                use_imm=ns_use_imm, max_frames=ns_max_frames,
            )
    else:
        with st.spinner("🔧 准备仿真..."):
            frames, elapsed = run_simulation(
                use_lidar, use_radar, use_camera, use_imu, use_gps,
                gate_threshold, min_hits,
                scenario_type, num_vehicles, duration, ego_speed, weather_name,
                association_mode=association_mode, use_ukf=use_ukf
            )
    st.session_state.frames = frames
    st.session_state.metrics = evaluate(frames)
    st.session_state.per_frame = compute_per_frame_metrics(frames)
    st.session_state.last_run_t = elapsed
    # 清除旧的 3D 缓存（下次访问会重新预渲染）
    for key in ['cached_3d', 'cached_top']:
        if key in st.session_state:
            del st.session_state[key]
    # 默认显示中间帧
    st.session_state.current_frame_idx = len(frames) // 2
    st.success(f"✅ 完成 {len(frames)} 帧仿真，用时 {elapsed:.2f}s")
    st.rerun()


# ===== 显示结果 =====
if st.session_state.frames is not None:
    frames = st.session_state.frames
    metrics = st.session_state.metrics
    per_frame = st.session_state.per_frame
    n_frames = len(frames)

    # ===== 顶部：关键指标 + 仿真信息 =====
    st.subheader(f"📊 仿真结果（{n_frames} 帧）")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("位置 RMSE", f"{metrics.get('position_rmse_m', 0):.2f} m")
    col2.metric("速度 RMSE", f"{metrics.get('velocity_rmse_mps', 0):.2f} m/s")
    col3.metric("跟踪精度", f"{metrics.get('tracking_precision', 0)*100:.1f}%")
    col4.metric("跟踪召回", f"{metrics.get('tracking_recall', 0)*100:.1f}%")
    col5.metric("ID 切换", f"{metrics.get('id_switches', 0)}")
    col6.metric("总 tracks", f"{metrics.get('total_tracks', 0)}")

    # ===== 时间回放（多媒体控制台风格） =====
    st.divider()
    st.subheader("⏯️ 时间回放")

    # 初始化 autoplay 状态
    if 'autoplay' not in st.session_state:
        st.session_state.autoplay = False
    if 'autoplay_speed' not in st.session_state:
        st.session_state.autoplay_speed = 4  # 4x = 5 fps @ 20Hz dt (舒服看)

    # 初始化 current_frame_idx
    if st.session_state.current_frame_idx is None or st.session_state.current_frame_idx >= n_frames:
        st.session_state.current_frame_idx = n_frames - 1
    current_idx = st.session_state.current_frame_idx
    current_frame = frames[current_idx]

    # ===== 一行：多媒体控制台（顺序与播放流程一致） =====
    # [⏮️ 首帧] [◀️ 上一帧] [▶️/⏸️ 播放] [下一帧 ▶️] [⏭️ 末帧]   [——————滑块——————]   [速度 1x/2x/.../16x]
    btn_cols = st.columns([0.7, 0.7, 1.0, 0.7, 0.7, 6, 1.5])
    with btn_cols[0]:
        if st.button("⏮️", key='btn_first', help='首帧', width="stretch"):
            st.session_state.current_frame_idx = 0
            st.session_state.autoplay = False
            st.rerun()
    with btn_cols[1]:
        if st.button("◀", key='btn_prev', help='上一帧', width="stretch"):
            st.session_state.current_frame_idx = max(0, current_idx - 1)
            st.session_state.autoplay = False
            st.rerun()
    with btn_cols[2]:
        play_label = "⏸️" if st.session_state.autoplay else "▶️"
        btn_type = "secondary" if st.session_state.autoplay else "primary"
        if st.button(play_label, key='btn_play', help='播放/暂停', width="stretch", type=btn_type):
            st.session_state.autoplay = not st.session_state.autoplay
            st.rerun()
    with btn_cols[3]:
        if st.button("▶", key='btn_next', help='下一帧', width="stretch"):
            st.session_state.current_frame_idx = min(n_frames - 1, current_idx + 1)
            st.session_state.autoplay = False
            st.rerun()
    with btn_cols[4]:
        if st.button("⏭️", key='btn_last', help='末帧', width="stretch"):
            st.session_state.current_frame_idx = n_frames - 1
            st.session_state.autoplay = False
            st.rerun()
    with btn_cols[5]:
        # 滑块占中中部分
        new_idx = st.slider(
            "帧位置",
            min_value=0, max_value=n_frames - 1,
            value=current_idx,
            step=1,
            key='frame_slider',
            label_visibility='collapsed',
        )
        if new_idx != current_idx:
            st.session_state.current_frame_idx = new_idx
            st.session_state.autoplay = False
            current_idx = new_idx
            current_frame = frames[current_idx]
    with btn_cols[6]:
        # 速度选择 (内嵌下拉)
        new_speed = st.selectbox(
            "速度",
            options=[1, 2, 4, 8, 16],
            index=[1, 2, 4, 8, 16].index(st.session_state.autoplay_speed),
            format_func=lambda x: f"{x}x",
            key='speed_select',
            label_visibility='collapsed',
        )
        if new_speed != st.session_state.autoplay_speed:
            st.session_state.autoplay_speed = new_speed

    # ===== 第二行：信息条 =====
    progress_pct = (current_idx + 1) / n_frames
    st.progress(progress_pct)

    info_cols = st.columns([2, 1, 1, 1, 1])
    with info_cols[0]:
        st.caption(
            f"📍 帧 **{current_idx + 1} / {n_frames}** &nbsp;·&nbsp; "
            f"**t = {current_frame.timestamp:.2f}s**"
        )
    with info_cols[1]:
        st.metric("GT", len(current_frame.ground_truth), label_visibility='visible')
    with info_cols[2]:
        st.metric("Tracks", len(current_frame.tracks), label_visibility='visible')
    with info_cols[3]:
        if st.session_state.autoplay:
            st.success(f"▶ 播放中 ({st.session_state.autoplay_speed}x)")
        else:
            st.info("⏸ 暂停")
    with info_cols[4]:
        st.metric("总时长", f"{n_frames * 0.05:.1f}s", label_visibility='visible')

    # Autoplay 逻辑: 只重计算与 autoplay 相关的状态
    # 智能 step: 根据 speed 调整跳跃帧数
    if st.session_state.autoplay:
        # 更粗的 step 保持体验流畅
        speed_to_step = {1: 1, 2: 1, 4: 2, 8: 4, 16: 8}
        step = speed_to_step.get(st.session_state.autoplay_speed, 2)
        next_idx = st.session_state.current_frame_idx + step
        if next_idx >= n_frames - 1:
            st.session_state.autoplay = False
            st.session_state.current_frame_idx = n_frames - 1
        else:
            st.session_state.current_frame_idx = next_idx
        # 不使用 st.rerun() 会卡死, 仍需 rerun
        # 提示: streamlit 本身 rerender 需要 2-5秒
        # 所以 1x (1 帧/轮) 实际看起来是 0.2-0.5 fps - 慢但可见
        # 16x (8 帧/轮) 看起来是 1.6-4 fps - 快但能看见
        st.rerun()

    # ===== 3D + 俯视图 =====
    # 预渲染所有帧的 3D + BEV 图，缓存到 session_state
    # 这样 autoplay 不需要每帧重生成 Plotly figure
    if 'cached_3d' not in st.session_state or len(st.session_state.cached_3d) != n_frames:
        from visualization import render_frame_plotly, render_topdown_plotly, render_bev_fusion
        # nuScenes adapter 输出的 ground_truth/tracks 已经是 ego-centric 坐标系
        # 仿真 SimFrame 输出的是 global 坐标系
        # view 函数需要根据坐标系决定是否再减 ego_pos
        ego_centric_data = data_source.startswith("真实")
        cache_3d = []
        cache_top = []
        cache_bev = []  # v0.2.2: BEV 融合图
        prog = st.progress(0.0, text=f"⚙️ 预渲染 3D 图... 0/{n_frames}")
        for i, f in enumerate(frames):
            lpts = f.lidar_data.points if f.lidar_data else None
            lints = f.lidar_data.intensity if f.lidar_data else None
            cache_3d.append(render_frame_plotly(
                ego_pos=f.ego_state.position,
                ground_truth=f.ground_truth,
                tracks=f.tracks,
                lidar_points=lpts,
                title=f"Frame #{i+1} @ t={f.timestamp:.2f}s",
                data_in_ego_centric=ego_centric_data,
            ))
            cache_top.append(render_topdown_plotly(
                ego_pos=f.ego_state.position,
                ground_truth=f.ground_truth,
                tracks=f.tracks,
                data_in_ego_centric=ego_centric_data,
            ))
            # BEV 融合图
            rdet = f.radar_data.detections if f.radar_data else None
            cdet = f.camera_data.detections_3d if f.camera_data else None
            cache_bev.append(render_bev_fusion(
                ego_pos=f.ego_state.position,
                ground_truth=f.ground_truth,
                tracks=f.tracks,
                lidar_points=lpts,
                lidar_intensity=lints,
                radar_detections=rdet,
                camera_detections=cdet,
                title=f"BEV Fusion @ t={f.timestamp:.2f}s",
                data_in_ego_centric=ego_centric_data,
            ))
            if (i+1) % 20 == 0 or i == n_frames - 1:
                prog.progress((i+1)/n_frames, text=f"⚙️ 预渲染 3D 图... {i+1}/{n_frames}")
        prog.empty()
        st.session_state.cached_3d = cache_3d
        st.session_state.cached_top = cache_top
        st.session_state.cached_bev = cache_bev

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.subheader("🌐 3D 视图")
        st.plotly_chart(st.session_state.cached_3d[current_idx],
                        use_container_width=True, key=f'3d_{current_idx}')
    with col_r:
        st.subheader("📐 俯视图 (BEV)")
        st.plotly_chart(st.session_state.cached_top[current_idx],
                        use_container_width=True, key=f'top_{current_idx}')

    # v0.2.2: BEV 鸟瞰融合图（单图展示 fusion 价值）
    st.divider()
    st.subheader("🛰️ BEV 融合视图 (LiDAR 灰度 + Radar 等高线 + Camera bbox + Track 协方差)")
    st.plotly_chart(st.session_state.cached_bev[current_idx],
                    use_container_width=True, key=f'bev_{current_idx}')

    st.divider()

    # ===== 跟踪列表 =====
    st.subheader(f"🎯 跟踪列表（当前帧 #{current_idx + 1}）")
    if current_frame.tracks:
        import pandas as pd
        rows = []
        for t in current_frame.tracks:
            rows.append({
                'ID': t.track_id,
                'X (m)': round(float(t.position[0]), 2),
                'Y (m)': round(float(t.position[1]), 2),
                'Vx (m/s)': round(float(t.velocity[0]), 2),
                'Vy (m/s)': round(float(t.velocity[1]), 2),
                'Age': t.age,
                'Hits': t.hits,
                'Sources': ','.join(t.source_sensors),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.info("暂无跟踪目标")

    # ===== 相机图像 =====
    # ===== 传感器独立面板 (P3 传感器面板) =====
    st.divider()
    st.subheader("📡 传感器独立面板（原始数据）")
    # Autoplay 时跳过传感器面板的慢渲染（保持流畅）
    if st.session_state.autoplay:
        st.info("🎞️ 播放中：传感器面板已暂停刷新以保证流畅。点 “⏸️ 暂停” 可查看原始数据。")
    else:
        from visualization import (
            render_lidar_panel, render_radar_panel, render_camera_panel,
            render_imu_panel, render_gps_panel
        )
        # nuScenes adapter 输出的 sensor data 是 lidar frame (ego-centric), 不需再减 ego_pos
        ego_centric_data = data_source.startswith("真实")
        lidars = list(current_frame.get_lidars().keys())
        radars = list(current_frame.get_radars().keys())
        cameras = list(current_frame.get_cameras().keys())
        imus = list(current_frame.get_imus().keys())
        gpses = list(current_frame.get_gpses().keys())

        if not (lidars or radars or cameras or imus or gpses):
            st.info("当前帧无传感器原始数据")
        else:
            sensor_tabs = []
            if lidars:
                sensor_tabs.append(f"🔵 LiDAR ({len(lidars)})")
            if radars:
                sensor_tabs.append(f"📡 Radar ({len(radars)})")
            if cameras:
                sensor_tabs.append(f"📷 Camera ({len(cameras)})")
            if imus:
                sensor_tabs.append(f"🧭 IMU ({len(imus)})")
            if gpses:
                sensor_tabs.append(f"🌐 GPS ({len(gpses)})")

            st_sensor = st.tabs(sensor_tabs)
            idx = 0
            if lidars:
                with st_sensor[idx]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        selected_lidar = st.selectbox("LiDAR", lidars, key=f'lidar_{current_idx}')
                    with col2:
                        lidar_view = st.radio("视角", ["bev", "3d"], key=f'lview_{current_idx}', horizontal=True)
                    st.plotly_chart(render_lidar_panel(current_frame, selected_lidar, lidar_view,
                                                       data_in_ego_centric=ego_centric_data),
                                    use_container_width=True, key=f'lpanel_{current_idx}')
                idx += 1
            if radars:
                with st_sensor[idx]:
                    selected_radar = st.selectbox("Radar", radars, key=f'radar_{current_idx}')
                    st.plotly_chart(render_radar_panel(current_frame, selected_radar,
                                                        data_in_ego_centric=ego_centric_data),
                                    use_container_width=True, key=f'rpanel_{current_idx}')
                idx += 1
            if cameras:
                with st_sensor[idx]:
                    selected_camera = st.selectbox("Camera", cameras, key=f'cam_{current_idx}')
                    st.plotly_chart(render_camera_panel(current_frame, selected_camera),
                                    use_container_width=True, key=f'cpanel_{current_idx}')
                idx += 1
            if imus:
                with st_sensor[idx]:
                    selected_imu = st.selectbox("IMU", imus, key=f'imu_{current_idx}')
                    st.plotly_chart(render_imu_panel(current_frame, selected_imu),
                                    use_container_width=True, key=f'ipanel_{current_idx}')
                idx += 1
            if gpses:
                with st_sensor[idx]:
                    selected_gps = st.selectbox("GPS", gpses, key=f'gps_{current_idx}')
                    st.plotly_chart(render_gps_panel(current_frame, selected_gps),
                                    use_container_width=True, key=f'gpanel_{current_idx}')
                idx += 1

    # ===== 指标时序曲线 =====
    st.divider()
    st.subheader("📈 指标时序曲线（全程）")

    tab1, tab2, tab3 = st.tabs(["位置/速度误差", "目标数量", "跟踪匹配"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=per_frame['timestamps'], y=per_frame['pos_rmse'],
                                 mode='lines', name='位置 RMSE (m)', line=dict(color='red', width=2)))
        fig.add_trace(go.Scatter(x=per_frame['timestamps'], y=per_frame['pos_max'],
                                 mode='lines', name='位置 Max (m)', line=dict(color='darkred', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=per_frame['timestamps'], y=per_frame['vel_rmse'],
                                 mode='lines', name='速度 RMSE (m/s)', line=dict(color='blue', width=2),
                                 yaxis='y2'))
        fig.update_layout(
            xaxis_title='时间 (s)', height=350,
            yaxis=dict(title='位置 (m)', color='red'),
            yaxis2=dict(title='速度 (m/s)', color='blue', overlaying='y', side='right'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=per_frame['timestamps'], y=per_frame['n_gt'],
                                 mode='lines', name='GT 目标数', line=dict(color='red', width=2)))
        fig.add_trace(go.Scatter(x=per_frame['timestamps'], y=per_frame['n_tracks'],
                                 mode='lines', name='跟踪数', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=per_frame['timestamps'], y=per_frame['n_matched'],
                                 mode='lines', name='成功匹配数', line=dict(color='blue', width=2, dash='dot')))
        fig.update_layout(
            xaxis_title='时间 (s)', yaxis_title='数量',
            height=350, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # 跟踪精度/召回时序
        timestamps = np.array(per_frame['timestamps'])
        n_tracks = np.array(per_frame['n_tracks'])
        n_gt = np.array(per_frame['n_gt'])
        n_matched = np.array(per_frame['n_matched'])
        # 计算每帧的 precision/recall
        prec = np.where(n_tracks > 0, n_matched / np.maximum(n_tracks, 1), 0)
        rec = np.where(n_gt > 0, n_matched / np.maximum(n_gt, 1), 0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timestamps, y=prec, mode='lines',
                                 name='Precision', line=dict(color='orange', width=2)))
        fig.add_trace(go.Scatter(x=timestamps, y=rec, mode='lines',
                                 name='Recall', line=dict(color='purple', width=2)))
        fig.update_layout(
            xaxis_title='时间 (s)', yaxis_title='比率', yaxis=dict(range=[0, 1.05]),
            height=350, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===== 详细指标 + 导出 =====
    st.divider()
    col_detail, col_export = st.columns([2, 1])
    with col_detail:
        with st.expander("📊 完整评估指标", expanded=False):
            st.json(metrics)
    with col_export:
        st.subheader("💾 导出")
        if st.button("📥 导出当前帧 (JSON)"):
            import json
            out = {
                'timestamp': current_frame.timestamp,
                'ego': current_frame.ego_state.position.tolist(),
                'ground_truth': [{'id': gt.object_id, 'pos': gt.position.tolist()} for gt in current_frame.ground_truth],
                'tracks': [{'id': t.track_id, 'pos': t.position.tolist()} for t in current_frame.tracks],
            }
            st.download_button(
                "下载 frame.json",
                json.dumps(out, indent=2, ensure_ascii=False),
                file_name=f"frame_{current_frame.timestamp:.2f}.json",
                mime="application/json"
            )

        # 全场 JSON 导出 (P2 新增)
        if st.button("📦 导出全场 (JSON)"):
            with st.spinner("📦 正在打包全部帧..."):
                from scripts.export_session import export_session_json
                meta = {
                    'scenario_type': scenario_type,
                    'weather': weather_name,
                    'num_vehicles': num_vehicles,
                    'duration': duration,
                    'ego_speed': ego_speed,
                    'gate_threshold': gate_threshold,
                    'min_hits': min_hits,
                    'metrics': metrics,
                }
                out = export_session_json(frames, '/tmp/session.json', metadata=meta)
                with open(out, 'r') as f:
                    session_data = f.read()
                st.success(f"✅ {len(frames)} 帧, {len(session_data)/1024:.1f} KB")
                st.download_button(
                    "📥 下载 session.json",
                    session_data,
                    file_name=f"session_{scenario_type.split(' ')[0]}_{weather_name.split(' ')[0]}.json",
                    mime="application/json"
                )

        # CSV 导出 (P2 新增)
        if st.button("📊 导出 track CSV"):
            with st.spinner("📊 正在生成 CSV..."):
                from scripts.export_session import export_tracks_csv, export_groundtruth_csv
                track_csv = export_tracks_csv(frames, '/tmp/tracks.csv')
                gt_csv = export_groundtruth_csv(frames, '/tmp/gt.csv')
                with open(track_csv, 'r') as f:
                    st.download_button(
                        "📥 下载 tracks.csv",
                        f.read(),
                        file_name=f"tracks_{scenario_type.split(' ')[0]}.csv",
                        mime="text/csv"
                    )
                with open(gt_csv, 'r') as f:
                    st.download_button(
                        "📥 下载 groundtruth.csv",
                        f.read(),
                        file_name=f"groundtruth_{scenario_type.split(' ')[0]}.csv",
                        mime="text/csv"
                    )

        # 动画 GIF 导出
        if st.button("🎬 导出动画 GIF"):
            with st.spinner("🎬 正在生成 GIF 动画..."):
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
                tmp.close()
                n = export_gif(frames, tmp.name, duration_ms=80, max_frames=200,
                                data_in_ego_centric=data_source.startswith("真实"))
                if n > 0:
                    with open(tmp.name, 'rb') as f:
                        gif_bytes = f.read()
                    st.success(f"✅ 生成 {n} 帧 GIF ({len(gif_bytes)/1024:.1f} KB)")
                    st.download_button(
                        "📥 下载 animation.gif",
                        gif_bytes,
                        file_name=f"sim_{scenario_type.split(' ')[0]}_{duration}s.gif",
                        mime="image/gif"
                    )
                    # 预览
                    st.image(gif_bytes, caption=f"动画预览 ({n} 帧, {duration}s)", width="stretch")
                os.unlink(tmp.name)

else:
    st.info("👈 在左侧配置参数后点击「▶️ 启动」")
    st.markdown("""
    ### 📋 系统架构

    ```
    场景引擎  →  5 个传感器（LiDAR / Radar / Camera / IMU / GPS）
              ↓
           统一 Detection 流
              ↓
         融合层：EKF + 匈牙利匹配
              ↓
         3D 跟踪结果 + 评估指标
    ```

    ### 🛠️ 模块
    - **场景**: 高速 / 城市
    - **传感器**: LiDAR（射线投射）、Radar（FMCW 简化）、Camera（针孔投影）、IMU、GPS
    - **融合**: EKF 匀速模型 + 匈牙利数据关联
    - **3D 渲染**: Open3D / Plotly
    - **Web UI**: Streamlit

    ### 📊 评估指标
    - 位置/速度 RMSE、MAE
    - 跟踪精度/召回、ID 切换
    - 时延统计

    ### 🎬 v0.2 新增 (P1 优化)
    - ⏯️ **时间滑块回放** - 拖动看任意帧
    - 📊 **进度条** - 仿真时实时显示
    - 📈 **指标时序曲线** - RMSE/数量/匹配率随时间变化
    - 🎯 **6 个顶部指标** - 一眼看到全局状态
    """)
