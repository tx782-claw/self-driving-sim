"""
端到端冒烟测试
跑通：场景 → 4 个传感器 → EKF 融合 → 评估
输出：3D 渲染图 + 评估指标
"""
import sys
import os
import numpy as np

# 把项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Simulator
from scenarios import HighwayScenario
from sensors import LidarSensor, RadarSensor, CameraSensor, IMUSensor, GPSSensor
from fusion import MultiObjectTracker
from evaluation import evaluate
from visualization import render_frame, render_topdown


def main():
    print("=" * 60)
    print("🚗 Self-Driving Sensor Simulation & Fusion")
    print("=" * 60)

    # 1. 创建场景
    scenario = HighwayScenario(num_vehicles=5, duration=20.0, dt=0.05)
    print("\n✅ 场景: HighwayScenario (5 vehicles, 20s, 20Hz)")

    # 2. 创建传感器
    sensors = {
        'lidar_top': LidarSensor(
            sensor_id='lidar_top',
            position=np.array([0, 0, 1.7]),  # 车顶
            rate_hz=10.0,
            num_lines=32,
            max_range_m=80.0,
            noise_std=0.05,
        ),
        'radar_front': RadarSensor(
            sensor_id='radar_front',
            position=np.array([2.0, 0, 0.5]),  # 车前
            rate_hz=20.0,
            max_range_m=200.0,
            noise_std=0.1,
        ),
        'camera_front': CameraSensor(
            sensor_id='camera_front',
            position=np.array([2.0, 0, 1.5]),
            rate_hz=10.0,
            max_range_m=100.0,
        ),
        'imu': IMUSensor(
            sensor_id='imu',
            position=np.zeros(3),
            rate_hz=100.0,
        ),
        'gps': GPSSensor(
            sensor_id='gps',
            position=np.zeros(3),
            rate_hz=1.0,
        ),
    }
    print(f"✅ 传感器: {len(sensors)} 个")
    for sid, s in sensors.items():
        print(f"   - {sid:15} {s.rate_hz:5.1f} Hz")

    # 3. 创建融合器 (P1 优化版参数)
    fusion = MultiObjectTracker(
        dt=0.05,
        gate_threshold=15.0,  # 大门限以容许 EKF 跳变
        min_hits_to_confirm=2,
        max_miss_streak=30,  # 1.5秒(30帧)容忍丢失
        use_confidence_weighted=True,
    )
    print("✅ 融合器: MultiObjectTracker (EKF + Hungarian)")

    # 4. 创建仿真器
    sim = Simulator(scenario, sensors, fusion, dt=0.05)
    print("✅ Simulator 就绪")

    # 5. 跑 20 秒 (400 帧 @ 20Hz)
    n_frames = 400
    print(f"\n🏃 开始仿真 ({n_frames} 帧, 约 {n_frames/20:.1f} 秒) ...")
    import time
    t0 = time.time()
    frames = sim.run(n_frames)
    elapsed = time.time() - t0
    print(f"   完成: {elapsed:.2f}s 实际用时, 仿真时长 {n_frames*0.05:.1f}s")
    print(f"   实时性: {(n_frames*0.05)/elapsed:.2f}x")

    # 6. 评估
    print("\n📊 评估指标:")
    metrics = evaluate(frames)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   {k:30s}: {v:.4f}")
        else:
            print(f"   {k:30s}: {v}")

    # 7. 渲染最后一帧 3D + 俯视图
    print("\n🎨 生成可视化...")
    last = frames[-1]
    lidar_pts = last.lidar_data.points if last.lidar_data else None
    fig_3d = render_frame(
        ego_pos=last.ego_state.position,
        ground_truth=last.ground_truth,
        tracks=last.tracks,
        lidar_points=lidar_pts,
        title=f"Frame #{len(frames)} @ t={last.timestamp:.2f}s"
    )
    fig_top = render_topdown(
        ego_pos=last.ego_state.position,
        ground_truth=last.ground_truth,
        tracks=last.tracks,
    )

    out_dir = '/Users/mac/.openclaw/workspace/self-driving-sim/scenarios_data'
    os.makedirs(out_dir, exist_ok=True)
    out_3d = os.path.join(out_dir, 'e2e_frame_3d.png')
    out_top = os.path.join(out_dir, 'e2e_topdown.png')
    # 同时写 HTML 和 PNG (根据后端)
    if hasattr(fig_3d, 'savefig'):
        fig_3d.savefig(out_3d, dpi=80, bbox_inches='tight')
    elif hasattr(fig_3d, 'write_html'):
        fig_3d.write_html(out_3d.replace('.png', '.html'))
    if hasattr(fig_top, 'savefig'):
        fig_top.savefig(out_top, dpi=80, bbox_inches='tight')
    elif hasattr(fig_top, 'write_html'):
        fig_top.write_html(out_top.replace('.png', '.html'))
    print(f"   ✅ 3D 视图: {out_3d}")
    print(f"   ✅ 俯视图:  {out_top}")

    # 8. 打印跟踪摘要
    print("\n🎯 跟踪摘要（最后 5 帧）:")
    for i, frame in enumerate(frames[-5:]):
        print(f"   t={frame.timestamp:.2f}s  tracks={len(frame.tracks):2d}  gt={len(frame.ground_truth):2d}")
        for trk in frame.tracks[:3]:
            print(f"      #{trk.track_id}  pos={trk.position[:2].round(1)}  vel={trk.velocity[:2].round(1)}  age={trk.age}")

    # 9. 保存 JSON 导出
    out_json = os.path.join(out_dir, 'last_frame.json')
    export_frame(last, out_json)
    print(f"\n💾 最后一帧已导出: {out_json}")

    print("\n✅ 全部完成!")
    return metrics, frames


def export_frame(frame, path):
    """导出单帧数据为 JSON"""
    import json
    data = {
        'timestamp': frame.timestamp,
        'ego': {
            'position': frame.ego_state.position.tolist(),
            'velocity': frame.ego_state.velocity.tolist(),
        },
        'ground_truth': [
            {
                'id': gt.object_id,
                'type': gt.obj_type,
                'position': gt.position.tolist(),
                'velocity': gt.velocity.tolist(),
                'size': gt.size.tolist(),
            }
            for gt in frame.ground_truth
        ],
        'tracks': [
            {
                'id': trk.track_id,
                'position': trk.position.tolist(),
                'velocity': trk.velocity.tolist(),
                'age': trk.age,
                'hits': trk.hits,
                'sources': list(trk.source_sensors),
            }
            for trk in frame.tracks
        ],
        'lidar_points': len(frame.lidar_data.points) if frame.lidar_data else 0,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
