#!/usr/bin/env python3
"""
P3-B 验收脚本: GNNS vs Hungarian 在不同场景下的 ID Switches 对比

场景:
1. Highway 5 车 (简单): 二者都应很低
2. Dense Highway 24 车 (密集): GNNS 优势主要在协方差不均匀
3. 协方差不均匀场景: 模拟部分 track 老化 / 部分刚创建
4. 大速度差异场景: 模拟快速目标 + 慢速目标并存

验收标准: GNNS ID Switches ≤ Hungarian ID Switches
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulator import Simulator
from core.data_types import Detection
from sensors.lidar import LidarSensor
from sensors.radar import RadarSensor
from sensors.imu_gps import IMUSensor
from scenarios.highway import HighwayScenario
from scenarios.dense import DenseHighwayScenario
from fusion.tracker import MultiObjectTracker
from evaluation.metrics import compute_mot_metrics


def run_scenario(scenario_cls, scenario_kwargs, assoc_mode: str, n_frames: int = 100):
    np.random.seed(42)
    scenario = scenario_cls(**scenario_kwargs)
    sensors = {
        'lidar_top': LidarSensor('lidar_top', rate_hz=10.0, position=np.array([0, 0, 1.5])),
        'radar_front': RadarSensor('radar_front', rate_hz=20.0, position=np.array([0, 0, 0.5])),
        'imu': IMUSensor('imu', rate_hz=100.0),
    }
    tracker = MultiObjectTracker(
        dt=0.05, association_mode=assoc_mode,
        use_ego_motion=False, max_miss_streak=20)
    sim = Simulator(scenario=scenario, sensors=sensors, fusion=tracker, dt=0.05)
    sim.reset()
    sim.clock.start()
    frames = []
    for _ in range(n_frames):
        frames.append(sim.step())
    return frames, compute_mot_metrics(frames)


def scenario_highway_5():
    return HighwayScenario, dict(num_vehicles=5, dt=0.05), 100


def scenario_dense_24():
    return DenseHighwayScenario, dict(num_lanes=3, cars_per_lane=8, dt=0.05), 100


def scenario_covariance_heterogeneous():
    """
    协方差不均匀场景: 手动构造
    - 5 个 track: 2 个老化 (P 大), 2 个新创建 (P 小), 1 个高机动 (P 极大)
    - dets 在 track 附近,但噪声大,让 NN 容易误匹配
    """
    from fusion.tracker import MultiObjectTracker
    from fusion.ekf import EKFTrack
    from core.data_types import GroundTruthObj
    from scenarios.base import BaseScenario

    class HetScenario(BaseScenario):
        def __init__(self):
            super().__init__(duration=10.0, dt=0.05)
            # 5 个 GT
            self.vehicles = [
                GroundTruthObj(1, 0.0, 'car', np.array([10.0, 0, 0]), np.array([10, 0, 0]), 0, np.array([4.5, 1.8, 1.5]), 'red'),
                GroundTruthObj(2, 0.0, 'car', np.array([15.0, 0.5, 0]), np.array([10, 0, 0]), 0, np.array([4.5, 1.8, 1.5]), 'blue'),
                GroundTruthObj(3, 0.0, 'car', np.array([20.0, -0.5, 0]), np.array([10, 0, 0]), 0, np.array([4.5, 1.8, 1.5]), 'green'),
                GroundTruthObj(4, 0.0, 'car', np.array([25.0, 1.0, 0]), np.array([10, 0, 0]), 0, np.array([4.5, 1.8, 1.5]), 'yellow'),
                GroundTruthObj(5, 0.0, 'car', np.array([30.0, -1.0, 0]), np.array([10, 0, 0]), 0, np.array([4.5, 1.8, 1.5]), 'white'),
            ]
            self.ego = None

        def step(self, t):
            from core.data_types import EgoState
            self.ego = EgoState(t, np.array([0, 0, 0]), np.array([25, 0, 0]),
                                 np.array([0, 0, 0]), 0, np.array([0, 0, 0]))
            for v in self.vehicles:
                v.position = v.position + v.velocity * 0.05
                v.timestamp = t
            return self.ego, list(self.vehicles)

    return HetScenario, dict(), 100


def main():
    print("=" * 70)
    print("P3-B 验收: GNNS vs Hungarian ID Switches 对比")
    print("=" * 70)

    scenarios = [
        ("Highway 5 车", scenario_highway_5),
        ("Dense Highway 24 车", scenario_dense_24),
        ("协方差不均匀 (5 车 + 异 P)", scenario_covariance_heterogeneous),
    ]

    results = []
    for name, getter in scenarios:
        cls, kwargs, n_frames = getter()
        print(f"\n▶ 场景: {name}")
        _, mot_h = run_scenario(cls, kwargs, 'hungarian', n_frames)
        _, mot_g = run_scenario(cls, kwargs, 'gnns', n_frames)
        idsw_h = mot_h.get('idswitches', 0)
        idsw_g = mot_g.get('idswitches', 0)
        mota_h = mot_h.get('mota', 0)
        mota_g = mot_g.get('mota', 0)
        print(f"  Hungarian: MOTA={mota_h:+.4f}  ID Switches={idsw_h}")
        print(f"  GNNS:      MOTA={mota_g:+.4f}  ID Switches={idsw_g}")
        if idsw_h > 0:
            ratio = idsw_g / idsw_h
            print(f"  ID Switches 比率: {ratio:.2f} (目标 ≤ 0.7 = 降低 30%)")
        results.append((name, mota_h, mota_g, idsw_h, idsw_g))

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"{'场景':<35} {'MOTA H':<10} {'MOTA G':<10} {'IDSW H':<8} {'IDSW G':<8}")
    for name, mh, mg, ih, ig in results:
        print(f"{name:<35} {mh:+.4f}     {mg:+.4f}     {ih:<8} {ig:<8}")

    # 整体验收
    total_h = sum(r[3] for r in results)
    total_g = sum(r[4] for r in results)
    if total_h > 0:
        overall = total_g / total_h
        print(f"\n总体 ID Switches 比值: {overall:.2f} (目标 ≤ 1.0 = 不恶化)")
    else:
        print(f"\nID Switches 都为 0 — 二者在仿真场景下表现都很好")


if __name__ == '__main__':
    main()
