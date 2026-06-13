"""
场景 + 评估单测
"""
import numpy as np
import pytest
from core import Simulator
from scenarios import HighwayScenario, DenseHighwayScenario, StopAndGoScenario
from sensors import LidarSensor, RadarSensor, IMUSensor, GPSSensor
from fusion import MultiObjectTracker
from evaluation import evaluate, compute_rmse, compute_tracking_stats


def make_basic_sensors():
    return {
        'lidar_top': LidarSensor('lidar_top', np.array([0,0,1.7]), 10.0, 32, (-25.0, 25.0), 80.0, 1.0, 0.2, 0.05, 0.8, 5),
        'radar_front': RadarSensor('radar_front', np.array([2.0,0,0.5]), 20.0, 200.0, 0.1),
        'imu': IMUSensor('imu', np.zeros(3), 100.0),
        'gps': GPSSensor('gps', np.zeros(3), 1.0),
    }


class TestHighwayScenario:
    """Highway 场景"""

    def test_default_count(self):
        """默认 5 辆周围车（num_vehicles=5）"""
        s = HighwayScenario(num_vehicles=5, duration=10.0, dt=0.05)
        frame = s.step(t=0.0)
        ego, gts = frame
        assert len(gts) == 5

    def test_count_configurable(self):
        s = HighwayScenario(num_vehicles=3, duration=10.0, dt=0.05)
        frame = s.step(t=0.0)
        _, gts = frame
        assert len(gts) == 3

    def test_deterministic(self):
        """同 seed 下场景应该一致（使用 numpy seed 模拟）"""
        np.random.seed(42)
        s1 = HighwayScenario(num_vehicles=5, duration=2.0, dt=0.05)
        _, gts1 = s1.step(t=1.0)
        np.random.seed(42)
        s2 = HighwayScenario(num_vehicles=5, duration=2.0, dt=0.05)
        _, gts2 = s2.step(t=1.0)
        for g1, g2 in zip(gts1, gts2):
            np.testing.assert_array_equal(g1.position, g2.position)


class TestDenseScenario:
    """密集场景"""

    def test_24_vehicles(self):
        """默认 3车道 × 8车 = 24 辆"""
        s = DenseHighwayScenario(num_lanes=3, cars_per_lane=8, duration=10.0, dt=0.05)
        _, gts = s.step(t=0.0)
        assert len(gts) == 24

    def test_reduces_count(self):
        """cars_per_lane 减少时车辆数也减少"""
        s = DenseHighwayScenario(num_lanes=2, cars_per_lane=4, duration=10.0, dt=0.05)
        _, gts = s.step(t=0.0)
        assert len(gts) == 8


class TestStopAndGo:
    """Stop & Go 场景"""

    def test_basic_run(self):
        """跑 100 帧不崩"""
        s = StopAndGoScenario(num_vehicles=3, duration=5.0, dt=0.05)
        for i in range(100):
            t = i * 0.05
            s.step(t)


class TestEvaluation:
    """评估函数"""

    def test_evaluate_basic(self):
        """完整跑 5车 100 帧，evaluate() 返回完整指标"""
        s = HighwayScenario(num_vehicles=5, duration=5.0, dt=0.05)
        sensors = make_basic_sensors()
        fusion = MultiObjectTracker(dt=0.05, gate_threshold=15.0, max_miss_streak=30)
        sim = Simulator(s, sensors, fusion, dt=0.05)
        frames = sim.run(100)
        m = evaluate(frames)
        # 关键字段
        assert 'position_rmse_m' in m
        assert 'velocity_rmse_mps' in m
        assert 'MOTA' in m
        assert 'id_switches' in m
        # Highway 5车 baseline 应该 MOTA > 0.5
        assert m['MOTA'] > 0.5, f"MOTA {m['MOTA']:.3f} 低于 0.5"
