"""
传感器 + 距离噪声模型单测
"""
import numpy as np
import pytest
from core.data_types import EgoState
from sensors.lidar import LidarSensor
from sensors.radar import RadarSensor
from sensors.camera import CameraSensor
from sensors.range_model import RangeNoiseModel, LIDAR_NOISE_MODEL, RADAR_NOISE_MODEL, CAMERA_NOISE_MODEL


class TestRangeNoiseModel:
    """距离依赖噪声 + 漏检率"""

    def test_near_m_low_sigma(self):
        """近距标准差应该低"""
        m = LIDAR_NOISE_MODEL
        sigma, miss = m.at(range_m=2.0)
        assert sigma < 0.05  # LiDAR 近距很准
        assert miss < 0.05

    def test_far_m_high_sigma(self):
        """远距标准差应该高"""
        m = LIDAR_NOISE_MODEL
        sigma, miss = m.at(range_m=80.0)
        assert sigma > 0.10
        assert miss > 0.0

    def test_interpolation(self):
        """中间距离应该线性插值"""
        m = LIDAR_NOISE_MODEL
        sigma_mid, _ = m.at(range_m=42.5)  # 远端中点
        sigma_near, _ = m.at(range_m=5.0)
        sigma_far, _ = m.at(range_m=80.0)
        # 中点 sigma 应该 ≈ (sigma_near + sigma_far) / 2
        expected = (sigma_near + sigma_far) / 2
        assert abs(sigma_mid - expected) < 0.01

    def test_out_of_range_clamp(self):
        """超出远端边界应该用 far 值（不外推）"""
        m = LIDAR_NOISE_MODEL
        sigma_100, _ = m.at(range_m=100.0)
        sigma_200, _ = m.at(range_m=200.0)
        assert sigma_100 == sigma_200, "超出 far_m 不应外推"

    def test_max_range_check(self):
        """超 max_range_m 应该返回 False"""
        m = RangeNoiseModel(max_range_m=50.0)
        assert m.is_in_range(40.0) is True
        assert m.is_in_range(60.0) is False
        assert m.is_in_range(-1.0) is False

    def test_invalid_near_far(self):
        """far_m <= near_m 应该报错"""
        with pytest.raises(ValueError):
            RangeNoiseModel(near_m=10.0, far_m=5.0)

    def test_sample_miss_rate(self):
        """蒙特卡洛 miss 采样率应该接近 miss_rate"""
        m = RangeNoiseModel(miss_rate_near=0.0, miss_rate_far=1.0,
                            near_m=5.0, far_m=20.0)
        np.random.seed(42)
        # miss_rate_far=1.0 时所有远距 sample() 都应该 False
        assert m.sample(20.0) is False
        # miss_rate_near=0.0 时所有近距 sample() 都应该 True
        assert m.sample(5.0) is True

    def test_presets(self):
        """3 个预设应该有不同的特征"""
        assert LIDAR_NOISE_MODEL.sigma_far < RADAR_NOISE_MODEL.sigma_far
        assert RADAR_NOISE_MODEL.sigma_far < CAMERA_NOISE_MODEL.sigma_far


class TestLidarSensor:
    """LiDAR 仿真基础"""

    def test_basic_sense(self):
        """在简单场景下 sense() 应返回 det"""
        sensor = LidarSensor('lidar', np.array([0,0,1.7]), 10.0,
                             num_lines=32, max_range_m=80.0, noise_std=0.05)
        ego = EgoState(timestamp=0.0, position=np.array([0,0,0]),
                       velocity=np.array([10,0,0]), acceleration=np.zeros(3),
                       heading=0.0, angular_velocity=np.zeros(3))
        from core.data_types import GroundTruthObj
        gt = [GroundTruthObj(
            object_id=1, timestamp=0.0,
            position=np.array([20, 0, 0]), velocity=np.array([20, 0, 0]),
            size=np.array([4.5, 1.8, 1.5]), heading=0.0,
            obj_type='car',
        )]
        np.random.seed(42)
        dets = sensor.sense(0.0, ego, gt)
        # 应该有 1 个检测
        assert len(dets) >= 1
        assert dets[0].object_id == 1
        # 距离约 20m
        assert 19 < dets[0].attributes['range_m'] < 21


class TestRadarSensor:
    """Radar 仿真基础"""

    def test_basic_sense(self):
        sensor = RadarSensor('radar', np.array([0,0,0.5]), 20.0,
                             max_range_m=200.0, noise_std=0.1)
        ego = EgoState(timestamp=0.0, position=np.array([0,0,0]),
                       velocity=np.array([10,0,0]), acceleration=np.zeros(3),
                       heading=0.0, angular_velocity=np.zeros(3))
        from core.data_types import GroundTruthObj
        gt = [GroundTruthObj(
            object_id=1, timestamp=0.0,
            position=np.array([30, 0, 0]), velocity=np.array([20, 0, 0]),
            size=np.array([4.5, 1.8, 1.5]), heading=0.0,
            obj_type='car',
        )]
        np.random.seed(42)
        dets = sensor.sense(0.0, ego, gt)
        assert len(dets) >= 1
        # Radar 应该输出多普勒
        assert 'doppler_mps' in dets[0].attributes


class TestCameraSensor:
    """Camera 仿真基础"""

    def test_basic_sense(self):
        sensor = CameraSensor('camera', np.array([0,0,1.5]), rate_hz=10.0,
                              max_range_m=100.0, noise_std=0.1)
        ego = EgoState(timestamp=0.0, position=np.array([0,0,0]),
                       velocity=np.array([10,0,0]), acceleration=np.zeros(3),
                       heading=0.0, angular_velocity=np.zeros(3))
        from core.data_types import GroundTruthObj
        gt = [GroundTruthObj(
            object_id=1, timestamp=0.0,
            position=np.array([20, 0, 0]), velocity=np.array([0, 0, 0]),
            size=np.array([4.5, 1.8, 1.5]), heading=0.0,
            obj_type='car',
        )]
        np.random.seed(42)
        dets = sensor.sense(0.0, ego, gt)
        assert len(dets) >= 1
        assert 'bbox_2d' in dets[0].attributes
