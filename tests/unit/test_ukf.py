"""
UKF 单测 — sigma points + UT 变换
"""
import numpy as np
import pytest
from fusion.ukf import UKFTrack, sigma_points


class TestSigmaPoints:
    """Sigma points 生成"""

    def test_basic_sigma_count(self):
        """n 维状态应该生成 2n+1 个 sigma points"""
        x = np.array([1.0, 2.0, 3.0])
        P = np.eye(3)
        sigmas, Wm, Wc = sigma_points(x, P)
        assert sigmas.shape == (7, 3)
        assert len(Wm) == 7
        assert len(Wc) == 7

    def test_weights_sum_to_one(self):
        """sigma point 权重应该和为 1（均值权重）"""
        x = np.zeros(3)
        P = np.eye(3)
        _, Wm, _ = sigma_points(x, P)
        assert abs(Wm.sum() - 1.0) < 1e-9

    def test_first_sigma_equals_mean(self):
        """第 0 个 sigma point 应该等于均值"""
        x = np.array([1.0, 2.0, 3.0])
        P = np.eye(3) * 2.0
        sigmas, _, _ = sigma_points(x, P)
        np.testing.assert_allclose(sigmas[0], x)

    def test_handles_singular_P(self):
        """奇异协方差不应该崩（应该正则化）"""
        x = np.array([1.0, 2.0])
        P = np.array([[1.0, 0.0], [0.0, 0.0]])  # 半正定
        sigmas, Wm, Wc = sigma_points(x, P)
        # 至少不应该崩
        assert sigmas.shape == (5, 2)
        assert np.all(np.isfinite(sigmas))


class TestUKF_Track:
    """UKFTrack 基本功能"""

    def test_create_track(self):
        """基本初始化"""
        tr = UKFTrack(1, initial_pos=np.array([0,0,0]),
                      initial_vel=np.array([10,0,0]), dt=0.05)
        assert tr.track_id == 1
        assert np.allclose(tr.kf.x[:3], [0, 0, 0])

    def test_predict_uniform(self):
        """UKF 匀速运动 5s 后位置应该 ~50m"""
        tr = UKFTrack(1, np.array([0,0,0]), np.array([10,0,0]), dt=0.05)
        for i in range(100):
            t = i * 0.05
            tr.update(np.array([10*t, 0, 0]), 'lidar', t)
        assert 48 < tr.kf.x[0] < 52, f"UKF 5s 后 x={tr.kf.x[0]:.2f}"
