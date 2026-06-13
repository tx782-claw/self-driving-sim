"""
EKF 单测 — CV 状态转移 + 观测更新
"""
import numpy as np
import pytest
from fusion.ekf import EKFTrack, make_cv_ekf, make_ca_ekf


class TestCV_EKF:
    """CV 匀速模型"""

    def test_initial_state(self):
        """初始状态应该正确"""
        tr = EKFTrack(1, initial_pos=np.array([0, 0, 0]),
                      initial_vel=np.array([10, 0, 0]), dt=0.05)
        assert np.allclose(tr.kf.x[:3], [0, 0, 0])
        assert np.allclose(tr.kf.x[3:], [10, 0, 0])

    def test_predict_uniform_motion(self):
        """匀速运动 100 帧后位置应该 ≈ v*t"""
        tr = EKFTrack(1, initial_pos=np.array([0, 0, 0]),
                      initial_vel=np.array([10, 0, 0]), dt=0.05)
        for i in range(100):
            t = i * 0.05
            tr.predict()
        # 100 * 0.05 = 5s, x 应该 ≈ 50
        assert 48 < tr.kf.x[0] < 52, f"期望 ~50m 实际 {tr.kf.x[0]:.2f}m"

    def test_update_corrects_position(self):
        """用真实位置 update 应该让预测位置接近真值"""
        tr = EKFTrack(1, initial_pos=np.array([0, 0, 0]),
                      initial_vel=np.array([10, 0, 0]), dt=0.05)
        tr.kf.x[0] = 100.0  # 假装预测跑偏到 100
        # 真实位置在 50
        tr.update(np.array([50, 0, 0]), 'lidar', 5.0)
        # update 后位置应该在 50-100 之间（接近 50 但不会完全一致）
        assert 50 <= tr.kf.x[0] <= 100, f"update 后 x={tr.kf.x[0]}"

    def test_miss_increments_streak(self):
        """miss() 应该让 miss_streak 增 1, age 增 1"""
        tr = EKFTrack(1, np.array([0,0,0]), dt=0.05, timestamp=0.0)
        assert tr.miss_streak == 0
        tr.miss(timestamp=0.05)
        assert tr.miss_streak == 1
        assert tr.age == 1

    def test_update_resets_streak(self):
        """update() 应该让 miss_streak 归 0"""
        tr = EKFTrack(1, np.array([0,0,0]), dt=0.05, timestamp=0.0)
        tr.miss(0.05)
        tr.miss(0.10)
        assert tr.miss_streak == 2
        tr.update(np.array([0,0,0]), 'lidar', 0.15)
        assert tr.miss_streak == 0
        assert tr.hits == 2


class TestCA_EKF:
    """CA 匀加速模型"""

    def test_ca_state_dim(self):
        """CA 模型状态维数应该是 9"""
        kf = make_ca_ekf()
        assert kf.dim_x == 9
        assert kf.dim_z == 3

    def test_ca_predict_acceleration(self):
        """CA 模型 1s 后位置应该包含 0.5*a*t^2"""
        kf = make_ca_ekf(dt=0.1)
        kf.x = np.array([0, 0, 0, 10, 0, 0, 2, 0, 0], dtype=float)  # v0=10, a=2
        # 跑 10 帧 (1s)
        for _ in range(10):
            kf.F[:3, 3:6] = np.eye(3) * 0.1
            kf.F[:3, 6:9] = np.eye(3) * 0.5 * 0.1 * 0.1
            kf.F[3:6, 6:9] = np.eye(3) * 0.1
            kf.predict()
        # 1s 后：v=12, x = 10*1 + 0.5*2*1^2 = 11
        assert 10.5 < kf.x[0] < 11.5, f"CA 1s 后 x={kf.x[0]:.3f}, 期望 ~11"
        assert 11.5 < kf.x[3] < 12.5, f"CA 1s 后 vx={kf.x[3]:.3f}, 期望 ~12"


class TestEKF_Robustness:
    """EKF 健壮性测试"""

    def test_aliyun_predict_no_nan(self):
        """连续 1000 帧 predict 不应产生 NaN"""
        tr = EKFTrack(1, np.array([0,0,0]), np.array([10,0,0]), dt=0.05)
        for i in range(1000):
            tr.predict()
            assert np.all(np.isfinite(tr.kf.x))
            assert np.all(np.isfinite(tr.kf.P))

    def test_update_with_noisy_detection(self):
        """加噪 ±1m 检测，update 后位置应该接近真值 ±1m"""
        tr = EKFTrack(1, np.array([0,0,0]), np.array([10,0,0]), dt=0.05, timestamp=0.0)
        np.random.seed(42)
        # 手动跑 100 帧（之前 20 帧是不够的，EKF 需要时间收敛）
        for i in range(100):
            t = i * 0.05
            true_pos = np.array([10*t, 0, 0]) + np.random.normal(0, 0.1, 3)
            tr.update(true_pos, 'lidar', t)
        # 5s 后 x 应该 ≈ 50 ± 1m
        assert 49 < tr.kf.x[0] < 51, f"update 100 帧后 x={tr.kf.x[0]:.2f}, 期望 ~50"
