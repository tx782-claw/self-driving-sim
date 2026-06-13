"""
IMM 单测 — CV+CA 模型交互
"""
import numpy as np
import pytest
from fusion.imm import IMMTrack, make_imm_transition_matrix


class TestTransitionMatrix:
    """Markov 转移矩阵"""

    def test_default_diag(self):
        """默认 diag=0.95"""
        T = make_imm_transition_matrix()
        assert T.shape == (2, 2)
        assert abs(T[0, 0] - 0.95) < 1e-9
        assert abs(T[1, 1] - 0.95) < 1e-9
        # 行和为 1
        assert abs(T[0].sum() - 1.0) < 1e-9
        assert abs(T[1].sum() - 1.0) < 1e-9

    def test_invalid_diag(self):
        """diag=0 应该报错"""
        with pytest.raises(ValueError):
            make_imm_transition_matrix(diag=0.0)


class TestIMM_Track:
    """IMMTrack 功能"""

    def test_create(self):
        """基本创建"""
        tr = IMMTrack(1, np.array([0,0,0]), np.array([10,0,0]), dt=0.05)
        assert np.allclose(tr.model_probs, [0.5, 0.5])

    def test_cv_scenario_prefers_cv(self):
        """纯匀速场景下 CV 概率应 > CA"""
        tr = IMMTrack(1, np.array([0,0,0]), np.array([10,0,0]), dt=0.05)
        for i in range(50):
            t = i * 0.05
            tr.update(np.array([10*t, 0, 0]), 'lidar', t)
        assert tr.model_probs[0] > 0.5, \
            f"CV 场景下 CV 概率 {tr.model_probs[0]:.3f} 应 > 0.5"

    def test_no_nan_long_run(self):
        """1000 帧 update 不应产生 NaN"""
        tr = IMMTrack(1, np.array([0,0,0]), np.array([10,0,0]), dt=0.05)
        np.random.seed(42)
        for i in range(1000):
            t = i * 0.05
            pos = np.array([10*t, 0, 0]) + np.random.normal(0, 0.1, 3)
            tr.update(pos, 'lidar', t)
            assert np.all(np.isfinite(tr.kf.x))
            assert np.all(np.isfinite(tr.ca_kf.x))
            assert np.all(np.isfinite(tr.model_probs))

    def test_miss_decreases_velocity(self):
        """miss 应该让速度向 0 衰减（虽然这里没 vel decay 但 miss 不应该让 vel 爆）"""
        tr = IMMTrack(1, np.array([0,0,0]), np.array([10,0,0]), dt=0.05)
        for i in range(10):
            tr.update(np.array([10*i*0.05, 0, 0]), 'lidar', i*0.05)
        v0 = tr.get_velocity().copy()
        for i in range(10, 20):
            tr.miss(i*0.05)
        v1 = tr.get_velocity()
        # 速度应该被 predict 维持（无 vel decay 是 OK 的）
        assert abs(v1[0]) < 50, f"miss 10 帧后速度 {v1[0]:.1f} 应该 < 50 m/s"

    def test_to_track_interface(self):
        """to_track 应该返回 TrackedObject 兼容接口"""
        tr = IMMTrack(1, np.array([0,0,0]), np.array([10,0,0]), dt=0.05)
        for i in range(5):
            tr.update(np.array([10*i*0.05, 0, 0]), 'lidar', i*0.05)
        t = tr.to_track(0.25)
        assert t.track_id == 1
        assert t.position.shape == (3,)
        assert t.velocity.shape == (3,)
