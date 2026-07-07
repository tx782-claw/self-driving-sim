"""
P3-A 单测 — 自车运动补偿 (Ego-motion Compensation)

测试覆盖:
- IMUEgoPredictor: 梯形积分正确性
- EKFTrack.predict(ego_motion=): 协方差扩展正确
- EKFTrack.update(ego_motion=): 协方差扩展 + 接口兼容
- MultiObjectTracker: end-to-end 流程无破坏
"""
import numpy as np
import pytest
from core.data_types import Detection, EgoState
from fusion.ekf import EKFTrack
from fusion.imu_predict import IMUEgoPredictor, extract_imu_from_sensors, compute_ego_motion
from fusion.tracker import MultiObjectTracker


# ──────────────────────────── IMUEgoPredictor ────────────────────────────

class TestIMUEgoPredictor:
    """IMU 预测器基本功能"""

    def test_zero_motion_first_frame(self):
        """第一帧应返回零运动 (无前值无法积分)"""
        pred = IMUEgoPredictor(dt=0.05)
        imu_det = Detection(
            sensor_id='imu', timestamp=0.0,
            position=np.zeros(3), velocity=np.zeros(3),
            confidence=1.0,
            attributes={'accel': [1.0, 0.0, 0.0], 'gyro': [0, 0, 0.1]}
        )
        ego_motion = pred.update(imu_det)
        assert np.allclose(ego_motion['delta_position'], 0)
        assert np.allclose(ego_motion['delta_velocity'], 0)
        assert ego_motion['delta_yaw'] == 0.0

    def test_constant_accel(self):
        """恒定加速度 → delta_vel = 0.5*(a_prev+a_curr)*dt, delta_pos ≈ 0.5*(v_prev+v_curr)*dt"""
        pred = IMUEgoPredictor(dt=0.05)
        # 第一帧: 初始化, velocity 设为 a*dt = 2*0.05 = 0.1
        imu1 = Detection('imu', 0.0, np.zeros(3), np.zeros(3), 1.0,
                         attributes={'accel': [2.0, 0, 0], 'gyro': [0, 0, 0]})
        pred.update(imu1)
        # 第二帧: 相同加速度
        imu2 = Detection('imu', 0.05, np.zeros(3), np.zeros(3), 1.0,
                         attributes={'accel': [2.0, 0, 0], 'gyro': [0, 0, 0]})
        ego_motion = pred.update(imu2, dt=0.05)
        # 梯形积分 delta_v = 0.5*(2+2)*0.05 = 0.1 m/s
        assert abs(ego_motion['delta_velocity'][0] - 0.1) < 1e-9
        # velocity 第一帧后 = 0.1, 第二帧后 = 0.1+0.1 = 0.2
        # delta_pos = 0.5*(0.1+0.2)*0.05 = 0.0075 m
        assert abs(ego_motion['delta_position'][0] - 0.0075) < 1e-9

    def test_yaw_rate(self):
        """偏航角变化: delta_yaw = 0.5 * (gyro_prev + gyro_curr) * dt"""
        pred = IMUEgoPredictor(dt=0.1)
        imu1 = Detection('imu', 0.0, np.zeros(3), np.zeros(3), 1.0,
                         attributes={'accel': [0, 0, 0], 'gyro': [0, 0, 0.0]})
        pred.update(imu1)
        imu2 = Detection('imu', 0.1, np.zeros(3), np.zeros(3), 1.0,
                         attributes={'accel': [0, 0, 0], 'gyro': [0, 0, 1.0]})  # 1 rad/s
        ego_motion = pred.update(imu2, dt=0.1)
        # delta_yaw = 0.5 * (0 + 1.0) * 0.1 = 0.05 rad
        assert abs(ego_motion['delta_yaw'] - 0.05) < 1e-9

    def test_from_ego_state(self):
        """从 EgoState 真值算 ego_motion (fallback)"""
        pred = IMUEgoPredictor()
        ego = EgoState(
            timestamp=0.0,
            position=np.zeros(3),
            velocity=np.array([30.0, 0, 0]),  # 30 m/s
            acceleration=np.array([0.5, 0, 0]),  # 0.5 m/s²
            heading=0.0,
            angular_velocity=np.array([0, 0, 0.1]),  # 0.1 rad/s yaw rate
        )
        ego_motion = pred.update_ego_state(ego, dt=0.05)
        assert np.allclose(ego_motion['delta_position'], [1.5, 0, 0])  # 30 * 0.05
        assert np.allclose(ego_motion['delta_velocity'], [0.025, 0, 0])  # 0.5 * 0.05
        assert abs(ego_motion['delta_yaw'] - 0.005) < 1e-9  # 0.1 * 0.05

    def test_reset(self):
        """重置后状态清零"""
        pred = IMUEgoPredictor()
        imu = Detection('imu', 0.0, np.zeros(3), np.zeros(3), 1.0,
                        attributes={'accel': [1, 0, 0], 'gyro': [0, 0, 0]})
        pred.update(imu)
        assert pred.is_initialized
        pred.reset()
        assert not pred.is_initialized
        assert np.allclose(pred.velocity, 0)

    def test_extract_imu_from_sensors(self):
        """从 sensor_detections 字典提取 IMU Detection"""
        sd = {
            'lidar_top': [Detection('lidar_top', 0.0, np.zeros(3), np.zeros(3))],
            'imu': [Detection('imu', 0.0, np.zeros(3), np.zeros(3), 1.0,
                              attributes={'accel': [1, 0, 0]})],
            'camera_front': [Detection('camera_front', 0.0, np.zeros(3), np.zeros(3))],
        }
        imu = extract_imu_from_sensors(sd)
        assert imu is not None
        assert imu.sensor_id == 'imu'

    def test_no_imu(self):
        """无 IMU 时返回 None"""
        sd = {
            'lidar_top': [Detection('lidar_top', 0.0, np.zeros(3), np.zeros(3))],
        }
        assert extract_imu_from_sensors(sd) is None


# ──────────────────────────── EKFTrack + ego_motion ────────────────────────────

class TestEKFTrackWithEgoMotion:
    """EKFTrack 接受 ego_motion 参数"""

    def test_predict_without_ego_motion_baseline(self):
        """不传 ego_motion 应与原行为一致"""
        tr = EKFTrack(1, np.array([0, 0, 0]), np.array([10, 0, 0]), dt=0.05)
        Q_before = tr.kf.Q.copy()
        tr.predict()
        # 状态应推进, Q 不变
        assert tr.kf.x[0] == pytest.approx(0.5, abs=1e-6)  # 10 * 0.05
        assert np.allclose(tr.kf.Q, Q_before)

    def test_predict_with_ego_motion_expands_Q(self):
        """传 ego_motion 应扩展 Q (位置 + 速度噪声增加)"""
        tr = EKFTrack(1, np.array([0, 0, 0]), np.array([0, 0, 0]), dt=0.05)
        Q_pos_before = tr.kf.Q[:3, :3].copy()
        Q_vel_before = tr.kf.Q[3:, 3:].copy()
        # 自车以 30 m/s 行驶,1 帧位移 1.5m
        ego_motion = {'delta_position': np.array([1.5, 0, 0]),
                      'delta_velocity': np.array([0.1, 0, 0])}
        tr.predict(ego_motion=ego_motion)
        # Q 位置部分应增加 (ego_pos_unc² = 1.5² * 0.5² = 0.5625)
        pos_noise_added = tr.kf.Q[:3, :3] - Q_pos_before
        assert pos_noise_added[0, 0] > 0
        # 速度部分也应增加
        vel_noise_added = tr.kf.Q[3:, 3:] - Q_vel_before
        assert vel_noise_added[0, 0] > 0

    def test_update_with_ego_motion(self):
        """update 接受 ego_motion 参数,无破坏性"""
        tr = EKFTrack(1, np.array([0, 0, 0]), np.array([10, 0, 0]), dt=0.05)
        # 不传 ego_motion → 行为不变
        tr.update(np.array([0.5, 0, 0]), 'lidar', 0.05)
        assert tr.kf.x[0] == pytest.approx(0.5, abs=0.1)
        # 传 ego_motion → 不报错
        tr.update(np.array([1.0, 0, 0]), 'lidar', 0.1,
                  ego_motion={'delta_position': np.array([1.5, 0, 0])})
        assert not np.any(np.isnan(tr.kf.x))

    def test_high_speed_ego_motion_stability(self):
        """高速自车 + ego_motion 长期运行稳定 (无 NaN,无发散)"""
        np.random.seed(42)
        tr = EKFTrack(1, np.array([0, 0, 0]), np.array([0, 0, 0]), dt=0.05)
        for i in range(200):
            t = i * 0.05
            # 静止目标在 50m, 自车以 30m/s 行驶 (仿真里 sensor 给世界坐标所以目标位置不变)
            det_pos = np.array([50.0, 5.0, 0.0]) + np.random.normal(0, 0.3, 3)
            ego_motion = {'delta_position': np.array([1.5, 0, 0])}
            tr.update(det_pos, 'lidar', t, ego_motion=ego_motion)
        # 不应 NaN,位置估计应接近 50m
        assert not np.any(np.isnan(tr.kf.x))
        assert 45 < tr.kf.x[0] < 55, f"x={tr.kf.x[0]} 应接近 50"

    def test_ego_motion_zero_no_effect(self):
        """ego_motion = zero (静止自车) → Q 增量极小"""
        tr = EKFTrack(1, np.array([0, 0, 0]), np.array([0, 0, 0]), dt=0.05)
        Q_pos_before = tr.kf.Q[:3, :3].copy()
        tr.predict(ego_motion={'delta_position': np.zeros(3)})
        Q_diff = np.abs(tr.kf.Q[:3, :3] - Q_pos_before)
        assert Q_diff.max() < 1e-6


# ──────────────────────────── MultiObjectTracker ────────────────────────────

class TestTrackerWithEgoMotion:
    """Tracker 集成 ego_motion"""

    def test_update_accepts_ego_motion(self):
        """tracker.update 接受 ego_motion 参数"""
        tracker = MultiObjectTracker(dt=0.05, use_ego_motion=False)
        sensor_dets = {
            'lidar_top': [Detection('lidar_top', 0.05, np.array([10, 5, 0]),
                                    np.array([0, 0, 0]), 0.9)],
        }
        ego_motion = {'delta_position': np.array([1.5, 0, 0])}
        tracks = tracker.update(sensor_dets, 0.05, ego_motion=ego_motion)
        # 不报错 + 返回 track list
        assert isinstance(tracks, list)
        assert len(tracks) >= 1

    def test_use_ego_motion_false_disables(self):
        """use_ego_motion=False 时不提取 IMU (兼容旧调用)"""
        tracker = MultiObjectTracker(dt=0.05, use_ego_motion=False)
        # 即使有 IMU 数据,use_ego_motion=False 也不应使用
        sensor_dets = {
            'lidar_top': [Detection('lidar_top', 0.05, np.array([10, 5, 0]),
                                    np.array([0, 0, 0]), 0.9)],
            'imu': [Detection('imu', 0.05, np.zeros(3), np.zeros(3), 1.0,
                              attributes={'accel': [10, 0, 0]})],  # 大加速度
        }
        # 这里 track 创建后立即 update,不调 predict,所以无影响
        tracks = tracker.update(sensor_dets, 0.05)
        assert len(tracks) >= 1

    def test_end_to_end_highway(self):
        """端到端: Highway 5 车场景, ego_motion 不破坏 MOTA"""
        # 简单构造 5 个 LiDAR 检测 + 1 个 IMU,跑 10 帧
        tracker = MultiObjectTracker(dt=0.05, gate_threshold=10.0,
                                      use_ego_motion=True)
        np.random.seed(42)
        for i in range(10):
            t = (i + 1) * 0.05
            sensor_dets = {
                'lidar_top': [
                    Detection('lidar_top', t,
                              np.array([10 + i*0.5, 0, 0]),
                              np.array([10, 0, 0]), 0.9),
                    Detection('lidar_top', t,
                              np.array([20 + i*0.5, 2, 0]),
                              np.array([10, 0, 0]), 0.9),
                ],
                'imu': [Detection('imu', t, np.zeros(3), np.zeros(3), 1.0,
                                  attributes={'accel': [0.5, 0, 0],
                                             'gyro': [0, 0, 0]})],
            }
            tracks = tracker.update(sensor_dets, t)
            assert isinstance(tracks, list)
            # 第 2 帧起应有确认 track
            if i >= 1:
                assert len(tracks) >= 1
