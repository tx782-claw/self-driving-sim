"""
EKF 跟踪器 - 匀速运动模型 + 自适应噪声 + 可选 ego-motion 补偿
状态: [x, y, z, vx, vy, vz]

P3-A 增强 (v0.4):
- predict(dt, ego_motion) 支持自车运动补偿
- 当 ego_motion 提供时,在过程噪声中加自车运动不确定性
  (因为 IMU 也有噪声,自车运动传递到目标跟踪的协方差)
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from core.data_types import TrackedObject

# 自车运动不确定性 (IMU 估计误差传递系数)
# v0.4: 经验值,IMU 加速度 bias 典型 ~0.05 m/s²,1 帧内速度误差 ~0.0025 m/s,
# 位置误差 ~0.00006 m,可忽略;但过程噪声用更保守值提升鲁棒性
EGO_MOTION_PROCESS_NOISE_SCALE = 0.5


def make_cv_ekf(dt: float = 0.05,
                process_noise_pos: float = 0.1,
                process_noise_vel: float = 0.5,
                measure_noise: float = 0.5):
    """
    匀速 (Constant Velocity) 卡尔曼滤波器
    状态: [x, y, z, vx, vy, vz]
    """
    kf = KalmanFilter(dim_x=6, dim_z=3)

    kf.F = np.eye(6)
    for i in range(3):
        kf.F[i, i+3] = dt

    kf.H = np.zeros((3, 6))
    kf.H[0, 0] = 1
    kf.H[1, 1] = 1
    kf.H[2, 2] = 1

    # 过程噪声：位置+速度
    kf.Q = np.eye(6)
    kf.Q[:3, :3] *= process_noise_pos
    kf.Q[3:, 3:] *= process_noise_vel

    kf.R = np.eye(3) * measure_noise
    kf.P = np.eye(6) * 5.0
    return kf


def make_ca_ekf(dt: float = 0.05,
                process_noise_pos: float = 0.5,
                process_noise_vel: float = 1.0,
                process_noise_acc: float = 2.0,
                measure_noise: float = 0.5):
    """
    匀加速 (Constant Acceleration) 卡尔曼滤波器
    状态: [x, y, z, vx, vy, vz, ax, ay, az]
    """
    kf = KalmanFilter(dim_x=9, dim_z=3)

    kf.F = np.eye(9)
    for i in range(3):
        kf.F[i, i+3] = dt
        kf.F[i, i+6] = 0.5 * dt * dt
        kf.F[i+3, i+6] = dt

    kf.H = np.zeros((3, 9))
    kf.H[0, 0] = 1
    kf.H[1, 1] = 1
    kf.H[2, 2] = 1

    kf.Q = np.eye(9)
    kf.Q[:3, :3] *= process_noise_pos
    kf.Q[3:6, 3:6] *= process_noise_vel
    kf.Q[6:, 6:] *= process_noise_acc

    kf.R = np.eye(3) * measure_noise
    kf.P = np.eye(9) * 5.0
    return kf


class EKFTrack:
    """单个 EKF 跟踪器（CV 基础 + 自适应）"""
    def __init__(self, track_id: int, initial_pos: np.ndarray, initial_vel: np.ndarray = None,
                 dt: float = 0.05, timestamp: float = 0.0,
                 process_noise_vel: float = 1.0,  # 适度
                 use_confidence_weighted: bool = True,
                 max_history: int = 50):
        self.track_id = track_id
        self.dt = dt
        self.age = 0
        self.hits = 1
        self.miss_streak = 0
        self.last_t = timestamp
        self.source_sensors = set()
        self.class_label = None
        self.history = []
        self._init_positions = []  # 仅用于速度估计，避免污染 history
        self._max_init_positions = 3
        self.use_confidence_weighted = use_confidence_weighted
        # 自适应过程噪声
        self._process_noise_vel_base = process_noise_vel

        self.kf = make_cv_ekf(dt=dt, process_noise_vel=process_noise_vel)
        self.kf.x = np.zeros(6)
        self.kf.x[:3] = initial_pos
        if initial_vel is not None:
            self.kf.x[3:] = initial_vel
        else:
            # 用相邻两帧的检测估速
            self.kf.x[3:] = np.zeros(3)

    def predict(self, dt: float = None, ego_motion: dict = None):
        """
        预测步骤

        Args:
            dt: 实际时间步长 (s)。None 时用 self.dt
            ego_motion: 自车运动补偿 (P3-A 新增)
                dict 包含:
                - 'delta_position': np.array(3,) 1 帧内自车位移 (米)
                - 'delta_velocity': np.array(3,) 1 帧内自车速度变化 (米/秒) [可选]
                - 'delta_yaw': float 偏航角变化 (弧度) [可选]

        当提供 ego_motion 时,会在过程噪声中加自车运动不确定性。
        这是因为:
        1) IMU 估计自车运动有误差,这个误差会"传递"到目标跟踪
        2) 高自车运动 → 目标在 sensor frame 中位置变化大 → 跟踪不确定度大
        """
        if dt is None:
            dt = self.dt
        self.kf.F[:3, 3:] = np.eye(3) * dt
        # 未命中时增大过程噪声（应对机动/不确定性）
        if self.miss_streak > 0:
            noise_scale = 1.0 + min(self.miss_streak, 5) * 0.5
            self.kf.Q[3:, 3:] = np.eye(3) * (self._process_noise_vel_base * noise_scale)

        # P3-A: ego-motion 补偿 - 在过程噪声中加入自车运动不确定性
        if ego_motion is not None:
            self._apply_ego_motion_noise(ego_motion, dt)

        self.kf.predict()

    def _apply_ego_motion_noise(self, ego_motion: dict, dt: float):
        """
        在过程噪声中加入自车运动不确定性

        物理意义:
        - 目标在 sensor frame 中的位置 = 目标世界位置 - 自车世界位置
        - 如果自车以 v_ego 移动,自车位置变化 = v_ego * dt
        - 这个变化会被 EKF 视为"目标在 sensor frame 的相对运动"
        - 如果我们用 IMU 估计 v_ego,IMU 的噪声 σ_imu 会在 1 帧内产生
          位置不确定性 v_ego * dt + 0.5 * σ_imu * dt²
        - 这个不确定性加到过程噪声 Q
        """
        # 位置不确定性: 自车位置变化 ± IMU bias 累积
        delta_pos = ego_motion.get('delta_position')
        if delta_pos is None:
            return
        # |delta_pos| 是 1 帧内自车位移,作为位置不确定性的下界
        ego_pos_unc = np.abs(delta_pos) * EGO_MOTION_PROCESS_NOISE_SCALE
        # 加到位置过程噪声
        self.kf.Q[:3, :3] = self.kf.Q[:3, :3] + np.diag(ego_pos_unc ** 2)
        # 速度不确定性: 自车速度变化 (1 帧内) 也加到速度过程噪声
        delta_vel = ego_motion.get('delta_velocity')
        if delta_vel is not None:
            ego_vel_unc = np.abs(delta_vel) * EGO_MOTION_PROCESS_NOISE_SCALE
            self.kf.Q[3:, 3:] = self.kf.Q[3:, 3:] + np.diag(ego_vel_unc ** 2)

    def update(self, det_pos: np.ndarray, sensor_id: str, timestamp: float,
               dt: float = None, det_confidence: float = 1.0,
               class_label: str = None, ego_motion: dict = None):
        """用检测位置更新滤波器 (P2: 支持多检测，只 predict 一次; P3-A: 支持 ego_motion 补偿)"""
        # P2 关键修复: 如果 timestamp == self.last_t，说明已预测过，不再 predict
        already_predicted = (timestamp == self.last_t)
        if not already_predicted:
            if dt is None:
                dt = timestamp - self.last_t if timestamp > self.last_t else self.dt
            self.kf.F[:3, 3:] = np.eye(3) * dt
            # P3-A: ego-motion 补偿
            if ego_motion is not None:
                self._apply_ego_motion_noise(ego_motion, dt)
            self.kf.predict()
            self.last_t = timestamp
        # 多次 update 复用同一个 kf.x[:3] 作为预测状态

        # 置信度自适应观测噪声
        if self.use_confidence_weighted and det_confidence < 1.0:
            self.kf.R = np.eye(3) * (0.5 / max(0.1, det_confidence))
        else:
            self.kf.R = np.eye(3) * 0.5

        self.kf.update(det_pos)
        # 恢复正常观测噪声
        self.kf.R = np.eye(3) * 0.5

        self.age += 1
        self.hits += 1
        self.miss_streak = 0
        self.source_sensors.add(sensor_id)
        if class_label:
            self.class_label = class_label

        self.history.append({
            't': timestamp,
            'pos': self.kf.x[:3].copy(),
            'vel': self.kf.x[3:].copy(),
        })
        if len(self.history) > 100:
            self.history = self.history[-50:]

    def miss(self, timestamp: float, dt: float = None):
        """本帧未匹配上（仅预测，P2: 避免重复预测）"""
        if timestamp == self.last_t:
            # 已预测过
            return
        if dt is None:
            dt = timestamp - self.last_t if timestamp > self.last_t else self.dt
        self.kf.F[:3, 3:] = np.eye(3) * dt
        self.kf.predict()
        self.last_t = timestamp
        self.age += 1
        self.miss_streak += 1
        self.history.append({
            't': timestamp,
            'pos': self.kf.x[:3].copy(),
            'vel': self.kf.x[3:].copy(),
        })

    def to_track(self, timestamp: float) -> TrackedObject:
        return TrackedObject(
            track_id=self.track_id,
            timestamp=timestamp,
            position=self.kf.x[:3].copy(),
            velocity=self.kf.x[3:].copy(),
            covariance=self.kf.P.copy(),
            age=self.age,
            hits=self.hits,
            miss_streak=self.miss_streak,
            source_sensors=set(self.source_sensors),
            class_label=self.class_label,
        )

    def is_alive(self, max_miss: int = 5) -> bool:
        return self.miss_streak < max_miss

    def get_predicted_position(self) -> np.ndarray:
        return self.kf.x[:3].copy()
