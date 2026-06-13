"""
EKF 跟踪器 - 匀速运动模型 + 自适应噪声
状态: [x, y, z, vx, vy, vz]
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from core.data_types import TrackedObject


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

    def predict(self, dt: float = None):
        if dt is None:
            dt = self.dt
        self.kf.F[:3, 3:] = np.eye(3) * dt
        # 未命中时增大过程噪声（应对机动/不确定性）
        if self.miss_streak > 0:
            noise_scale = 1.0 + min(self.miss_streak, 5) * 0.5
            self.kf.Q[3:, 3:] = np.eye(3) * (self._process_noise_vel_base * noise_scale)
        self.kf.predict()

    def update(self, det_pos: np.ndarray, sensor_id: str, timestamp: float,
               dt: float = None, det_confidence: float = 1.0,
               class_label: str = None):
        """用检测位置更新滤波器 (P2: 支持多检测，只 predict 一次)"""
        # P2 关键修复: 如果 timestamp == self.last_t，说明已预测过，不再 predict
        already_predicted = (timestamp == self.last_t)
        if not already_predicted:
            if dt is None:
                dt = timestamp - self.last_t if timestamp > self.last_t else self.dt
            self.kf.F[:3, 3:] = np.eye(3) * dt
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
