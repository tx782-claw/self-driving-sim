"""
IMU + GPS 传感器仿真
- IMU: 高频 (100Hz) 加速度 + 角速度
- GPS: 低频 (1Hz) 绝对位置
"""
import numpy as np
from core.data_types import Detection
from .base import BaseSensor


class IMUSensor(BaseSensor):
    """IMU 仿真"""
    def __init__(self, sensor_id: str = "imu",
                 position: np.ndarray = None,
                 rate_hz: float = 100.0,
                 accel_noise_std: float = 0.05,
                 gyro_noise_std: float = 0.001,
                 bias_accel: np.ndarray = None,
                 bias_gyro: np.ndarray = None):
        super().__init__(sensor_id, rate_hz, position, noise_std=accel_noise_std)
        self.accel_noise = accel_noise_std
        self.gyro_noise = gyro_noise_std
        self.bias_accel = np.zeros(3) if bias_accel is None else np.array(bias_accel)
        self.bias_gyro = np.zeros(3) if bias_gyro is None else np.array(bias_gyro)

    def sense(self, t, ego, gt_objs):
        if not self._should_fire(t):
            return []  # IMU 不产生目标检测

        # IMU 测量 = 真值 + 偏置 + 噪声
        accel = ego.acceleration + self.bias_accel + np.random.normal(0, self.accel_noise, 3)
        gyro = ego.angular_velocity + self.bias_gyro + np.random.normal(0, self.gyro_noise, 3)

        # 简单返回：Detection 但 sensor 标记为 IMU（融合层特殊处理）
        return [Detection(
            sensor_id=self.sensor_id,
            timestamp=t,
            position=np.zeros(3, dtype=float),  # IMU 不输出位置
            velocity=np.zeros(3, dtype=float),
            confidence=1.0,
            attributes={
                'accel': accel.tolist(),
                'gyro': gyro.tolist(),
            }
        )]


class GPSSensor(BaseSensor):
    """GPS 仿真"""
    def __init__(self, sensor_id: str = "gps",
                 position: np.ndarray = None,
                 rate_hz: float = 1.0,
                 position_noise_std: float = 2.0,  # 2m 单点定位
                 heading_noise_std: float = 0.1):
        super().__init__(sensor_id, rate_hz, position, noise_std=position_noise_std)
        self.heading_noise = heading_noise_std

    def sense(self, t, ego, gt_objs):
        if not self._should_fire(t):
            return []

        pos = ego.position + np.random.normal(0, self.noise_std, 3).astype(float)
        heading = ego.heading + np.random.normal(0, self.heading_noise)

        return [Detection(
            sensor_id=self.sensor_id,
            timestamp=t,
            position=pos,
            velocity=ego.velocity.astype(float),  # GPS 也可输出速度
            confidence=0.9,
            attributes={
                'heading': float(heading),
                'fix_quality': 1,
            }
        )]
