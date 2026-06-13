"""
传感器基类
"""
from abc import ABC, abstractmethod
import numpy as np
from core.data_types import EgoState


class BaseSensor(ABC):
    """所有传感器的基类"""
    def __init__(self, sensor_id: str, rate_hz: float, position: np.ndarray = None,
                 heading: float = 0.0, noise_std: float = 0.05):
        """
        Args:
            sensor_id: 唯一标识
            rate_hz: 采样率
            position: 相对自车位置 (3,)
            heading: 相对自车朝向 (rad)
            noise_std: 噪声标准差
        """
        self.sensor_id = sensor_id
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.position = np.zeros(3) if position is None else np.array(position, dtype=float)
        self.heading = heading
        self.noise_std = noise_std
        self.last_t = -1e9
        # P2: 天气/光照影响 (子类可覆盖)
        self.weather = None

    def _apply_weather_noise(self, base_noise: float) -> float:
        """根据天气调整噪声"""
        if self.weather is None:
            return base_noise
        # 各传感器的天气影响系数
        factors = {
            'lidar': getattr(self.weather, 'lidar_noise_factor', 1.0),
            'radar': getattr(self.weather, 'radar_noise_factor', 1.0),
            'camera': getattr(self.weather, 'camera_noise_factor', 1.0),
        }
        # 从 sensor_id 前缀判断类型
        for prefix, f in factors.items():
            if self.sensor_id.startswith(prefix):
                return base_noise * f
        return base_noise

    def _apply_weather_range(self, base_range: float) -> float:
        """根据天气调整有效距离"""
        if self.weather is None:
            return base_range
        factors = {
            'lidar': getattr(self.weather, 'lidar_range_factor', 1.0),
            'radar': getattr(self.weather, 'radar_range_factor', 1.0),
            'camera': getattr(self.weather, 'camera_range_factor', 1.0),
        }
        for prefix, f in factors.items():
            if self.sensor_id.startswith(prefix):
                return base_range * f
        return base_range

    @abstractmethod
    def sense(self, t: float, ego: EgoState, gt_objs: list) -> list:
        """
        执行一次感知
        Returns:
            list of Detection
        """
        pass

    def _should_fire(self, t: float) -> bool:
        """检查是否到达本传感器采样时间"""
        if t - self.last_t < self.dt - 1e-6:
            return False
        self.last_t = t
        return True

    def reset(self):
        self.last_t = -1e9

    @staticmethod
    def _add_noise(value: np.ndarray, std: float) -> np.ndarray:
        if std <= 0:
            return value
        return value + np.random.normal(0, std, size=value.shape)
