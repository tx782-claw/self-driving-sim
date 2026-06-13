"""
天气/光照条件 - 影响传感器性能
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class Weather:
    """天气/光照条件"""
    name: str = "clear"           # clear/rain/fog/night
    visibility_m: float = 200.0   # 能见度
    rain_rate: float = 0.0        # 雨量 mm/h
    fog_density: float = 0.0      # 雾密度 0~1
    light_level: float = 1.0      # 光照 0~1 (1=白天, 0=夜晚)
    surface_friction: float = 1.0 # 路面摩擦系数

    # 各传感器受影响系数 (1.0=无影响, 越大越差)
    @property
    def camera_noise_factor(self) -> float:
        """夜晚相机噪声大，雨天镜头模糊"""
        f = 1.0
        if self.light_level < 0.5:
            f *= 2.5  # 夜晚噪点
        if self.rain_rate > 1.0:
            f *= 1.5  # 雨天
        if self.fog_density > 0.3:
            f *= 2.0  # 雾天
        return f

    @property
    def camera_range_factor(self) -> float:
        """相机有效距离"""
        base = 1.0
        if self.light_level < 0.5:
            base *= 0.5
        if self.fog_density > 0.3:
            base *= 0.6
        if self.rain_rate > 1.0:
            base *= 0.7
        return base

    @property
    def lidar_noise_factor(self) -> float:
        """雨雾对 LiDAR 影响较小"""
        f = 1.0
        if self.rain_rate > 5.0:
            f *= 1.8  # 大雨
        if self.fog_density > 0.7:
            f *= 1.5  # 浓雾
        return f

    @property
    def lidar_range_factor(self) -> float:
        base = 1.0
        if self.fog_density > 0.5:
            base *= 0.7
        if self.rain_rate > 10.0:
            base *= 0.6
        return base

    @property
    def radar_noise_factor(self) -> float:
        """雷达几乎不受天气影响"""
        return 1.0 + 0.05 * self.rain_rate

    @property
    def radar_range_factor(self) -> float:
        return 0.95 if self.rain_rate > 5.0 else 1.0

    def summary(self) -> Dict[str, float]:
        return {
            'name': self.name,
            'visibility_m': self.visibility_m,
            'rain_rate': self.rain_rate,
            'fog_density': self.fog_density,
            'light_level': self.light_level,
            'camera_noise_x': self.camera_noise_factor,
            'camera_range_x': self.camera_range_factor,
            'lidar_noise_x': self.lidar_noise_factor,
            'lidar_range_x': self.lidar_range_factor,
            'radar_noise_x': self.radar_noise_factor,
            'radar_range_x': self.radar_range_factor,
        }


# 预定义天气
PRESETS = {
    'Clear Day (晴天)': Weather(name='clear', visibility_m=300.0, light_level=1.0),
    'Overcast (阴天)': Weather(name='overcast', visibility_m=200.0, light_level=0.7, fog_density=0.1),
    'Light Rain (小雨)': Weather(name='light_rain', visibility_m=150.0, rain_rate=2.5, light_level=0.8, fog_density=0.2),
    'Heavy Rain (大雨)': Weather(name='heavy_rain', visibility_m=80.0, rain_rate=10.0, light_level=0.6, fog_density=0.4, surface_friction=0.7),
    'Fog (雾)': Weather(name='fog', visibility_m=60.0, fog_density=0.7, light_level=0.6, surface_friction=0.8),
    'Night Clear (夜晚晴)': Weather(name='night_clear', visibility_m=200.0, light_level=0.1),
    'Night Rainy (夜晚雨)': Weather(name='night_rain', visibility_m=80.0, rain_rate=5.0, light_level=0.1, fog_density=0.3, surface_friction=0.7),
}


def get_weather_by_name(name: str) -> Weather:
    return PRESETS.get(name, PRESETS['Clear Day (晴天)'])
