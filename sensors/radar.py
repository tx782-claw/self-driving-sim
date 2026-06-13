"""
Radar 仿真（简化版 - 复用你 FMCW 思路）
输出：距离 + 多普勒（径向速度） + 角度 + RCS
v0.2.2: 距离依赖噪声 + 漏检率（用 RADAR_NOISE_MODEL）
"""
import numpy as np
from core.data_types import EgoState, Detection, RadarTrack
from .base import BaseSensor
from .range_model import RangeNoiseModel


class RadarSensor(BaseSensor):
    """
    简化版 FMCW 雷达：
    - 测距（脉冲延迟 / FMCW 频率差）
    - 测速（多普勒频移）
    - 测角（DBF 数字波束形成 / MUSIC）
    - 加目标 RCS + 噪声模型
    - v0.2.2: 距离依赖噪声 + 漏检率
    """
    def __init__(self, sensor_id: str = "radar_front",
                 position: np.ndarray = None,
                 rate_hz: float = 20.0,
                 max_range_m: float = 200.0,
                 min_range_m: float = 0.5,
                 range_resolution_m: float = 0.5,
                 velocity_resolution_mps: float = 0.2,
                 angular_resolution_deg: float = 3.0,
                 fov_horizontal_deg: tuple = (-45.0, 45.0),
                 fov_vertical_deg: tuple = (-10.0, 10.0),
                 max_velocity_mps: float = 70.0,
                 noise_std: float = 0.1,
                 range_model: RangeNoiseModel = None):
        super().__init__(sensor_id, rate_hz, position, noise_std=noise_std)
        self.max_range = max_range_m
        self.min_range = min_range_m
        self.range_res = range_resolution_m
        self.vel_res = velocity_resolution_mps
        self.ang_res = np.deg2rad(angular_resolution_deg)
        self.min_h, self.max_h = np.deg2rad(fov_horizontal_deg)
        self.min_v, self.max_v = np.deg2rad(fov_vertical_deg)
        self.max_vel = max_velocity_mps

        # v0.2.2: 距离依赖噪声模型
        if range_model is None:
            range_model = RangeNoiseModel(
                near_m=2.0, far_m=max_range_m,
                sigma_near=self.range_res / 2, sigma_far=self.range_res * 6,
                miss_rate_near=0.10, miss_rate_far=0.40,
                max_range_m=max_range_m,
            )
        self.range_model = range_model

        self.last_track = None

    def sense(self, t: float, ego: EgoState, gt_objs: list) -> list:
        if not self._should_fire(t):
            return self.last_track.detections if self.last_track else []

        detections = []
        sensor_pos = ego.position + self.position
        # 应用天气影响
        effective_max_range = self._apply_weather_range(self.max_range)

        for gt in gt_objs:
            # 相对位置
            rel = gt.position - sensor_pos
            range_3d = np.linalg.norm(rel)
            if range_3d > effective_max_range or range_3d < self.min_range:
                continue
            # v0.2.2: 距离依赖漏检
            if not self.range_model.sample(range_3d):
                continue

            # 水平角 (相对传感器朝向)
            az = np.arctan2(rel[1], rel[0])  # 简化：忽略 self.heading
            if az < self.min_h or az > self.max_h:
                continue
            # 俯仰角
            el = np.arctan2(rel[2], np.sqrt(rel[0]**2 + rel[1]**2))
            if el < self.min_v or el > self.max_v:
                continue

            # 径向速度 (相对传感器径向方向)
            radial_dir = rel / range_3d
            v_radial = np.dot(gt.velocity, radial_dir)
            if abs(v_radial) > self.max_vel:
                continue

            # 距离/速度量化
            range_q = np.round(range_3d / self.range_res) * self.range_res
            vel_q = np.round(v_radial / self.vel_res) * self.vel_res

            # 噪声：v0.2.2 用 RangeNoiseModel（受距离 + 天气影响）
            sigma_pos_base, _ = self.range_model.at(range_3d)
            sigma_pos = sigma_pos_base * self._apply_weather_noise(1.0)
            sigma_vel = self.vel_res / 2
            sigma_ang = self.ang_res / 2

            # 极坐标位置 (range, az, el) → 笛卡尔
            r_noisy = range_3d + np.random.normal(0, sigma_pos)
            az_noisy = az + np.random.normal(0, sigma_ang)
            el_noisy = el + np.random.normal(0, sigma_ang)
            pos_noisy = sensor_pos + np.array([
                r_noisy * np.cos(el_noisy) * np.cos(az_noisy),
                r_noisy * np.cos(el_noisy) * np.sin(az_noisy),
                r_noisy * np.sin(el_noisy),
            ])
            vel_noisy = np.array([
                v_radial * np.cos(el_noisy) * np.cos(az_noisy),
                v_radial * np.cos(el_noisy) * np.sin(az_noisy),
                v_radial * np.sin(el_noisy),
            ]) + np.random.normal(0, sigma_vel, 3)

            # RCS（与小车尺寸相关）
            rcs = float(np.prod(gt.size)) * 0.5  # 简化模型
            # 距离衰减
            snr = 1.0 / (1 + (range_3d / 50) ** 2)
            conf = min(0.95, snr * (1 - np.random.rand() * 0.1))

            detections.append(Detection(
                sensor_id=self.sensor_id,
                timestamp=t,
                position=pos_noisy,
                velocity=vel_noisy,
                object_id=gt.object_id,
                confidence=conf,
                attributes={
                    'range_m': float(range_q),
                    'doppler_mps': float(vel_q),
                    'azimuth_deg': float(np.rad2deg(az_noisy)),
                    'elevation_deg': float(np.rad2deg(el_noisy)),
                    'rcs_dbsm': float(10 * np.log10(rcs + 1e-6)),
                    'snr': float(snr),
                }
            ))

        self.last_track = RadarTrack(
            sensor_id=self.sensor_id,
            timestamp=t,
            detections=detections
        )
        return detections
