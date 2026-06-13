"""
LiDAR 仿真（简化版 - 机械旋转式 + 射线投射）
"""
import numpy as np
from core.data_types import EgoState, Detection, LidarScan
from .base import BaseSensor
from .range_model import LIDAR_NOISE_MODEL, RangeNoiseModel


class LidarSensor(BaseSensor):
    """
    简化版 LiDAR：
    - 多线束（如 32/64/128 线），机械 360° 旋转
    - 射线投射 + 采样噪声
    - DBSCAN 聚类 → 目标级检测
    - v0.2.2: 距离依赖噪声 + 漏检率（默认 LIDAR_NOISE_MODEL）
    """
    def __init__(self, sensor_id: str = "lidar_top",
                 position: np.ndarray = None,
                 rate_hz: float = 10.0,
                 num_lines: int = 32,
                 vertical_fov_deg: tuple = (-25.0, 25.0),
                 max_range_m: float = 80.0,
                 min_range_m: float = 1.0,
                 horizontal_resolution_deg: float = 0.2,
                 noise_std: float = 0.02,
                 cluster_eps_m: float = 0.8,
                 cluster_min_points: int = 5,
                 range_model: RangeNoiseModel = None):
        super().__init__(sensor_id, rate_hz, position, noise_std=noise_std)
        self.num_lines = num_lines
        self.min_v, self.max_v = np.deg2rad(vertical_fov_deg)
        self.max_range = max_range_m
        self.min_range = min_range_m
        self.h_res = np.deg2rad(horizontal_resolution_deg)

        # 预计算垂直角度
        self.vertical_angles = np.linspace(self.min_v, self.max_v, num_lines)

        # 聚类参数
        self.cluster_eps = cluster_eps_m
        self.cluster_min_points = cluster_min_points

        # v0.2.2: 距离依赖噪声模型
        # 默认用 LIDAR_NOISE_MODEL；用户可传入自定义
        if range_model is None:
            range_model = RangeNoiseModel(
                near_m=5.0, far_m=max_range_m,
                sigma_near=noise_std,
                sigma_far=noise_std * 7.5,
                miss_rate_near=0.0, miss_rate_far=0.05,
                max_range_m=max_range_m,
            )
        self.range_model = range_model

        self.last_scan = None

    def sense(self, t: float, ego: EgoState, gt_objs: list) -> list:
        if not self._should_fire(t):
            return self.last_scan.detections if self.last_scan else []

        # 1. 射线投射生成点云
        points = self._raycast(ego, gt_objs)
        intensity = np.ones(len(points)) * 0.8

        # 2. 目标级检测 (基于真实 GT 加噪声 + DBSCAN 聚类)
        detections = self._detect_objects(t, ego, gt_objs, points)

        self.last_scan = LidarScan(
            sensor_id=self.sensor_id,
            timestamp=t,
            points=points,
            intensity=intensity,
            detections=detections
        )
        return detections

    def _raycast(self, ego: EgoState, gt_objs: list) -> np.ndarray:
        """
        简化射线投射：每个目标生成其表面点云
        """
        sensor_world_pos = self._sensor_world_pos(ego)
        all_points = []

        # 应用天气影响的有效距离
        effective_max_range = self._apply_weather_range(self.max_range)
        effective_noise = self._apply_weather_noise(self.noise_std)

        for gt in gt_objs:
            # 目标中心（世界坐标）
            cx, cy, cz = gt.position
            sx, sy, sz = gt.size  # length, width, height
            heading = gt.heading

            # 目标相对传感器的位置
            dx = cx - sensor_world_pos[0]
            dy = cy - sensor_world_pos[1]
            dz = cz - sensor_world_pos[2]

            # 目标在传感器系下的距离
            range_h = np.sqrt(dx**2 + dy**2)
            if range_h > effective_max_range or range_h < self.min_range:
                continue

            # 简化为"目标体表采样" - 取目标朝向线
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            # 在目标表面均匀采样
            n_samples_per_face = max(3, int(range_h / 5))  # 距离越近采样越多
            for _ in range(n_samples_per_face):
                # 在目标 box 表面随机采样
                l_off = (np.random.rand() - 0.5) * sx
                w_off = (np.random.rand() - 0.5) * sy
                h_off = (np.random.rand() - 0.5) * sz
                px = cx + l_off * cos_h - w_off * sin_h
                py = cy + l_off * sin_h + w_off * cos_h
                pz = cz + h_off
                # 加距离相关噪声 (含天气影响)
                sigma = effective_noise * (1 + range_h / 30)
                px += np.random.normal(0, sigma)
                py += np.random.normal(0, sigma)
                pz += np.random.normal(0, sigma)
                all_points.append([px, py, pz])

        if len(all_points) == 0:
            return np.zeros((0, 3))
        return np.array(all_points)

    def _detect_objects(self, t, ego, gt_objs, points) -> list:
        """
        目标级检测：基于真实 GT 加 LiDAR 噪声（v0.2.2 用 RangeNoiseModel）
        """
        detections = []
        sensor_world_pos = self._sensor_world_pos(ego)
        effective_max_range = self._apply_weather_range(self.max_range)
        for gt in gt_objs:
            # 距离过滤
            dist = np.linalg.norm(gt.position - sensor_world_pos)
            if dist > effective_max_range or dist < self.min_range:
                continue
            # v0.2.2: 用 RangeNoiseModel 分段模型（带漏检）
            if not self.range_model.sample(dist):
                continue  # 漏检
            sigma, _ = self.range_model.at(dist)
            # 天气加成
            sigma *= self._apply_weather_noise(1.0)
            pos = gt.position + np.random.normal(0, sigma, 3)
            # 速度噪声
            vel_noise = sigma * 0.5
            vel = gt.velocity + np.random.normal(0, vel_noise, 3)
            # 置信度（距离相关）
            conf = max(0.3, 1.0 - dist / effective_max_range)

            detections.append(Detection(
                sensor_id=self.sensor_id,
                timestamp=t,
                position=pos,
                velocity=vel,
                object_id=gt.object_id,
                confidence=conf,
                attributes={
                    'num_points': max(5, int(50 / (1 + dist/10))),
                    'intensity_avg': 0.8,
                    'range_m': float(dist),
                }
            ))
        return detections

    def _sensor_world_pos(self, ego: EgoState) -> np.ndarray:
        """传感器世界坐标（简化为：自车位置 + 相对偏移，不考虑自车朝向旋转）"""
        return ego.position + self.position
