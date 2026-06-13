"""
Camera 仿真（简化版）
- 针孔投影模型
- 2D 目标检测框（基于 GT 投影）
- 不渲染真实图像（用占位图）
- v0.2.2: 距离依赖噪声 + 漏检率（用 CAMERA_NOISE_MODEL）
"""
import numpy as np
from core.data_types import EgoState, Detection, CameraImage
from .base import BaseSensor
from .range_model import RangeNoiseModel


class CameraSensor(BaseSensor):
    """
    针孔相机模型
    - 内参：焦距、主点
    - 外参：相对自车位置 + 朝向
    - 输出：2D bbox + 深度估计
    - v0.2.2: 距离依赖噪声 + 漏检率
    """
    def __init__(self, sensor_id: str = "camera_front",
                 position: np.ndarray = None,
                 heading: float = 0.0,
                 rate_hz: float = 10.0,
                 image_size: tuple = (640, 480),  # (W, H)
                 fov_horizontal_deg: float = 90.0,
                 max_range_m: float = 100.0,
                 min_range_m: float = 1.0,
                 noise_std: float = 0.1,
                 range_model: RangeNoiseModel = None):
        super().__init__(sensor_id, rate_hz, position, heading, noise_std=noise_std)
        self.W, self.H = image_size
        self.max_range = max_range_m
        self.min_range = min_range_m
        self.fov_h = np.deg2rad(fov_horizontal_deg)
        self.fov_v = 2 * np.arctan(np.tan(self.fov_h / 2) * self.H / self.W)
        # 焦距（像素）
        self.fx = self.W / (2 * np.tan(self.fov_h / 2))
        self.fy = self.H / (2 * np.tan(self.fov_v / 2))
        self.cx = self.W / 2
        self.cy = self.H / 2

        # v0.2.2: 距离依赖噪声模型
        if range_model is None:
            range_model = RangeNoiseModel(
                near_m=2.0, far_m=max_range_m,
                sigma_near=noise_std, sigma_far=noise_std * 10,
                miss_rate_near=0.02, miss_rate_far=0.30,
                max_range_m=max_range_m,
            )
        self.range_model = range_model

        # 简化：占位图像
        self.placeholder_img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        # 画网格方便观察
        for i in range(0, self.H, 40):
            self.placeholder_img[i, :, :] = 30
        for j in range(0, self.W, 40):
            self.placeholder_img[:, j, :] = 30

        self.last_image = None
        self._last_3d_dets = []

    def sense(self, t: float, ego: EgoState, gt_objs: list) -> list:
        if not self._should_fire(t):
            # 返回上一次的 3D 检测（如果不是首次）
            return self._last_3d_dets if self._last_3d_dets else []

        detections_2d = []
        detections_3d = []
        sensor_pos = ego.position + self.position

        for gt in gt_objs:
            rel = gt.position - sensor_pos
            # 距离过滤
            range_3d = np.linalg.norm(rel)
            if range_3d > self.max_range or range_3d < self.min_range:
                continue
            # v0.2.2: 距离依赖漏检
            if not self.range_model.sample(range_3d):
                continue

            # 简化：忽略 heading 旋转，假设相机朝向 +x
            x = rel[0]
            y = rel[1]
            z = rel[2]
            if x <= 0:  # 在后面
                continue
            # FOV 检查
            az = np.arctan2(y, x)
            if abs(az) > self.fov_h / 2:
                continue

            # 目标在图像中的 2D 边界框（简化：投影目标中心 + 尺寸）
            u = self.fx * y / x + self.cx
            v = self.fy * (-z) / x + self.cy  # 假设 z up，但图像 v 朝下
            # 目标尺寸投影（简化）
            du = self.fx * gt.size[1] / x
            dv = self.fy * gt.size[0] / x
            x1 = u - du / 2
            y1 = v - dv / 2
            x2 = u + du / 2
            y2 = v + dv / 2
            # 裁剪
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.W, x2), min(self.H, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # 像素位置噪声
            u_n = u + np.random.normal(0, 2)
            v_n = v + np.random.normal(0, 2)
            # 深度估计
            depth = range_3d + np.random.normal(0, self.noise_std * range_3d / 10)

            detections_2d.append({
                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                'center_px': (float(u_n), float(v_n)),
                'depth_m': float(depth),
                'class': gt.obj_type,
                'object_id': gt.object_id,
            })

            # 也生成 3D 检测（用于融合）
            # v0.2.2: 距离依赖噪声 (RangeNoiseModel)
            sigma, _ = self.range_model.at(range_3d)
            pos_noisy = gt.position + np.random.normal(0, sigma, 3)
            vel_noisy = gt.velocity + np.random.normal(0, sigma * 0.5, 3)
            conf = max(0.4, 1.0 - range_3d / self.max_range)
            detections_3d.append(Detection(
                sensor_id=self.sensor_id,
                timestamp=t,
                position=pos_noisy,
                velocity=vel_noisy,
                object_id=gt.object_id,
                confidence=conf,
                attributes={
                    'bbox_2d': (float(x1), float(y1), float(x2), float(y2)),
                    'class': gt.obj_type,
                    'depth_m': float(depth),
                }
            ))

        # 拷贝占位图 + 画框
        img = self.placeholder_img.copy()
        for det in detections_2d:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            color = (0, 255, 0)
            # 画 2 点
            for i in range(3):
                yy1, yy2 = max(0, y1-i), min(self.H, y2+i)
                xx1, xx2 = max(0, x1-i), min(self.W, x2+i)
                if yy2 > yy1 and xx2 > xx1:
                    img[yy1, xx1:xx2, :] = color
                    img[yy2-1, xx1:xx2, :] = color
                    img[yy1:yy2, xx1, :] = color
                    img[yy1:yy2, xx2-1, :] = color

        self.last_image = CameraImage(
            sensor_id=self.sensor_id,
            timestamp=t,
            image=img,
            detections_2d=detections_2d
        )
        self._last_3d_dets = detections_3d
        return detections_3d
