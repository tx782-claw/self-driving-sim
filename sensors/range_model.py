"""
距离依赖传感器噪声模型 (v0.2.2 优化B)
========================================
真实传感器噪声不恒定：近距离准、远距离噪声大、超出有效范围漏检。
旧版 sensor 简单用 `sigma = base * (1 + dist/scale)` 线性放大，
本模块提供更精细的分段模型 + 漏检率曲线，方便 LiDAR/Radar/Camera 共用。

使用示例:
    model = RangeNoiseModel(
        near_m=10.0, far_m=80.0,
        sigma_near=0.02, sigma_far=0.20,
        miss_rate_near=0.0, miss_rate_far=0.30)
    sigma, miss = model.at(range_m=45.0)
    # sigma ≈ 0.11 m, miss = 0.15
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class RangeNoiseModel:
    """
    距离依赖的噪声 + 漏检率模型

    Attributes:
        near_m: 近距边界 (此范围内 sigma=sigma_near, miss=miss_rate_near)
        far_m: 远距边界 (此范围外 sigma=sigma_far, miss=miss_rate_far)
        sigma_near: 近距标准差 (m)
        sigma_far: 远距标准差 (m)
        miss_rate_near: 近距漏检率 [0,1]
        miss_rate_far: 远距漏检率 [0,1]
        max_range_m: 完全失效距离 (硬切)
    """
    near_m: float = 10.0
    far_m: float = 80.0
    sigma_near: float = 0.02
    sigma_far: float = 0.20
    miss_rate_near: float = 0.0
    miss_rate_far: float = 0.20
    max_range_m: float = 100.0

    def __post_init__(self):
        if self.far_m <= self.near_m:
            raise ValueError(f"far_m ({self.far_m}) 必须 > near_m ({self.near_m})")

    def at(self, range_m: float) -> tuple[float, float]:
        """
        返回 (sigma, miss_rate) 在给定距离处
        - 距离 < near_m: 用 near 值
        - 距离 > far_m:  用 far 值
        - 中间：线性插值
        """
        if range_m < 0:
            range_m = 0.0
        if range_m <= self.near_m:
            t = 0.0
        elif range_m >= self.far_m:
            t = 1.0
        else:
            t = (range_m - self.near_m) / (self.far_m - self.near_m)

        sigma = self.sigma_near + t * (self.sigma_far - self.sigma_near)
        miss = self.miss_rate_near + t * (self.miss_rate_far - self.miss_rate_near)
        return float(sigma), float(miss)

    def is_in_range(self, range_m: float) -> bool:
        """是否在传感器有效范围内 (range_m <= max_range_m)"""
        return 0 <= range_m <= self.max_range_m

    def sample(self, range_m: float, rng: np.random.Generator = None) -> bool:
        """
        蒙特卡洛采样：是否漏检
        Returns: True=检测到, False=漏检
        """
        if rng is None:
            rng = np.random
        _, miss = self.at(range_m)
        return rng.random() >= miss

    def apply(self, range_m: float, true_pos: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        """
        给定真值位置，返回加噪后的位置
        """
        if rng is None:
            rng = np.random
        sigma, _ = self.at(range_m)
        return true_pos + rng.normal(0, sigma, 3)


# 预设：3 种典型传感器
LIDAR_NOISE_MODEL = RangeNoiseModel(
    near_m=5.0, far_m=80.0,
    sigma_near=0.02, sigma_far=0.15,           # LiDAR 准但远距也还行
    miss_rate_near=0.0, miss_rate_far=0.05,    # 远距偶尔漏
    max_range_m=100.0,
)

RADAR_NOISE_MODEL = RangeNoiseModel(
    near_m=2.0, far_m=200.0,
    sigma_near=0.10, sigma_far=1.50,           # Radar 距离噪声比 LiDAR 大
    miss_rate_near=0.10, miss_rate_far=0.40,   # 近距 clutter 多，远距漏检多
    max_range_m=250.0,
)

CAMERA_NOISE_MODEL = RangeNoiseModel(
    near_m=2.0, far_m=60.0,
    sigma_near=0.30, sigma_far=2.00,           # Camera 测距精度最差
    miss_rate_near=0.05, miss_rate_far=0.60,   # 远距相机基本不可信
    max_range_m=80.0,
)
