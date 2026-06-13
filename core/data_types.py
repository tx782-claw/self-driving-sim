"""
核心数据类型定义
所有模块间传递的数据结构统一在此
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class EgoState:
    """自车状态"""
    timestamp: float
    position: np.ndarray  # (3,) x, y, z
    velocity: np.ndarray  # (3,) vx, vy, vz
    acceleration: np.ndarray  # (3,)
    heading: float  # 偏航角 (rad)
    angular_velocity: np.ndarray  # (3,) roll/pitch/yaw rate


@dataclass
class GroundTruthObj:
    """仿真真实目标（上帝视角）"""
    object_id: int
    timestamp: float
    obj_type: str  # 'car', 'truck', 'pedestrian', 'cyclist'
    position: np.ndarray  # (3,)
    velocity: np.ndarray  # (3,)
    heading: float  # 偏航角
    size: np.ndarray  # (3,) length, width, height
    color: str = 'red'


@dataclass
class Detection:
    """单个传感器检测结果"""
    sensor_id: str
    timestamp: float
    position: np.ndarray  # (3,) 世界坐标
    velocity: np.ndarray  # (3,)
    object_id: Optional[int] = None  # 真值ID（融合时不用）
    confidence: float = 1.0
    attributes: dict = field(default_factory=dict)


@dataclass
class TrackedObject:
    """融合后跟踪目标"""
    track_id: int
    timestamp: float
    position: np.ndarray  # (3,)
    velocity: np.ndarray  # (3,)
    covariance: np.ndarray  # (6, 6) 状态协方差
    age: int = 1
    hits: int = 1
    miss_streak: int = 0
    source_sensors: set = field(default_factory=set)
    class_label: Optional[str] = None


@dataclass
class CameraImage:
    """相机输出"""
    sensor_id: str
    timestamp: float
    image: np.ndarray  # (H, W, 3) RGB
    detections_2d: list = field(default_factory=list)  # 2D 框


@dataclass
class LidarScan:
    """LiDAR 输出"""
    sensor_id: str
    timestamp: float
    points: np.ndarray  # (N, 3) 点云
    intensity: Optional[np.ndarray] = None  # (N,) 强度
    detections: list = field(default_factory=list)  # 目标级检测


@dataclass
class RadarTrack:
    """Radar 输出"""
    sensor_id: str
    timestamp: float
    detections: list = field(default_factory=list)  # 距离/多普勒/角度


@dataclass
class IMUReading:
    sensor_id: str
    timestamp: float
    linear_accel: np.ndarray  # (3,)
    angular_vel: np.ndarray  # (3,)


@dataclass
class GPSReading:
    sensor_id: str
    timestamp: float
    position: np.ndarray  # (3,)
    heading: float
    fix_quality: int = 1


@dataclass
class SimFrame:
    """单帧仿真数据"""
    timestamp: float
    ego_state: EgoState
    ground_truth: list  # List[GroundTruthObj]
    # 主传感器数据 (向后兼容 - 取第一个)
    camera_data: Optional[CameraImage] = None
    lidar_data: Optional[LidarScan] = None
    radar_data: Optional[RadarTrack] = None
    imu_data: Optional[IMUReading] = None
    gps_data: Optional[GPSReading] = None
    # 所有传感器数据 (支持多传感器同类型)
    all_sensors: dict = field(default_factory=dict)  # {sensor_id: data}
    tracks: list = field(default_factory=list)  # List[TrackedObject]

    def get_lidars(self) -> dict:
        """返回所有 LiDAR 数据"""
        return {sid: data for sid, data in self.all_sensors.items() if sid.startswith('lidar')}

    def get_radars(self) -> dict:
        """返回所有 Radar 数据"""
        return {sid: data for sid, data in self.all_sensors.items() if sid.startswith('radar')}

    def get_cameras(self) -> dict:
        """返回所有 Camera 数据"""
        return {sid: data for sid, data in self.all_sensors.items() if sid.startswith('camera')}

    def get_imus(self) -> dict:
        """返回所有 IMU 数据"""
        return {sid: data for sid, data in self.all_sensors.items() if sid.startswith('imu')}

    def get_gpses(self) -> dict:
        """返回所有 GPS 数据"""
        return {sid: data for sid, data in self.all_sensors.items() if sid.startswith('gps')}
