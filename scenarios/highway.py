"""
高速公路场景
自车 + 4-6 辆周围车辆
"""
import numpy as np
from core.data_types import EgoState, GroundTruthObj
from .base import BaseScenario


class HighwayScenario(BaseScenario):
    """高速公路 - 多车道直线行驶场景"""
    def __init__(self, num_vehicles: int = 5, duration: float = 20.0, dt: float = 0.05,
                 ego_speed_mps: float = 25.0, lane_width_m: float = 3.5):
        super().__init__(duration, dt)
        self.num_vehicles = num_vehicles
        self.ego_speed = ego_speed_mps
        self.lane_width = lane_width_m

        # 预定义其他车辆参数
        # (起始 x 位置, 车道偏移, 速度, 目标类型, 颜色, 尺寸)
        self.other_vehicles = self._init_vehicles()

    def _init_vehicles(self):
        """初始化其他车辆（车道 -2, -1, 0, 1, 2）"""
        configs = [
            # 前方慢车
            (40.0, 0.0, 18.0, 'car', 'blue', [4.5, 1.8, 1.5]),
            # 前方右侧车
            (35.0, 1.0, 22.0, 'car', 'green', [4.5, 1.8, 1.5]),
            # 前方左侧车
            (50.0, -1.0, 26.0, 'car', 'white', [4.5, 1.8, 1.5]),
            # 后方左侧车
            (-15.0, -1.0, 28.0, 'car', 'yellow', [4.5, 1.8, 1.5]),
            # 后方右侧车
            (-20.0, 1.0, 30.0, 'truck', 'red', [8.0, 2.5, 3.0]),
        ]
        return configs[:self.num_vehicles]

    def step(self, t: float) -> tuple:
        # 自车沿 x 方向匀速行驶
        ego = EgoState(
            timestamp=t,
            position=np.array([self.ego_speed * t, 0.0, 0.0]),
            velocity=np.array([self.ego_speed, 0.0, 0.0]),
            acceleration=np.array([0.0, 0.0, 0.0]),
            heading=0.0,
            angular_velocity=np.array([0.0, 0.0, 0.0]),
        )

        # 其他车辆：匀速
        gt_objs = []
        for i, (x0, lane_off, speed, obj_type, color, size) in enumerate(self.other_vehicles):
            x = x0 + speed * t
            y = lane_off * self.lane_width
            gt_objs.append(GroundTruthObj(
                object_id=i + 1,
                timestamp=t,
                obj_type=obj_type,
                position=np.array([x, y, size[2] / 2]),  # z 中心
                velocity=np.array([speed, 0.0, 0.0]),
                heading=0.0,
                size=np.array(size),
                color=color,
            ))

        return ego, gt_objs

    def reset(self):
        super().reset()


class UrbanScenario(BaseScenario):
    """城市场景 - 十字路口 + 行人"""
    def __init__(self, num_vehicles: int = 4, num_pedestrians: int = 2,
                 duration: float = 20.0, dt: float = 0.05,
                 ego_speed_mps: float = 10.0):
        super().__init__(duration, dt)
        self.num_vehicles = num_vehicles
        self.num_pedestrians = num_pedestrians
        self.ego_speed = ego_speed_mps

    def step(self, t: float) -> tuple:
        # 自车低速直行
        ego = EgoState(
            timestamp=t,
            position=np.array([self.ego_speed * t, 0.0, 0.0]),
            velocity=np.array([self.ego_speed, 0.0, 0.0]),
            acceleration=np.zeros(3),
            heading=0.0,
            angular_velocity=np.zeros(3),
        )

        gt_objs = []
        # 4 辆车 - 横向穿过（Y 方向）
        # vehicle i 在 x=20,40,60,80
        for i in range(self.num_vehicles):
            x_pos = 30.0 + i * 20.0
            # 横向速度，从右向左穿过自车前方
            vx = 0.0
            vy = 8.0 if i % 2 == 0 else -8.0
            y_pos = -15.0 + vy * t
            gt_objs.append(GroundTruthObj(
                object_id=i + 1,
                timestamp=t,
                obj_type='car',
                position=np.array([x_pos, y_pos, 0.75]),
                velocity=np.array([vx, vy, 0.0]),
                heading=np.pi/2 if vy > 0 else -np.pi/2,
                size=np.array([4.5, 1.8, 1.5]),
                color='blue' if i % 2 == 0 else 'green',
            ))

        # 行人 - 沿 z 方向走过
        for i in range(self.num_pedestrians):
            x_pos = 20.0 + i * 15.0
            # 来回走动
            y_pos = 3.0 * np.sin(0.5 * t + i)
            gt_objs.append(GroundTruthObj(
                object_id=100 + i,
                timestamp=t,
                obj_type='pedestrian',
                position=np.array([x_pos, y_pos, 0.9]),
                velocity=np.array([0.0, 1.5 * np.cos(0.5 * t + i), 0.0]),
                heading=0.0,
                size=np.array([0.5, 0.5, 1.8]),
                color='orange',
            ))

        return ego, gt_objs

    def reset(self):
        super().reset()
