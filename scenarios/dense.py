"""
密集场景 - 模拟拥堵/复杂路况
- DenseHighway: 20+ 辆车
- Junction: 4 方向汇入路口
- StopAndGo: 走走停停
"""
import numpy as np
from core.data_types import EgoState, GroundTruthObj
from .base import BaseScenario


class DenseHighwayScenario(BaseScenario):
    """密集高速：3 车道 × 8 辆车 = 24 辆"""
    def __init__(self, num_lanes: int = 3, cars_per_lane: int = 8,
                 duration: float = 30.0, dt: float = 0.05,
                 ego_speed_mps: float = 25.0, lane_width_m: float = 3.5):
        super().__init__(duration, dt)
        self.num_lanes = num_lanes
        self.cars_per_lane = cars_per_lane
        self.ego_speed = ego_speed_mps
        self.lane_width = lane_width_m
        self.vehicles = self._init_vehicles()

    def _init_vehicles(self):
        """初始化：每车道 cars_per_lane 辆"""
        configs = []
        vid = 1
        for lane_idx in range(self.num_lanes):
            # 自车在中间车道 (lane=0)
            lane_offset = lane_idx - 1  # -1, 0, 1
            for car_idx in range(self.cars_per_lane):
                # 起始位置交错分布
                x0 = -50 + car_idx * 15 + (lane_idx * 5)
                # 速度变化
                speed = self.ego_speed + (car_idx - 4) * 2.0 + (lane_idx - 1) * 1.0
                speed = max(15.0, min(35.0, speed))  # 限速 15-35 m/s
                obj_type = 'car' if car_idx % 4 != 0 else 'truck'
                size = [4.5, 1.8, 1.5] if obj_type == 'car' else [8.0, 2.5, 3.0]
                color = ['red', 'blue', 'green', 'yellow', 'white'][car_idx % 5]
                configs.append((x0, lane_offset, speed, obj_type, color, size))
                vid += 1
        return configs

    def step(self, t: float) -> tuple:
        # 自车
        ego = EgoState(
            timestamp=t,
            position=np.array([self.ego_speed * t, 0.0, 0.0]),
            velocity=np.array([self.ego_speed, 0.0, 0.0]),
            acceleration=np.zeros(3),
            heading=0.0,
            angular_velocity=np.zeros(3),
        )
        gt_objs = []
        for i, (x0, lane_off, speed, obj_type, color, size) in enumerate(self.vehicles):
            x = x0 + speed * t
            y = lane_off * self.lane_width
            gt_objs.append(GroundTruthObj(
                object_id=i + 1,
                timestamp=t,
                obj_type=obj_type,
                position=np.array([x, y, size[2] / 2]),
                velocity=np.array([speed, 0.0, 0.0]),
                heading=0.0,
                size=np.array(size),
                color=color,
            ))
        return ego, gt_objs

    def reset(self):
        super().reset()


class JunctionScenario(BaseScenario):
    """4 方向汇入路口"""
    def __init__(self, num_vehicles: int = 12, duration: float = 20.0, dt: float = 0.05):
        super().__init__(duration, dt)
        self.num_vehicles = num_vehicles
        # 4 个方向的车道
        # 北: y>0, 向南移动
        # 南: y<0, 向北移动
        # 东: x>0, 向西移动
        # 西: x<0, 向东移动
        self.vehicles = self._init_vehicles()

    def _init_vehicles(self):
        """每方向 3 辆车"""
        configs = []
        vid = 1
        # 北向 (y>0, 向南 vy<0)
        for i in range(3):
            configs.append({
                'start': np.array([10 + i*15, 50.0, 0.75]),
                'vel': np.array([0.0, -8.0, 0.0]),
                'heading': -np.pi/2,
                'type': 'car', 'color': 'blue',
            })
        # 南向
        for i in range(3):
            configs.append({
                'start': np.array([-10 - i*15, -50.0, 0.75]),
                'vel': np.array([0.0, 8.0, 0.0]),
                'heading': np.pi/2,
                'type': 'car', 'color': 'green',
            })
        # 东向
        for i in range(3):
            configs.append({
                'start': np.array([50.0, 10 + i*15, 0.75]),
                'vel': np.array([-8.0, 0.0, 0.0]),
                'heading': np.pi,
                'type': 'car', 'color': 'red',
            })
        # 西向
        for i in range(3):
            configs.append({
                'start': np.array([-50.0, -10 - i*15, 0.75]),
                'vel': np.array([8.0, 0.0, 0.0]),
                'heading': 0.0,
                'type': 'car', 'color': 'yellow',
            })
        return configs[:self.num_vehicles]

    def step(self, t: float) -> tuple:
        ego = EgoState(
            timestamp=t,
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([10.0, 0.0, 0.0]),
            acceleration=np.zeros(3),
            heading=0.0,
            angular_velocity=np.zeros(3),
        )
        gt_objs = []
        for i, v in enumerate(self.vehicles):
            pos = v['start'] + v['vel'] * t
            # 加点随机扰动
            pos = pos + np.random.normal(0, 0.05, 3)
            gt_objs.append(GroundTruthObj(
                object_id=i + 1,
                timestamp=t,
                obj_type=v['type'],
                position=pos,
                velocity=v['vel'],
                heading=v['heading'],
                size=np.array([4.5, 1.8, 1.5]),
                color=v['color'],
            ))
        return ego, gt_objs

    def reset(self):
        super().reset()


class StopAndGoScenario(BaseScenario):
    """走走停停 - 模拟拥堵"""
    def __init__(self, num_vehicles: int = 8, duration: float = 30.0, dt: float = 0.05):
        super().__init__(duration, dt)
        self.num_vehicles = num_vehicles
        self.lane_width = 3.5

    def step(self, t: float) -> tuple:
        # 自车走走停停
        # 周期 10s: 0-3 加速, 3-7 匀速, 7-10 减速
        cycle = t % 10.0
        if cycle < 3.0:
            ego_speed = 5.0 + cycle * 5.0
        elif cycle < 7.0:
            ego_speed = 20.0
        else:
            ego_speed = 20.0 - (cycle - 7.0) * 6.67
        ego_speed = max(0.0, ego_speed)

        # 累积位置（积分）
        # 简化: 用平均速度
        if not hasattr(self, '_ego_x'):
            self._ego_x = 0.0
        self._ego_x += ego_speed * self.dt

        ego = EgoState(
            timestamp=t,
            position=np.array([self._ego_x, 0.0, 0.0]),
            velocity=np.array([ego_speed, 0.0, 0.0]),
            acceleration=np.array([5.0 if cycle < 3.0 else (-6.67 if cycle > 7.0 else 0.0), 0.0, 0.0]),
            heading=0.0,
            angular_velocity=np.zeros(3),
        )

        gt_objs = []
        # 其他车辆跟随自车节奏
        for i in range(self.num_vehicles - 1):
            offset = -10.0 - i * 8.0
            x = self._ego_x + offset
            y = (i % 2) * 3.5  # 0 或 3.5
            gt_objs.append(GroundTruthObj(
                object_id=i + 1,
                timestamp=t,
                obj_type='car',
                position=np.array([x, y, 0.75]),
                velocity=np.array([ego_speed, 0.0, 0.0]),
                heading=0.0,
                size=np.array([4.5, 1.8, 1.5]),
                color='blue',
            ))
        return ego, gt_objs

    def reset(self):
        super().reset()
        self._ego_x = 0.0
