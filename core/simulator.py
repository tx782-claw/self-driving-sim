"""
主仿真器
组合所有传感器 + 场景 + 融合
"""
from .clock import SimClock
from .data_types import (
    SimFrame
)


class Simulator:
    """主仿真器（编排器）"""
    def __init__(self, scenario, sensors: dict, fusion, dt: float = 0.05, weather=None):
        """
        Args:
            scenario: 场景对象，提供 ground truth
            sensors: {sensor_id: sensor_obj}
            fusion: 融合模块
            dt: 时间步长
            weather: 天气/光照条件
        """
        self.scenario = scenario
        self.sensors = sensors
        self.fusion = fusion
        self.clock = SimClock(dt=dt)
        self.frame_count = 0
        self.weather = weather
        # 将 weather 应用到所有传感器
        for s in self.sensors.values():
            if hasattr(s, 'weather'):
                s.weather = weather

    def set_weather(self, weather):
        """动态切换天气"""
        self.weather = weather
        for s in self.sensors.values():
            if hasattr(s, 'weather'):
                s.weather = weather

    def reset(self):
        self.clock.reset()
        self.frame_count = 0
        self.fusion.reset()
        for s in self.sensors.values():
            if hasattr(s, 'reset'):
                s.reset()

    def step(self) -> SimFrame:
        """推进一帧"""
        t = self.clock.tick()
        # 1. 场景生成 ground truth
        ego, gt_objs = self.scenario.step(t)

        # 2. 每个传感器生成检测
        sensor_detections = {}
        for sid, sensor in self.sensors.items():
            dets = sensor.sense(t, ego, gt_objs)
            sensor_detections[sid] = dets

        # 3. 融合
        tracks = self.fusion.update(sensor_detections, t)

        # 4. 组装 SimFrame
        frame = SimFrame(
            timestamp=t,
            ego_state=ego,
            ground_truth=gt_objs,
            tracks=tracks,
        )
        # Attach sensor data (向后兼容 + 存全量到 all_sensors)
        all_sensors = {}
        for sid, dets in sensor_detections.items():
            sensor = self.sensors[sid]
            data = None
            if hasattr(sensor, 'last_image') and sensor.last_image is not None:
                frame.camera_data = sensor.last_image
                data = sensor.last_image
            elif hasattr(sensor, 'last_scan') and sensor.last_scan is not None:
                frame.lidar_data = sensor.last_scan
                data = sensor.last_scan
            elif hasattr(sensor, 'last_track') and sensor.last_track is not None:
                frame.radar_data = sensor.last_track
                data = sensor.last_track
            # IMU/GPS 数据也存 (但不入向后兼容字段)
            if data is not None:
                all_sensors[sid] = data
            else:
                # 为 IMU/GPS 存储最近一次检测作为原始数据
                if sid.startswith(('imu', 'gps')) and dets:
                    all_sensors[sid] = dets[0]  # 存 Detection 对象
        frame.all_sensors = all_sensors

        self.frame_count += 1
        return frame

    def run(self, n_frames: int):
        """批量运行多帧"""
        self.reset()
        self.clock.start()
        frames = []
        for _ in range(n_frames):
            frames.append(self.step())
        return frames
