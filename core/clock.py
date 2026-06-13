"""
仿真时钟
"""
import time


class SimClock:
    """固定步长仿真时钟"""
    def __init__(self, dt: float = 0.05, start_time: float = 0.0):
        """
        Args:
            dt: 时间步长（秒），默认 50ms = 20Hz
            start_time: 起始时间
        """
        self.dt = dt
        self.t = start_time
        self.running = False
        self.start_wall_time = 0.0

    def reset(self):
        self.t = 0.0
        self.running = False

    def start(self):
        self.running = True
        self.start_wall_time = time.time()

    def pause(self):
        self.running = False

    def tick(self) -> float:
        """推进一个时间步"""
        if self.running:
            self.t += self.dt
        return self.t

    def elapsed(self) -> float:
        """仿真已运行时间（秒）"""
        return self.t

    def real_time_factor(self) -> float:
        """实时性指标 (仿真时间 / 实际时间)"""
        if not self.running:
            return 0.0
        wall = time.time() - self.start_wall_time
        if wall == 0:
            return 0.0
        return self.t / wall
