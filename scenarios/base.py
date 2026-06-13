"""
场景基类
"""
from abc import ABC, abstractmethod


class BaseScenario(ABC):
    """场景基类"""
    def __init__(self, duration: float = 20.0, dt: float = 0.05):
        self.duration = duration
        self.dt = dt
        self.t = 0.0

    @abstractmethod
    def step(self, t: float) -> tuple:
        """
        Returns:
            (EgoState, list[GroundTruthObj])
        """
        pass

    def reset(self):
        self.t = 0.0
