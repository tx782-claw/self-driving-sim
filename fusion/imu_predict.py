"""
IMU 自车运动预测器 (P3-A 增强 v0.4)

目的
----
- 从 IMU 数据 (加速度 + 角速度) 推算自车在 1 帧内的运动
- 输出 ego_motion dict 给 EKF predict,补偿自车运动对目标跟踪的影响

物理模型
-------
梯形积分 (在 ego frame 中,因为 IMU 测的是"自车坐标系下的运动"):
1. delta_velocity = 0.5 * (a_prev + a_curr) * dt
2. velocity += delta_velocity
3. delta_position = 0.5 * (v_prev + v_curr) * dt

参考: Titterton & Weston "Strapdown Inertial Navigation Technology" 2004

接口
----
- update(imu_det, dt) -> dict
  - imu_det: IMU Detection, 含 attributes['accel']/['gyro'] (世界坐标系下测量)
  - dt: 1 帧时间 (s)
  - 返回: {'delta_position': np.array(3,), 'delta_velocity': np.array(3,), 'delta_yaw': float}

或用 EgoState 直接:
- update_ego_state(ego_state, dt) -> dict
  - 简单用 ego.velocity * dt 算位置变化
  - 用于无 IMU 时的 fallback
"""
import numpy as np
from core.data_types import Detection, EgoState


class IMUEgoPredictor:
    """
    IMU 自车运动预测器

    维护内部状态:
    - velocity: 自车当前速度 (世界坐标系) 累积估计
    - prev_accel: 上一帧 IMU 加速度 (用于梯形积分)
    - prev_yaw: 上一帧偏航角
    - prev_gyro_z: 上一帧 z 轴角速度 (yaw rate)

    注意
    ----
    当前 self-driving-sim 的 sensor 输出已经是"世界坐标真值 + 噪声",
    所以 sensor 检测里的"目标世界位置"不会因为自车运动而变化。
    但当未来 sensor 改为"基于 t-1 时刻真值 + 噪声" (更真实),
    fusion 层会需要 IMUEgoPredictor 把检测外推到当前时刻。
    本次 P3-A 实施的是它的基础版,先用作 EKF process_noise 的输入。
    """

    def __init__(self, dt: float = 0.05):
        self.dt = dt
        self.velocity = np.zeros(3)  # 世界坐标系
        self.prev_accel = None
        self.prev_gyro_z = 0.0
        self.prev_yaw = 0.0
        self.is_initialized = False
        # 历史 (debug 用)
        self.history = []

    def reset(self):
        """重置状态"""
        self.velocity = np.zeros(3)
        self.prev_accel = None
        self.prev_gyro_z = 0.0
        self.prev_yaw = 0.0
        self.is_initialized = False
        self.history.clear()

    def update(self, imu_det: Detection, dt: float = None) -> dict:
        """
        处理一帧 IMU 数据,返回 1 帧内自车运动

        Args:
            imu_det: IMU Detection, attributes['accel'] = [ax, ay, az] (世界坐标)
                                       attributes['gyro'] = [gx, gy, gz] (rad/s)
            dt: 时间步长 (s),None 时用 self.dt

        Returns:
            ego_motion dict:
                - 'delta_position': np.array(3,) 1 帧内位移 (米)
                - 'delta_velocity': np.array(3,) 1 帧内速度变化 (米/秒)
                - 'delta_yaw': float 偏航角变化 (弧度)
        """
        if dt is None:
            dt = self.dt

        if imu_det is None or 'accel' not in (imu_det.attributes or {}):
            # 无 IMU 数据,返回零运动
            return self._zero_motion()

        accel = np.array(imu_det.attributes['accel'], dtype=float)
        gyro = np.array(imu_det.attributes['gyro'], dtype=float) if 'gyro' in imu_det.attributes else np.zeros(3)
        yaw_rate_z = float(gyro[2])  # z 轴角速度 = yaw rate (简化,忽略横滚/俯仰)

        if not self.is_initialized:
            # 第一帧:初始化,无运动增量
            self.prev_accel = accel.copy()
            self.prev_gyro_z = yaw_rate_z
            self.velocity = accel * dt  # 假设从静止开始
            self.is_initialized = True
            return self._zero_motion()

        # 梯形积分
        prev_vel = self.velocity.copy()
        delta_vel = 0.5 * (self.prev_accel + accel) * dt
        self.velocity = self.velocity + delta_vel
        delta_pos = 0.5 * (prev_vel + self.velocity) * dt
        delta_yaw = 0.5 * (self.prev_gyro_z + yaw_rate_z) * dt

        # 更新状态
        self.prev_accel = accel.copy()
        self.prev_gyro_z = yaw_rate_z
        self.prev_yaw += delta_yaw

        ego_motion = {
            'delta_position': delta_pos,
            'delta_velocity': delta_vel,
            'delta_yaw': delta_yaw,
        }
        self.history.append(ego_motion)
        if len(self.history) > 100:
            self.history = self.history[-50:]
        return ego_motion

    def update_ego_state(self, ego: EgoState, dt: float = None) -> dict:
        """
        直接从 EgoState (真值) 推算 ego_motion
        - 用于无 IMU 或测试场景
        - 注意: 这是"完美 ego motion",实际工程中应来自 IMU
        """
        if dt is None:
            dt = self.dt
        delta_pos = ego.velocity * dt
        delta_vel = ego.acceleration * dt
        # 偏航角变化 ≈ yaw_rate * dt (用 angular_velocity[2])
        delta_yaw = float(ego.angular_velocity[2]) * dt
        return {
            'delta_position': delta_pos,
            'delta_velocity': delta_vel,
            'delta_yaw': delta_yaw,
        }

    def _zero_motion(self) -> dict:
        return {
            'delta_position': np.zeros(3),
            'delta_velocity': np.zeros(3),
            'delta_yaw': 0.0,
        }


def extract_imu_from_sensors(sensor_detections: dict) -> Detection:
    """
    从 sensor_detections 字典中提取第一个 IMU Detection

    Args:
        sensor_detections: {sensor_id: [Detection, ...]}

    Returns:
        IMU Detection 或 None (无 IMU 数据时)
    """
    for sid, dets in sensor_detections.items():
        if sid.startswith('imu') and dets:
            return dets[0]
    return None


def compute_ego_motion(sensor_detections: dict,
                       ego: EgoState,
                       predictor: IMUEgoPredictor,
                       dt: float = None) -> dict:
    """
    一站式计算 ego_motion

    Args:
        sensor_detections: 传感器检测字典
        ego: EgoState (真值) — IMU 缺失时 fallback
        predictor: IMUEgoPredictor 实例
        dt: 时间步长

    Returns:
        ego_motion dict
    """
    imu_det = extract_imu_from_sensors(sensor_detections)
    if imu_det is not None:
        return predictor.update(imu_det, dt=dt)
    # fallback: 用 EgoState 真值
    return predictor.update_ego_state(ego, dt=dt)
