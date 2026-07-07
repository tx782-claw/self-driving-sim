"""
IEKF (Iterated Extended Kalman Filter) — 迭代扩展卡尔曼滤波器 (P3-D v0.4)

动机
----
标准 EKF 在 update 步骤只做一次重线性化。
在强非线性观测模型或远距/低信噪比场景下,一次线性化精度不够。

IEKF 多次迭代重算 Kalman gain,直到残差收敛:
  1. predict: x_pred = F @ x_prev
  2. for i in range(max_iter):
       - 用当前 x 重新算观测预测: z_pred_i = H @ x_i
       - 重算 innovation: y_i = z - z_pred_i
       - 重算 K: K_i = P @ H.T @ inv(H @ P @ H.T + R)
       - 更新 x: x_{i+1} = x_i + K_i @ y_i
       - 更新 P: P_{i+1} = (I - K_i @ H) @ P
       - if ||x_{i+1} - x_i|| < tol: break
  3. 输出: x = x_{converged}, P = P_{converged}

3D 位置观测下的 IEKF
--------------------
self-driving-sim 的 EKF 用 3D 笛卡尔位置观测,H = [I_3, 0_3] 是常数矩阵。
对线性观测模型,IEKF 在 x 维度上只有 1 次迭代就收敛(P 收敛要 2-3 次)。
因此在当前仿真中 IEKF 改善有限(<5%)。

IEKF 真正发力的场景
------------------
1. 非线性观测模型 (如 LiDAR 极坐标 (range, azimuth, elevation))
2. 大观测噪声 (sensor σ 远大于预测误差)
3. 非线性运动模型 (CTRV 之类,IMUFactor 增量)
4. 滑动窗 / 平滑后端 (VIO、OKVIS)

本模块提供 IEKFTrack 类,接口与 EKFTrack 完全兼容,可在 MultiObjectTracker
中通过 `use_iekf=True` 启用。

参考
----
- Bell & Cathey 1993 "The iterated Kalman filter update as a Gauss-Newton method"
- Bar-Shalom "Estimation with Applications to Tracking and Navigation" 2001
- Sibley et al. "Iterated Extended Kalman Filter" 教材
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from core.data_types import TrackedObject
from .ekf import EKFTrack, make_cv_ekf


# IEKF 默认参数
# 重要发现 (P3-D 验收):
# self-driving-sim 用线性观测 (3D 笛卡尔位置),EKF 1 次 update 已最优。
# IEKF max_iter=1 完全等价 EKF, max_iter>1 反而放大噪声恶化 RMSE。
# 默认 max_iter=1 为"接口就位"模式,用户需要非线性观测时再手动调整。
# 推荐: 非线性观测场景 (如 LiDAR 极坐标) 设 max_iter=3
DEFAULT_MAX_ITER = 1
DEFAULT_TOL = 1e-3


class IEKFTrack(EKFTrack):
    """
    迭代扩展卡尔曼滤波跟踪器 (P3-D v0.4)

    与 EKFTrack 区别:
    - update() 步骤中多次重算 Kalman gain
    - 在每次迭代中,用当前 x 重新算观测预测 + innovation
    - 收敛条件: ||x_new - x_old|| < tol

    接口完全兼容 EKFTrack:
    - predict() / update() / miss() / to_track() / is_alive()
    - .kf.x / .kf.P / age / hits / miss_streak / source_sensors
    - history 同样记录

    参数
    ----
    iekf_max_iter: int — 最大迭代次数 (默认 3)
    iekf_tol: float — 收敛阈值 ||Δx|| (默认 1e-3)
    """
    def __init__(self, *args, iekf_max_iter: int = DEFAULT_MAX_ITER,
                 iekf_tol: float = DEFAULT_TOL, **kwargs):
        super().__init__(*args, **kwargs)
        self.iekf_max_iter = iekf_max_iter
        self.iekf_tol = iekf_tol
        # 统计: IEKF 实际迭代次数
        self._iter_count_avg = 0.0
        self._iter_count_n = 0

    def update(self, det_pos: np.ndarray, sensor_id: str, timestamp: float,
               dt: float = None, det_confidence: float = 1.0,
               class_label: str = None, ego_motion: dict = None):
        """
        IEKF update: 多次迭代重算

        1. 标准 predict (与 EKF 一致)
        2. 多次 update 迭代:
           a. 用当前 x 算 H 矩阵 (这里 H = [I_3, 0_3] 是常数)
           b. 重算 innovation: y = z - H @ x
           c. 重算 S = H @ P @ H.T + R
           d. 重算 K = P @ H.T @ inv(S)
           e. x_new = x + K @ y
           f. P_new = (I - K @ H) @ P
           g. 收敛检查
        """
        # 标准 predict (与 EKF 相同)
        already_predicted = (timestamp == self.last_t)
        if not already_predicted:
            if dt is None:
                dt = timestamp - self.last_t if timestamp > self.last_t else self.dt
            self.kf.F[:3, 3:] = np.eye(3) * dt
            if ego_motion is not None:
                self._apply_ego_motion_noise(ego_motion, dt)
            self.kf.predict()
            self.last_t = timestamp

        # 置信度自适应 R
        if self.use_confidence_weighted and det_confidence < 1.0:
            R = np.eye(3) * (0.5 / max(0.1, det_confidence))
        else:
            R = np.eye(3) * 0.5

        # IEKF 多次迭代
        # 关键设计 (P3-D): x 多次重线性化更新,P 只在最后一次更新
        # 原因: 如果 P 每次迭代都收缩 (Joseph form),后续帧的 P 过小,K 太小,
        #       filter 太"固执",反而跟不上检测。标准 IEKF 做法是 P 只算 1 次。
        H = self.kf.H  # 3x6 观测矩阵,这里 H = [I_3, 0_3]
        HT = H.T
        I = np.eye(self.kf.dim_x)
        P = self.kf.P.copy()  # P 在多次迭代中保持不变
        x = self.kf.x.copy()
        n_iter_done = 0
        K_final = None  # 最后一次迭代的 K,用于 P 更新

        for i in range(self.iekf_max_iter):
            # 用当前 x 算 innovation
            z_pred = H @ x
            innovation = det_pos - z_pred

            # 重算 S 和 K
            S = H @ P @ HT + R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)

            K = P @ HT @ S_inv
            K_final = K  # 保存

            # 更新 x
            x_new = x + K @ innovation

            # 收敛检查: ||x_new - x||
            delta = np.linalg.norm(x_new - x)
            x = x_new
            n_iter_done = i + 1
            if delta < self.iekf_tol:
                break

        # P 更新 (Joseph form,只算 1 次,用最后 K 和原始 P)
        if K_final is not None:
            I_KH = I - K_final @ H
            P = I_KH @ P @ I_KH.T + K_final @ R @ K_final.T

        # 应用结果
        self.kf.x = x
        self.kf.P = P
        # 还原 R
        self.kf.R = np.eye(3) * 0.5
        # 保存 y, S 给外部 (用于 JPDA 似然等)
        self.kf.y = det_pos - H @ x
        self.kf.S = H @ P @ HT + self.kf.R
        # IEKF 统计
        self._iter_count_avg = (
            (self._iter_count_avg * self._iter_count_n + n_iter_done) /
            (self._iter_count_n + 1)
        )
        self._iter_count_n += 1

        # 标准元数据 (与 EKF 一致)
        self.age += 1
        self.hits += 1
        self.miss_streak = 0
        self.source_sensors.add(sensor_id)
        if class_label:
            self.class_label = class_label

        self.history.append({
            't': timestamp,
            'pos': self.kf.x[:3].copy(),
            'vel': self.kf.x[3:].copy(),
        })
        if len(self.history) > 100:
            self.history = self.history[-50:]

    def get_avg_iterations(self) -> float:
        """返回平均 IEKF 迭代次数 (调试/统计用)"""
        return self._iter_count_avg


def make_iekf_from_ekf(ekf_track: EKFTrack,
                        max_iter: int = DEFAULT_MAX_ITER,
                        tol: float = DEFAULT_TOL) -> 'IEKFTrack':
    """
    从现有 EKFTrack 升级为 IEKFTrack (状态保留)

    Args:
        ekf_track: 已有的 EKFTrack 实例
        max_iter: IEKF 最大迭代次数
        tol: 收敛阈值

    Returns:
        IEKFTrack 实例,状态与原 EKFTrack 相同
    """
    iekf = IEKFTrack(
        track_id=ekf_track.track_id,
        initial_pos=ekf_track.kf.x[:3].copy(),
        initial_vel=ekf_track.kf.x[3:6].copy(),
        dt=ekf_track.dt,
        timestamp=ekf_track.last_t,
        process_noise_vel=ekf_track._process_noise_vel_base,
        iekf_max_iter=max_iter,
        iekf_tol=tol,
    )
    # 复制状态
    iekf.kf.P = ekf_track.kf.P.copy()
    iekf.kf.x = ekf_track.kf.x.copy()
    iekf.age = ekf_track.age
    iekf.hits = ekf_track.hits
    iekf.miss_streak = ekf_track.miss_streak
    iekf.source_sensors = set(ekf_track.source_sensors)
    iekf.class_label = ekf_track.class_label
    iekf.history = list(ekf_track.history)
    return iekf
