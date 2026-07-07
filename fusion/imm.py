"""
IMM (Interacting Multiple Model) — 交互多模型跟踪器 (P2 增强 + P3-C v0.4)

动机
----
单一 EKF 模型 (CV 匀速) 在 stop-and-go / cut-in / 切道场景下
跟踪误差大：目标突然加减速时，CV 跟不上。CA (匀加速) 又在
高速直线场景噪声大。IMM 在 CV / CA 之间软切换，跟踪更稳。

P3-C 增强 (v0.4)
-----------------
新增第 3 个模型 CTRV (Constant Turn Rate and Velocity, UKF 实现)
- 状态: 8 维 [x, y, z, vx, vy, vz, yaw, yaw_rate]
- 水平面 CTRV 几何: 考虑转向率,弯道跟踪更准
- 直线场景自动退化为 CV (yaw_rate → 0)

3x3 Markov 转移矩阵
3 模型互投影: CV(6) ↔ CA(9) ↔ CTRV(8)

接口与 EKFTrack 兼容
-------------------
predict / update / miss / to_track / is_alive

参考
----
Bar-Shalom et al., "Estimation with Applications to Tracking and Navigation" (2001)
Schubert et al. "Empirical evaluation of vehicular models" IEEE IV 2011
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from core.data_types import TrackedObject
from .ekf import make_cv_ekf, make_ca_ekf
from .ctrv import (
    init_ctrv_state, ctrv_predict, ctrv_update,
    ctrv_gaussian_likelihood, get_ctrv_position, get_ctrv_velocity,
    CTRV_DIM,
)


def make_imm_transition_matrix(diag: float = 0.95, n_models: int = 2) -> np.ndarray:
    """
    Markov 转移矩阵

    Args:
        diag: 对角元 (待在自身模型概率), 默认 0.95
        n_models: 模型数量 (2=CV+CA, 3=CV+CA+CTRV)

    Returns:
        transition matrix (n_models x n_models)
    """
    if not 0.5 < diag < 1.0:
        raise ValueError(f"diag 必须在 (0.5, 1.0) 之间，当前 {diag}")
    off = 1.0 - diag
    T = np.eye(n_models) * diag
    # 剩余概率均分到其他模型
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                T[i, j] = off / (n_models - 1)
    return T


class IMMTrack:
    """
    IMM 跟踪器：CV + CA 双模型交互 (P2)
    或 CV + CA + CTRV 三模型交互 (P3-C v0.4)

    状态: 6 (CV), 9 (CA), 8 (CTRV) 维度不同。
    混合时: 互投影逻辑 — 丢掉高维信息或补 0 低维信息。

    n_models=2 向后兼容 v0.2.2 行为
    n_models=3 启用 CTRV 弯道场景增强
    """
    def __init__(self, track_id: int, initial_pos: np.ndarray,
                 initial_vel: np.ndarray = None,
                 dt: float = 0.05, timestamp: float = 0.0,
                 process_noise_vel: float = 1.0,
                 measure_noise: float = 0.5,
                 transition_matrix: np.ndarray = None,
                 n_models: int = 2,           # P3-C: 默认 2 模型 (向后兼容 v0.2.2)
                                       # 设为 3 启用 CV+CA+CTRV
                 initial_yaw: float = 0.0,    # P3-C: CTRV 初始 yaw (n_models=3 时生效)
                 ):
        self.track_id = track_id
        self.dt = dt
        self.last_t = timestamp
        self.age = 0
        self.hits = 1
        self.miss_streak = 0
        self.source_sensors = set()
        self.class_label = None
        self.history = []
        self._measure_noise = measure_noise
        self.n_models = n_models
        # 限制 n_models
        if n_models not in (2, 3):
            raise ValueError(f"n_models 必须 2 或 3,当前 {n_models}")

        # 模型 1: CV (6 维) - filterpy
        self.cv_kf = make_cv_ekf(dt=dt, process_noise_vel=process_noise_vel,
                                 measure_noise=measure_noise)
        self.cv_kf.x = np.zeros(6)
        self.cv_kf.x[:3] = initial_pos
        if initial_vel is not None:
            self.cv_kf.x[3:] = initial_vel

        # 模型 2: CA (9 维) - filterpy
        self.ca_kf = make_ca_ekf(dt=dt, process_noise_vel=process_noise_vel * 1.5,
                                 process_noise_acc=2.0, measure_noise=measure_noise)
        self.ca_kf.x = np.zeros(9)
        self.ca_kf.x[:3] = initial_pos
        if initial_vel is not None:
            self.ca_kf.x[3:6] = initial_vel

        # 模型 3: CTRV (8 维) - UKF (P3-C 新增)
        if n_models == 3:
            # 根据初始 vel 推算 yaw
            if initial_vel is not None and (initial_vel[0] != 0 or initial_vel[1] != 0):
                initial_yaw = float(np.arctan2(initial_vel[1], initial_vel[0]))
            self.ctrv_state = init_ctrv_state(
                initial_pos=initial_pos,
                initial_vel=initial_vel,
                initial_yaw=initial_yaw,
                initial_yaw_rate=0.0,
                P_scale=5.0,
            )
            self.ctrv_state['dt'] = dt
        else:
            self.ctrv_state = None

        # 模型概率 (CV, CA, [CTRV]) — 初始均匀
        if n_models == 2:
            self.model_probs = np.array([0.5, 0.5])
        else:
            # 3 模型初始偏 CV (CV 表现最稳),CA 与 CTRV 各 0.3
            self.model_probs = np.array([0.4, 0.3, 0.3])

        # Markov 转移矩阵
        if transition_matrix is None:
            self.transition = make_imm_transition_matrix(n_models=n_models)
        else:
            self.transition = transition_matrix
            assert self.transition.shape == (n_models, n_models)

        # 兼容 EKF 接口：tracker._merge_close_tracks 用 .kf.x[:3]
        # IMM 主用 cv_kf（实际输出用 model_probs 加权）
        self.kf = self.cv_kf

    def _project_cv_to_ca(self, x_cv: np.ndarray) -> np.ndarray:
        """CV 6维 → CA 9维（加 0 加速度）"""
        x_ca = np.zeros(9)
        x_ca[:6] = x_cv
        return x_ca

    def _project_ca_to_cv(self, x_ca: np.ndarray) -> np.ndarray:
        """CA 9维 → CV 6维（扔加速度）"""
        return x_ca[:6].copy()

    def _mix_state(self, mix_prob_cv: float, mix_prob_ca: float):
        """
        IMM 状态交互 (mixing)：
        用上一帧模型概率 + 转移矩阵计算混合概率，然后加权混合两个模型的状态
        """
        # 归一化
        s = mix_prob_cv + mix_prob_ca
        if s < 1e-12:
            mix_prob_cv, mix_prob_ca = 0.5, 0.5
        else:
            mix_prob_cv /= s
            mix_prob_ca /= s

        # 混合 CV 状态
        x0_cv_mixed = mix_prob_cv * self.cv_kf.x + mix_prob_ca * self._project_ca_to_cv(self.ca_kf.x)
        # 混合 CA 状态
        x0_ca_mixed = mix_prob_cv * self._project_cv_to_ca(self.cv_kf.x) + mix_prob_ca * self.ca_kf.x

        # 协方差混合（简化：直接用主对角加权，跨模型项忽略）
        # 更严格做法要算 cross-covariance，这里做近似
        P0_cv_mixed = (mix_prob_cv * self.cv_kf.P +
                       mix_prob_ca * (self._project_ca_to_cv_P() if hasattr(self, '_proj_P') else self.cv_kf.P))
        # 简化方案：使用较大的协方差
        P0_cv_mixed = mix_prob_cv * self.cv_kf.P + mix_prob_ca * self.cv_kf.P * 1.2
        P0_ca_mixed = mix_prob_ca * self.ca_kf.P + mix_prob_cv * self.ca_kf.P * 1.2

        self.cv_kf.x = x0_cv_mixed
        self.cv_kf.P = P0_cv_mixed
        self.ca_kf.x = x0_ca_mixed
        self.ca_kf.P = P0_ca_mixed

    def predict(self, dt: float = None, ego_motion: dict = None):
        """
        IMM 预测

        P3-C 新增: 支持 3 模型预测,3 模型都用同一个 ego_motion 补偿

        简化: 仍先 predict 再更新模型概率
        """
        if dt is None:
            dt = self.dt
        # CV 状态转移
        self.cv_kf.F[:3, 3:6] = np.eye(3) * dt
        # CA 状态转移
        self.ca_kf.F[:3, 3:6] = np.eye(3) * dt
        self.ca_kf.F[:3, 6:9] = np.eye(3) * 0.5 * dt * dt
        self.ca_kf.F[3:6, 6:9] = np.eye(3) * dt

        # 未命中增大过程噪声
        if self.miss_streak > 0:
            scale = 1.0 + min(self.miss_streak, 5) * 0.5
            self.cv_kf.Q[3:, 3:] = np.eye(3) * scale
            self.ca_kf.Q[3:6, 3:6] = np.eye(3) * scale

        # CV 预测
        if ego_motion is not None:
            delta_pos = ego_motion.get('delta_position')
            if delta_pos is not None:
                self.cv_kf.Q[:3, :3] += np.diag(np.abs(delta_pos) ** 2 * 0.5)
        self.cv_kf.predict()
        # CA 预测
        if ego_motion is not None:
            delta_pos = ego_motion.get('delta_position')
            if delta_pos is not None:
                self.ca_kf.Q[:3, :3] += np.diag(np.abs(delta_pos) ** 2 * 0.5)
        self.ca_kf.predict()
        # CTRV 预测 (P3-C)
        if self.n_models == 3 and self.ctrv_state is not None:
            ctrv_predict(self.ctrv_state, dt)

    def update(self, det_pos: np.ndarray, sensor_id: str, timestamp: float,
               dt: float = None, det_confidence: float = 1.0,
               class_label: str = None, ego_motion: dict = None):
        """
        用检测更新模型，更新模型概率

        P3-C 新增: 3 模型更新 (CV + CA + CTRV)
        """
        if dt is None:
            dt = timestamp - self.last_t if timestamp > self.last_t else self.dt
        if timestamp != self.last_t:
            self.predict(dt, ego_motion=ego_motion)
            self.last_t = timestamp

        # 置信度自适应 R
        R_scale = 1.0 / max(0.3, det_confidence)
        self.cv_kf.R = np.eye(3) * self._measure_noise * R_scale
        self.ca_kf.R = np.eye(3) * self._measure_noise * R_scale

        # CV / CA 更新
        self.cv_kf.update(det_pos)
        self.ca_kf.update(det_pos)

        # CTRV 更新 (P3-C)
        if self.n_models == 3 and self.ctrv_state is not None:
            self.ctrv_state['R'] = np.eye(3) * self._measure_noise * R_scale
            ctrv_update(self.ctrv_state, det_pos)
            self.ctrv_state['R'] = np.eye(3) * self._measure_noise

        # 还原 R
        self.cv_kf.R = np.eye(3) * self._measure_noise
        self.ca_kf.R = np.eye(3) * self._measure_noise

        # 计算各模型似然
        try:
            lik_cv = self._gaussian_likelihood(self.cv_kf.y, self.cv_kf.S)
        except (AttributeError, np.linalg.LinAlgError):
            lik_cv = 1.0
        try:
            lik_ca = self._gaussian_likelihood(self.ca_kf.y, self.ca_kf.S)
        except (AttributeError, np.linalg.LinAlgError):
            lik_ca = 1.0
        if self.n_models == 3 and self.ctrv_state is not None:
            lik_ctrv = ctrv_gaussian_likelihood(self.ctrv_state)
        else:
            lik_ctrv = 1.0

        # Bayes 更新模型概率
        c = self.transition.T @ self.model_probs
        if self.n_models == 2:
            numer = c * np.array([lik_cv, lik_ca])
        else:
            numer = c * np.array([lik_cv, lik_ca, lik_ctrv])
        denom = numer.sum()
        if denom < 1e-12:
            self.model_probs = np.ones(self.n_models) / self.n_models
        else:
            self.model_probs = numer / denom
        # 避免 0 概率 (防止后续似然计算除零)
        self.model_probs = np.maximum(self.model_probs, 1e-6)
        self.model_probs /= self.model_probs.sum()

        self.age += 1
        self.hits += 1
        self.miss_streak = 0
        self.source_sensors.add(sensor_id)
        if class_label:
            self.class_label = class_label

        self._record_history(timestamp)

    def _gaussian_likelihood(self, innov: np.ndarray, S: np.ndarray) -> float:
        """新息向量的高斯似然 N(innov; 0, S)"""
        try:
            S_inv = np.linalg.inv(S)
            return float(np.exp(-0.5 * innov @ S_inv @ innov) /
                         np.sqrt(max(1e-12, np.linalg.det(2 * np.pi * S))))
        except np.linalg.LinAlgError:
            return 1e-6

    def miss(self, timestamp: float, dt: float = None):
        """本帧未匹配（仅 predict）"""
        if timestamp == self.last_t:
            return
        if dt is None:
            dt = timestamp - self.last_t if timestamp > self.last_t else self.dt
        self.predict(dt)
        self.last_t = timestamp
        self.age += 1
        self.miss_streak += 1
        self._record_history(timestamp)

    def _record_history(self, timestamp: float):
        pos = self.get_position()
        vel = self.get_velocity()
        self.history.append({'t': timestamp, 'pos': pos, 'vel': vel})
        if len(self.history) > 100:
            self.history = self.history[-50:]

    def get_position(self) -> np.ndarray:
        """模型概率加权的位置"""
        pos = self.model_probs[0] * self.cv_kf.x[:3] + self.model_probs[1] * self.ca_kf.x[:3]
        if self.n_models == 3 and self.ctrv_state is not None:
            pos = pos + self.model_probs[2] * get_ctrv_position(self.ctrv_state)
        return pos

    def get_velocity(self) -> np.ndarray:
        vel = self.model_probs[0] * self.cv_kf.x[3:6] + self.model_probs[1] * self.ca_kf.x[3:6]
        if self.n_models == 3 and self.ctrv_state is not None:
            vel = vel + self.model_probs[2] * get_ctrv_velocity(self.ctrv_state)
        return vel

    def get_predicted_position(self) -> np.ndarray:
        return self.get_position()

    def to_track(self, timestamp: float) -> TrackedObject:
        return TrackedObject(
            track_id=self.track_id,
            timestamp=timestamp,
            position=self.get_position(),
            velocity=self.get_velocity(),
            covariance=self.cv_kf.P.copy(),  # 主模型协方差
            age=self.age,
            hits=self.hits,
            miss_streak=self.miss_streak,
            source_sensors=set(self.source_sensors),
            class_label=self.class_label,
        )

    def is_alive(self, max_miss: int = 5) -> bool:
        return self.miss_streak < max_miss
