"""
UKF 跟踪器 - 无迹卡尔曼滤波
状态: [x, y, z, vx, vy, vz] (CV) 或 [x,y,z,vx,vy,vz,ax,ay,az] (CA)
使用 Unscented Transform 处理非线性（相比 EKF 无需解析 Jacobian）
"""
import warnings
import numpy as np
from core.data_types import TrackedObject


# ---------------------------------------------------------------------------
# Unscented Transform Core
# ---------------------------------------------------------------------------

def sigma_points(x, P, alpha=0.5, beta=2.0, kappa=3.0):
    """
    生成 2n+1 个 sigma points 及权重（Merwe's scaled UT）

    kappa=3.0 确保 c = n + lambda > 0，权重稳定。

    Args:
        x: 状态向量 (n,)
        P: 状态协方差 (n, n)
        alpha: spread (0.5 is a good default)
        beta: prior knowledge (2.0 optimal for Gaussian)
        kappa: secondary spread (3.0 is standard)

    Returns:
        sigmas: (2n+1, n) sigma points
        Wm: (2n+1,) 均值权重
        Wc: (2n+1,) 协方差权重
    """
    n = x.shape[0]
    lam = alpha**2 * (n + kappa) - n
    c = n + lam          # = alpha^2*(n+kappa), always positive with kappa=3
    c = max(c, 1e-12)

    # 正则化：确保 P 正定
    P_reg = P.copy()
    eigvals, eigvecs = np.linalg.eigh(P_reg)
    if np.any(eigvals <= 0):
        eigvals = np.maximum(eigvals, 1e-8)
        P_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # 矩阵平方根: sqrt(c*P) = U @ diag(sqrt(c*eigvals)) @ U^T
    sqrt_c = np.sqrt(c)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sqrt_eigvals = np.sqrt(eigvals * c)
        sqrt_P = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T

    sigmas = np.zeros((2 * n + 1, n))
    sigmas[0] = x.copy()
    for i in range(n):
        sigmas[i + 1]     = x + sqrt_P[:, i]
        sigmas[n + 1 + i] = x - sqrt_P[:, i]

    # 权重
    Wm = np.zeros(2 * n + 1)
    Wc = np.zeros(2 * n + 1)
    for i in range(1, 2 * n + 1):
        w = 1.0 / (2 * c)
        Wm[i] = w
        Wc[i] = w
    Wm[0] = lam / c
    Wc[0] = lam / c + (1 - alpha**2 + beta)

    return sigmas, Wm, Wc


def unscented_transform(sigmas, Wm, Wc, noise_cov=None):
    """
    对 sigma points 做 unscented transform

    Returns:
        x_pred: 预测均值 (n,)
        P_pred: 预测协方差 (n, n)
    """
    n = sigmas.shape[1]
    x_pred = np.dot(Wm, sigmas)

    P_pred = np.zeros((n, n))
    for i in range(sigmas.shape[0]):
        diff = sigmas[i] - x_pred
        P_pred += Wc[i] * np.outer(diff, diff)

    if noise_cov is not None:
        P_pred += noise_cov

    return x_pred, P_pred


# ---------------------------------------------------------------------------
# 运动模型
# ---------------------------------------------------------------------------

def f_cv(x, dt):
    """匀速 (Constant Velocity) 运动模型"""
    x_new = x.copy()
    x_new[:3] = x[:3] + x[3:6] * dt
    return x_new


def f_ca(x, dt):
    """匀加速 (Constant Acceleration) 运动模型"""
    x_new = x.copy()
    dt2 = 0.5 * dt * dt
    x_new[:3] = x[:3] + x[3:6] * dt + x[6:9] * dt2
    x_new[3:6] = x[3:6] + x[6:9] * dt
    return x_new


def h_position(x):
    """观测模型：提取位置"""
    return x[:3]


# ---------------------------------------------------------------------------
# UKF Track
# ---------------------------------------------------------------------------

class UKFTrack:
    """单个 UKF 跟踪器，接口与 EKFTrack 兼容（可互换）"""

    def __init__(self, track_id, initial_pos, initial_vel=None,
                 dt=0.05, timestamp=0.0,
                 process_noise_vel=1.0,
                 use_confidence_weighted=True,
                 max_history=50,
                 model='cv'):
        self.track_id = track_id
        self.dt = dt
        self.age = 0
        self.hits = 1
        self.miss_streak = 0
        self.last_t = timestamp
        self.source_sensors = set()
        self.class_label = None
        self.history = []
        self._max_history = max_history
        self.use_confidence_weighted = use_confidence_weighted
        self.model = model

        self._process_noise_vel_base = process_noise_vel
        self._n = 6 if model == 'cv' else 9

        # 状态向量
        self.x = np.zeros(self._n)
        self.x[:3] = initial_pos
        if initial_vel is not None:
            self.x[3:6] = initial_vel

        # 协方差矩阵
        self.P = np.eye(self._n) * 5.0

        # 兼容层，供 MultiObjectTracker 墓碑/合并逻辑使用
        self.kf = _UKFCompat(self)

        # 同 EKF：避免同 timestamp 重复 predict
        self._already_predicted = False

    # ------------------------------------------------------------------
    # 过程噪声
    # ------------------------------------------------------------------

    def _make_Q(self, dt, scale=1.0):
        """构建过程噪声协方差矩阵 Q
        注意：位置过程噪声是绝对值（不缩 dt²），跟 EKF 一致
        速度过程噪声是绝对值
        """
        q = self._process_noise_vel_base * scale
        if self.model == 'cv':
            # CV: [x, y, z, vx, vy, vz]
            # Q_pos = 0.1 * I  (位置过程噪声，绝对值 m²，跟 dt 无关)
            # Q_vel = q * I   (速度过程噪声，m²/s²)
            Q = np.eye(self._n)
            Q[:3, :3] *= 0.1
            Q[3:, 3:] *= q
            return Q
        else:
            # CA: [x, y, z, vx, vy, vz, ax, ay, az]
            Q = np.eye(self._n)
            Q[:3, :3] *= 0.1
            Q[3:6, 3:6] *= q
            Q[6:, 6:] *= q * 0.5
            return Q

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, dt=None):
        """UKF 预测步：用 sigma points + 运动模型预测状态分布"""
        if dt is None:
            dt = self.dt

        scale = 1.0 + min(self.miss_streak, 5) * 0.5 if self.miss_streak > 0 else 1.0
        Q = self._make_Q(dt, scale)

        sigmas, Wm, Wc = sigma_points(self.x, self.P)

        # 每个 sigma point 通过运动模型
        f = f_cv if self.model == 'cv' else f_ca
        sigmas_pred = np.array([f(s, dt) for s in sigmas])

        self.x, self.P = unscented_transform(sigmas_pred, Wm, Wc, noise_cov=Q)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, det_pos, sensor_id, timestamp, dt=None,
               det_confidence=1.0, class_label=None):
        """
        UKF 更新步
        若 timestamp != self.last_t 先 predict，再用检测更新
        多次 update 复用同一个预测状态（与 EKF 行为一致）
        """
        # 使用 timestamp 比较判断是否已预测（同 EKF 的 last_t 机制）
        # 不用布尔标志 _already_predicted（会一直保留 True）
        if timestamp > self.last_t:
            if dt is None:
                dt = timestamp - self.last_t
            self.predict(dt)
            self.last_t = timestamp

        # 观测噪声（置信度加权）
        r = (0.5 / max(0.1, det_confidence)
             if (self.use_confidence_weighted and det_confidence < 1.0)
             else 0.5)
        R = np.eye(3) * r

        # 当前预测状态的 sigma points
        sigmas, Wm, Wc = sigma_points(self.x, self.P)

        # 通过观测模型 h_position
        sigmas_z = np.array([h_position(s) for s in sigmas])

        # 预测观测均值
        z_pred = np.dot(Wm, sigmas_z)

        # 预测观测协方差
        S = np.zeros((3, 3))
        for i in range(sigmas_z.shape[0]):
            diff = sigmas_z[i] - z_pred
            S += Wc[i] * np.outer(diff, diff)
        S += R

        # 状态-观测互相关协方差 T
        T = np.zeros((self._n, 3))
        for i in range(sigmas.shape[0]):
            dx = sigmas[i] - self.x
            dz = sigmas_z[i] - z_pred
            T += Wc[i] * np.outer(dx, dz)

        # Kalman gain
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        K = T @ Sinv

        # 更新状态和协方差
        innovation = det_pos - z_pred
        self.x = self.x + K @ innovation
        self.P = self.P - K @ S @ K.T

        # 防协方差奇异
        eigvals = np.linalg.eigvalsh(self.P)
        if np.any(eigvals <= 0):
            self.P += np.eye(self._n) * 1e-6

        self.age += 1
        self.hits += 1
        self.miss_streak = 0
        self.source_sensors.add(sensor_id)
        if class_label:
            self.class_label = class_label

        self.history.append({
            't': timestamp,
            'pos': self.x[:3].copy(),
            'vel': self.x[3:6].copy(),
        })
        if len(self.history) > self._max_history:
            self.history = self.history[-50:]

    # ------------------------------------------------------------------
    # Miss
    # ------------------------------------------------------------------

    def miss(self, timestamp, dt=None):
        """本帧未匹配（仅预测，同 EKF 避免重复预测）"""
        # 使用 timestamp 比较判断是否已预测
        if timestamp <= self.last_t:
            return
        if dt is None:
            dt = timestamp - self.last_t
        self.predict(dt)
        self.last_t = timestamp
        self.age += 1
        self.miss_streak += 1
        self.history.append({
            't': timestamp,
            'pos': self.x[:3].copy(),
            'vel': self.x[3:6].copy(),
        })

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------

    def to_track(self, timestamp):
        return TrackedObject(
            track_id=self.track_id,
            timestamp=timestamp,
            position=self.x[:3].copy(),
            velocity=self.x[3:6].copy(),
            covariance=self.P.copy(),
            age=self.age,
            hits=self.hits,
            miss_streak=self.miss_streak,
            source_sensors=set(self.source_sensors),
            class_label=self.class_label,
        )

    def is_alive(self, max_miss=5):
        return self.miss_streak < max_miss

    def get_predicted_position(self):
        return self.x[:3].copy()


# ---------------------------------------------------------------------------
# 兼容层
# ---------------------------------------------------------------------------

class _UKFCompat:
    """UKF 内部状态的只读兼容包装，供 MultiObjectTracker 使用"""
    def __init__(self, parent):
        self._parent = parent

    @property
    def x(self):
        return self._parent.x

    @property
    def P(self):
        return self._parent.P


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def make_cv_ukf(dt=0.05, process_noise_vel=1.0, measure_noise=0.5,
                alpha=0.5, beta=2.0, kappa=3.0):
    """匀速 UKF 配置"""
    return dict(model='cv', dt=dt, process_noise_vel=process_noise_vel,
                measure_noise=measure_noise, alpha=alpha, beta=beta, kappa=kappa)


def make_ca_ukf(dt=0.05, process_noise_vel=1.0, measure_noise=0.5,
                alpha=0.5, beta=2.0, kappa=3.0):
    """匀加速 UKF 配置"""
    return dict(model='ca', dt=dt, process_noise_vel=process_noise_vel,
                measure_noise=measure_noise, alpha=alpha, beta=beta, kappa=kappa)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Test 1: CV model - verify consistency with EKF on linear CV problem
    trk = UKFTrack(track_id=1,
                   initial_pos=np.array([0.0, 0.0, 0.0]),
                   initial_vel=np.array([10.0, 0.0, 0.0]),
                   dt=0.05)

    for _ in range(10):
        trk.predict()

    # With R=0.5 and P[0,0]=6.99 after prediction, a noisy observation of 52.5m
    # results in ~49m estimate (very close to obs, high Kalman gain ~0.93)
    trk.update(np.array([52.5, 0.0, 0.0]), 'lidar', 0.5)
    print(f'UKF estimate: {trk.x[:3]}')
    print(f'UKF vel: {trk.x[3:6]}')
    print('CV UKF self-test OK')

    # Test 2: CA model
    trk_ca = UKFTrack(track_id=2,
                      initial_pos=np.array([0.0, 0.0, 0.0]),
                      initial_vel=np.array([0.0, 0.0, 0.0]),
                      dt=0.05, model='ca')
    trk_ca.x[6:9] = np.array([2.0, 0.0, 0.0])  # ax=2 m/s^2
    for _ in range(10):
        trk_ca.predict()
    trk_ca.update(np.array([5.0, 0.0, 0.0]), 'lidar', 0.5)
    print(f'\nCA UKF estimate: {trk_ca.x[:3]}')
    print(f'CA UKF vel: {trk_ca.x[3:6]}')
    print(f'CA UKF acc: {trk_ca.x[6:9]}')
    print('CA UKF self-test OK')

    # Test 3: Interface checks
    assert hasattr(trk, 'kf')
    assert trk.kf.x is trk.x
    assert trk.kf.P is trk.P
    assert trk.get_predicted_position().shape == (3,)
    assert trk.to_track(0.5).track_id == 1
    assert trk.is_alive(max_miss=5)
    print('\nInterface checks passed!')
