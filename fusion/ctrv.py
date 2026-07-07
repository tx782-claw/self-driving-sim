"""
CTRV 模型 (Constant Turn Rate and Velocity) — UKF 实现 (P3-C 增强 v0.4)

动机
----
CV/CA 假设目标在固定方向运动,弯道场景误差大。
CTRV 引入 yaw_rate (偏航角速度),能跟踪正在转向的目标。
- 直线场景: yaw_rate → 0,自动退化为 CV
- 弯道场景: yaw_rate > 0,真实跟踪曲线轨迹

自驾驶 sim 适配
----------------
self-driving-sim 用 3D 位置 (x, y, z) + 3D 速度 (vx, vy, vz)。
标准 CTRV 是 2D: [x, y, v, yaw, yaw_rate] (5 维)。
本实现扩展为 3D 兼容:
- 状态: [x, y, z, vx, vy, vz, yaw, yaw_rate] (8 维)
- 水平面 (xy): CTRV — 转向率影响 xy 位置
- 垂直方向 (z): CV — 简单匀速

为什么用 UKF
------------
CTRV 运动模型是非线性的 (含 sin/cos 和除以 yaw_rate),EKF 需求 Jacobian (复杂且易错)。
UKF 用 sigma points + unscented transform 自动处理非线性,代码更简洁。

参考
----
- Schubert et al. "Empirical evaluation of vehicular models for ego motion estimation", IEEE IV 2011
- Bar-Shalom "Estimation with Applications to Tracking and Navigation" 2001, Chapter 11
"""
import numpy as np
from filterpy.kalman import KalmanFilter  # 仅用于复用 UKF 框架的接口结构
from .ukf import sigma_points, unscented_transform


# 状态索引
CTRV_X = 0
CTRV_Y = 1
CTRV_Z = 2
CTRV_VX = 3
CTRV_VY = 4
CTRV_VZ = 5
CTRV_YAW = 6
CTRV_YAW_RATE = 7
CTRV_DIM = 8  # 状态维度


def f_ctrv(x: np.ndarray, dt: float) -> np.ndarray:
    """
    CTRV 运动模型 (8 维,3D 扩展)

    Args:
        x: 状态 [x, y, z, vx, vy, vz, yaw, yaw_rate]
        dt: 时间步长

    Returns:
        预测状态

    物理模型 (水平面)
    ----------------
    - v = sqrt(vx^2 + vy^2) — 水平面速度大小
    - yaw_new = yaw + yaw_rate * dt
    - 若 |yaw_rate| > eps:
        x_new = x + (v / yaw_rate) * (sin(yaw_new) - sin(yaw))
        y_new = y + (v / yaw_rate) * (cos(yaw) - cos(yaw_new))
    - 若 |yaw_rate| ≈ 0 (直线):
        x_new = x + vx * dt
        y_new = y + vy * dt

    垂直方向
    --------
    z_new = z + vz * dt
    yaw_rate 保持不变
    """
    x_new = np.zeros(CTRV_DIM)
    px, py, pz = x[CTRV_X], x[CTRV_Y], x[CTRV_Z]
    vx, vy, vz = x[CTRV_VX], x[CTRV_VY], x[CTRV_VZ]
    yaw, yaw_rate = x[CTRV_YAW], x[CTRV_YAW_RATE]

    # 水平面速度大小
    v = np.sqrt(vx ** 2 + vy ** 2)
    # 转向率阈值 (小于此值视为直线,避免除零)
    yaw_rate_eps = 1e-4

    if abs(yaw_rate) > yaw_rate_eps and v > 0.1:
        # 真实 CTRV 几何
        yaw_new = yaw + yaw_rate * dt
        x_new[CTRV_X] = px + (v / yaw_rate) * (np.sin(yaw_new) - np.sin(yaw))
        x_new[CTRV_Y] = py + (v / yaw_rate) * (np.cos(yaw) - np.cos(yaw_new))
    else:
        # 退化 CV (直线或静止)
        x_new[CTRV_X] = px + vx * dt
        x_new[CTRV_Y] = py + vy * dt

    # 垂直方向 CV
    x_new[CTRV_Z] = pz + vz * dt

    # 速度 (在 CTRV 模型中 vx,vy 是 yaw 方向的分量,但我们用绝对 vx,vy)
    x_new[CTRV_VX] = vx
    x_new[CTRV_VY] = vy
    x_new[CTRV_VZ] = vz

    # yaw 变化
    x_new[CTRV_YAW] = yaw + yaw_rate * dt
    # yaw_rate 保持 (匀速转弯假设)
    x_new[CTRV_YAW_RATE] = yaw_rate

    # yaw 归一化到 [-pi, pi]
    x_new[CTRV_YAW] = (x_new[CTRV_YAW] + np.pi) % (2 * np.pi) - np.pi
    return x_new


def h_ctrv_3d(x: np.ndarray) -> np.ndarray:
    """
    观测模型: 3D 位置 (x, y, z)

    CTRV 8 维 → 3 维位置观测
    """
    return np.array([x[CTRV_X], x[CTRV_Y], x[CTRV_Z]])


def make_ctrv_ukf(dt: float = 0.05,
                   process_noise_pos: float = 0.5,
                   process_noise_vel: float = 1.0,
                   process_noise_yaw: float = 0.1,
                   process_noise_yaw_rate: float = 0.5,
                   measure_noise: float = 0.5,
                   alpha: float = 0.5, beta: float = 2.0, kappa: float = 3.0):
    """
    构造 CTRV UKF 状态 (轻量包装,类似 EKFTrack 的 kf 属性)

    Returns:
        dict 包含:
        - x: 状态向量 (8,)
        - P: 协方差 (8x8)
        - Q: 过程噪声 (8x8)
        - R: 观测噪声 (3x3)
        - alpha/beta/kappa: UT 参数
        - dt: 时间步长
    """
    ukf_state = {
        'x': np.zeros(CTRV_DIM),
        'P': np.eye(CTRV_DIM) * 5.0,
        'Q': np.diag([
            process_noise_pos, process_noise_pos, process_noise_pos,  # xyz
            process_noise_vel, process_noise_vel, process_noise_vel,  # vxvyvz
            process_noise_yaw, process_noise_yaw_rate,  # yaw, yaw_rate
        ]),
        'R': np.eye(3) * measure_noise,
        'alpha': alpha,
        'beta': beta,
        'kappa': kappa,
        'dt': dt,
        '_measure_noise': measure_noise,  # 内部用,update 时改 R 后还原
    }
    return ukf_state


def ctrv_predict(ukf_state: dict, dt: float = None) -> dict:
    """
    CTRV UKF 预测步骤

    Args:
        ukf_state: make_ctrv_ukf() 返回的状态字典
        dt: 时间步长

    Returns:
        更新后的 ukf_state (in-place)
    """
    if dt is None:
        dt = ukf_state['dt']
    x = ukf_state['x']
    P = ukf_state['P']
    Q = ukf_state['Q']

    # 1. 生成 sigma points
    sigmas, Wm, Wc = sigma_points(x, P,
                                    alpha=ukf_state['alpha'],
                                    beta=ukf_state['beta'],
                                    kappa=ukf_state['kappa'])

    # 2. 通过运动模型传播
    n = len(x)
    sigmas_pred = np.zeros_like(sigmas)
    for i in range(sigmas.shape[0]):
        sigmas_pred[i] = f_ctrv(sigmas[i], dt)

    # 3. Unscented transform
    x_pred, P_pred = unscented_transform(sigmas_pred, Wm, Wc, noise_cov=Q)

    # yaw 归一化
    x_pred[CTRV_YAW] = (x_pred[CTRV_YAW] + np.pi) % (2 * np.pi) - np.pi

    ukf_state['x'] = x_pred
    ukf_state['P'] = P_pred
    return ukf_state


def ctrv_update(ukf_state: dict, z: np.ndarray) -> dict:
    """
    CTRV UKF 更新步骤 (3D 位置观测)

    Args:
        ukf_state: CTRV UKF 状态
        z: 观测 (3,) 位置 (x, y, z)

    Returns:
        更新后的 ukf_state (in-place)
    """
    x = ukf_state['x']
    P = ukf_state['P']
    R = ukf_state['R']

    # 1. Sigma points
    sigmas, Wm, Wc = sigma_points(x, P,
                                    alpha=ukf_state['alpha'],
                                    beta=ukf_state['beta'],
                                    kappa=ukf_state['kappa'])

    # 2. 传播到观测空间
    z_dim = len(z)
    sigmas_h = np.zeros((sigmas.shape[0], z_dim))
    for i in range(sigmas.shape[0]):
        sigmas_h[i] = h_ctrv_3d(sigmas[i])

    # 3. 观测预测
    z_pred = np.dot(Wm, sigmas_h)
    S = np.zeros((z_dim, z_dim))
    for i in range(sigmas.shape[0]):
        diff = sigmas_h[i] - z_pred
        S += Wc[i] * np.outer(diff, diff)
    S = S + R

    # 4. 状态-观测互协方差
    n = len(x)
    P_xz = np.zeros((n, z_dim))
    for i in range(sigmas.shape[0]):
        diff_x = sigmas[i] - x
        diff_z = sigmas_h[i] - z_pred
        P_xz += Wc[i] * np.outer(diff_x, diff_z)

    # 5. Kalman gain
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    K = P_xz @ S_inv

    # 6. 状态更新
    innovation = z - z_pred
    x_new = x + K @ innovation

    # 7. 协方差更新
    P_new = P - K @ S @ K.T

    # yaw 归一化
    x_new[CTRV_YAW] = (x_new[CTRV_YAW] + np.pi) % (2 * np.pi) - np.pi

    ukf_state['x'] = x_new
    ukf_state['P'] = P_new
    # 保存 S 和 innovation (用于模型概率计算,类似 IMM 的 _gaussian_likelihood)
    ukf_state['_S'] = S
    ukf_state['_innovation'] = innovation
    return ukf_state


def ctrv_gaussian_likelihood(ukf_state: dict) -> float:
    """
    计算 CTRV 模型的新息高斯似然 (用于 IMM 模型概率更新)

    Returns:
        likelihood: N(innov; 0, S) 概率密度
    """
    S = ukf_state.get('_S')
    innov = ukf_state.get('_innovation')
    if S is None or innov is None:
        return 1e-6
    try:
        S_inv = np.linalg.inv(S)
        lik = float(np.exp(-0.5 * innov @ S_inv @ innov) /
                     np.sqrt(max(1e-12, np.linalg.det(2 * np.pi * S))))
        return max(lik, 1e-12)
    except np.linalg.LinAlgError:
        return 1e-6


def get_ctrv_position(ukf_state: dict) -> np.ndarray:
    """获取 CTRV 位置 (3,)"""
    return ukf_state['x'][:3].copy()


def get_ctrv_velocity(ukf_state: dict) -> np.ndarray:
    """获取 CTRV 速度 (3,)"""
    return ukf_state['x'][3:6].copy()


def init_ctrv_state(initial_pos: np.ndarray,
                    initial_vel: np.ndarray = None,
                    initial_yaw: float = 0.0,
                    initial_yaw_rate: float = 0.0,
                    P_scale: float = 5.0) -> dict:
    """
    初始化 CTRV 状态

    Args:
        initial_pos: 初始位置 (3,)
        initial_vel: 初始速度 (3,),None 时设为 0
        initial_yaw: 初始偏航角 (rad),默认 0
        initial_yaw_rate: 初始转向率 (rad/s),默认 0
        P_scale: 协方差初值缩放

    Returns:
        CTRV UKF 状态字典
    """
    state = make_ctrv_ukf(dt=0.05)
    state['x'] = np.array([
        initial_pos[0], initial_pos[1], initial_pos[2],
        initial_vel[0] if initial_vel is not None else 0,
        initial_vel[1] if initial_vel is not None else 0,
        initial_vel[2] if initial_vel is not None else 0,
        initial_yaw,
        initial_yaw_rate,
    ])
    state['P'] = np.eye(CTRV_DIM) * P_scale
    return state
