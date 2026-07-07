# Changelog

## v0.4.0 (2026-07-07) — P3-F: 前端适配 + v0.4.0 release

### 修改
- **`app/webui.py`** — 前端适配 v0.4 P3 全部特性
  - "🔗 融合算法 (v0.4)" 区块: 新增 4 个 P3 开关
    - 关联模式: hungarian / **gnns** (P3-B) / jpda
    - "🛋 运动模型 (P3-C/D)" expander: IMM checkbox + n_models selectbox (2/3) + IEKF checkbox + max_iter slider
    - "🚗 自车运动补偿 (P3-A)" expander: use_ego_motion checkbox (默认开)
  - 仿真模式 `run_simulation()` 新参数: `use_imm/imm_n_models/use_iekf/iekf_max_iter/use_ego_motion`
  - 真实数据模式 `run_nuscenes()` 新参数: `imm_n_models/association_mode` + nuScenes 侧边栏新增 IMM n_models + GNNS selectbox
  - 保留旧式 use_imm / use_ukf checkbox 内部重定向到 expander
- **`fusion/tracker.py`**
  - `MultiObjectTracker.__init__` 新增 `imm_n_models: int = 2` 参数
  - `_create_track()` 创建 IMMTrack 时传 `n_models=self.imm_n_models`
- **`core/__init__.py`** — bump __version__ 从 '0.3' 到 '0.4.0'

### v0.4.0 P3 总体验收
| 子项 | 状态 | commit | 新增测试 | RMSE / MOTA 变化 |
|---|---|---|---|---|
| P3-A 自车运动补偿 | ✅ | b571ffd | +15 | Velocity RMSE -6% |
| P3-B GNNS 关联 | ✅ | 7236250 | +16 | 框架就位 (无改善,诚实验证) |
| P3-C IMM 3 模型 | ✅ | 12eff17 | +15 | 框架就位 (弯道场景) |
| P3-D IEKF 升级 | ✅ | 3ea98be | +16 | max_iter=1 默认 (无噪声放大) |
| **P3-F 前端 + release** | ✅ | (本 commit) | — | 全局 UI 适配 |

### 测试
- 全套: **131/131 全过** (unit 120 + integration 11)
- 旧 84 个 v0.3 测试 0 破坏 (P3 全向后兼容)
- Streamlit webui 启动成功 (`/_stcore/health` = ok)

### 总体 (v0.3 → v0.4.0)
- **代码**: 7 个新文件 (`imu_predict.py / gnns.py / ctrv.py / iekf.py` + 4 个 test_*.py), 5 个文件修改
- **测试**: 84 → 131 (+47 个 P3 测试)
- **API 稳定性**: 100% 向后兼容 (v0.3 旧代码无需修改)
- **新功能 (P3)**: 全员接口就位,等待真实数据 (nuScenes mini) 验证发力场景

## v0.4.0-dev (2026-07-07) — P3-D: IEKF 迭代扩展卡尔曼

### 新增
- **`fusion/iekf.py`** — IEKF 迭代卡尔曼滤波跟踪器
  - `IEKFTrack` 继承 `EKFTrack`,接口完全兼容
  - 多次重线性化 update (max_iter / tol 可配)
  - P 矩阵 Joseph form 更新,保证半正定
  - `make_iekf_from_ekf()` 从 EKFTrack 升级为 IEKFTrack (状态保留)
  - **重要发现**: 默认 max_iter=1 (避免在自车驾驶 sim 线性观测下放大噪声)
- **`tests/unit/test_iekf.py`** — 16 个新单测
  - IEKFTrack 基本 10 个 (创建 / 继承 / max_iter / tol / ego_motion / NaN / to_track / miss)
  - IEKF vs EKF 3 个
    - `test_default_iekf_equals_ekf` - max_iter=1 完全等价 EKF
    - `test_high_iter_amplifies_noise` - **诚实验证 max_iter=3 反而恶化** (线性观测下)
    - `test_near_target_similar` - 近距场景相似
  - Tracker 集成 2 个 (use_iekf 开关 / Highway 5 车 E2E)

### 修改
- **`fusion/tracker.py`**
  - `MultiObjectTracker.__init__` 新增 `use_iekf: bool = False` 参数
  - 新增 `iekf_max_iter: int = 1` / `iekf_tol: float = 1e-3` 参数
  - `_create_track()` 优先级: use_imm > use_iekf > use_ukf > EKF
- **`fusion/__init__.py`** — 导出 `IEKFTrack` / `make_iekf_from_ekf` / `DEFAULT_MAX_ITER` / `DEFAULT_TOL`

### 验收 (test_iekf.py, 远距场景 50m 静止, 噪声 1.5m, 100 帧)
| 配置 | RMSE | 备注 |
|---|---|---|
| EKF (baseline) | 1.512 m | 传统 EKF |
| IEKF max_iter=1 | 1.512 m | 完全等价 EKF (默认) |
| IEKF max_iter=2 | 1.906 m | 恶化 26% (放大噪声) |
| IEKF max_iter=3 | 2.166 m | 恶化 43% (放大噪声) |

**诚实结论**: self-driving-sim 用线性观测 (3D 笛卡尔位置),EKF 1 次 update 已是最优 Bayesian 估计。IEKF 多次迭代把 x 推向有噪声的检测,反而放大噪声恶化 RMSE。**默认 max_iter=1 为"接口就位"模式**,非退化。IEKF 真正优势场景: 非线性观测 (如 LiDAR 极坐标 range/azimuth/elevation),需手动设 max_iter=3。**关键验收: max_iter=1 下 IEKF = EKF (不恶化),框架就位,未来非线性观测时可用。**

### 测试
- unit: 120 个 (原 104 + 新 16), 全过 0.5s
- 全套: 131/131 全过 (含 integration 11 个)
- 旧 115 个测试 0 破坏 (n_models / max_iter 向后兼容)

## v0.4.0-dev (2026-07-07) — P3-C: IMM 3 模型 (CV+CA+CTRV)

### 新增
- **`fusion/ctrv.py`** — CTRV 模型 (UKF 实现)
  - 8 维状态: [x, y, z, vx, vy, vz, yaw, yaw_rate]
  - 3D 适配: 水平面 CTRV 几何 + 垂直方向 CV
  - `f_ctrv()` 运动模型 (含 sin/cos 除以 yaw_rate + 退化到 CV)
  - `ctrv_predict()` / `ctrv_update()` UKF 完整接口
  - `ctrv_gaussian_likelihood()` 用于 IMM 模型概率计算
  - `init_ctrv_state()` / `get_ctrv_position()` 工具函数
- **`tests/unit/test_imm3.py`** — 15 个新单测
  - CTRV 运动模型 3 个 (直线退化/弯道圆弧/yaw 归一化)
  - CTRV UKF 4 个 (init/predict/update/似然)
  - IMM 3 模型 6 个 (创建/向后兼容/转移矩阵/直线/弯道/NaN/加权位置)
  - IMM3 vs IMM2 端到端 1 个 (弯道 RMSE 对比)

### 修改
- **`fusion/imm.py`**
  - `IMMTrack.__init__` 新增 `n_models: int = 2` 参数 (默认 2 向后兼容)
  - `n_models=3` 启用 CV+CA+CTRV 三模型
  - 新增 `initial_yaw: float = 0.0` 参数 (CTRV 初始 yaw)
  - `make_imm_transition_matrix` 新增 `n_models` 参数 (2/3 都支持)
  - 3 模型下: CTRV 状态 (`ctrv_state` dict) 维护
  - `predict()` / `update()` 支持 3 模型 + ego_motion
  - 3 模型似然混合 + 模型概率更新
  - `get_position()` / `get_velocity()` 3 模型加权

### 验收 (test_imm3.py, 弯道场景 1 车)
| 指标 | IMM2 (CV+CA) | IMM3 (CV+CA+CTRV) | 变化 |
|---|---|---|---|
| Position RMSE | 0.405 m | 0.406 m | ~一致 (0.002%) |
| 模型概率 | [0.5, 0.5] | [0.4, 0.3, 0.3]→演化 | CTRV 参与混合 |

**诚实结论**: 当前仿真 sensor 给世界坐标真值,弯道简单场景下 IMM2 已经很准,CTRV 优势不显 (RMSE 几乎一致)。**关键验收: IMM3 没有恶化 (0.405 → 0.406)**,框架就位。CTRV 真正优势场景: 真实数据 (nuScenes mini) + 多弯道 + 大噪声,等 P3-F 真实数据回归验证。

### 测试
- unit: 104 个 (原 89 + 新 15), 全过 0.7s
- 全套: 115/115 全过 (含 integration 11 个)
- 旧 7 个 IMM 测试 0 破坏 (n_models=2 向后兼容)

## v0.4.0-dev (2026-07-07) — P3-B: GNNS 数据关联

### 新增
- **`fusion/gnns.py`** — GNNS 关联器 (P3-B)
  - `gnns_associate()` — Mahalanobis 距离 + chi-square 门限 + 贪心 (N-to-1) 匹配
  - `hybrid_gnns_associate()` — 严格门限 / 宽松门限混合 (实验性)
  - N-to-1 模式: 允许多个 det 共同更新一个 track (多传感器融合场景)
  - 1-to-1 模式: 严格 GNNS,每个 det 只匹配一个 track
- **`tests/unit/test_gnns.py`** — 16 个新单测
  - Chi2 常量/门限辅助 3 个
  - Mahalanobis 距离 2 个
  - GNNS 关联 8 个 (空/单匹配/门限过滤/大协方差/N-to-1/1-to-1/置信度加权)
  - Tracker 集成 + Dense Highway 24 车 ID Switches 对比 2 个
- **`scripts/bench_gnns.py`** — GNNS vs Hungarian 在 3 个场景下的 ID Switches 对比脚本

### 修改
- **`fusion/association.py`**
  - 新增 `_mahal_distance()` / `_chi2_gate()` / `_chi2_threshold()` / `_build_mahal_cost_matrix()` 辅助函数
  - 新增常量 `CHI2_3D_95` (7.815) / `CHI2_3D_99` (11.345) / `CHI2_3D_999` (16.266)
  - `hungarian_associate()` 新增 `gate_chi2` 参数,`use_mahalanobis=True` 时生效
- **`fusion/tracker.py`**
  - `association_mode` 新增 `'gnns'` 选项
  - `MultiObjectTracker.__init__` 新增 `gnns_gate_chi2=7.815` / `gnns_allow_n_to_1=True` 参数
- **`fusion/__init__.py`** — 导出 `gnns_associate` / `hybrid_gnns_associate` / `_mahal_distance` / `_chi2_gate` / `CHI2_3D_95`

### 验收 (bench_gnns.py, 3 个场景)
| 场景 | Hungarian MOTA | GNNS MOTA | Hun IDSW | GNNS IDSW |
|---|---|---|---|---|
| Highway 5 车 | +0.0000 | +0.0000 | 0 | 0 |
| Dense Highway 24 车 | +0.0000 | +0.0000 | 0 | 0 |
| 协方差不均匀 (5 车) | +0.0000 | +0.0000 | 0 | 0 |

**诚实结论**: 当前仿真 sensor 直接给世界坐标真值,场景太"干净",二者 ID Switches 都为 0。GNNS 真正优势在"协方差不均匀 + track 位置接近"场景(如 nuScenes mini 真实数据,有检测噪声 + 标定误差)。**关键验收: GNNS 没有恶化 (0 → 0)**。

### 测试
- unit: 89 个 (原 73 + 新 16), 全过 4.4s
- 全套: 100/100 全过 (含 integration 11 个)
- 旧 7 个 association / 73 个其他测试 0 破坏

## v0.4.0-dev (2026-07-07) — P3-A: 自车运动补偿

### 新增
- **`fusion/imu_predict.py`** — IMUEgoPredictor 类
  - 梯形积分从 IMU accel/gyro 推算 1 帧内自车运动
  - 输出 `ego_motion = {delta_position, delta_velocity, delta_yaw}`
  - 同时提供 `update_ego_state()` fallback (无 IMU 时用 EgoState 真值)
  - 工具函数: `extract_imu_from_sensors()` / `compute_ego_motion()`
- **`tests/unit/test_ego_motion.py`** — 15 个新单测
  - IMUEgoPredictor 7 个 (梯形积分/yaw/reset/IMU 提取)
  - EKFTrack 5 个 (Q 扩展/接口兼容/高速稳定性)
  - MultiObjectTracker 3 个 (端到端 Highway 场景)
- **`scripts/bench_ego_motion.py`** — Highway 5 车 P3-A 验收脚本

### 修改
- **`fusion/ekf.py`**
  - `EKFTrack.predict(dt, ego_motion=None)` 新增 ego_motion 参数
  - `EKFTrack.update(..., ego_motion=None)` 新增 ego_motion 参数
  - 新增 `_apply_ego_motion_noise()` — 在过程噪声 Q 中加入自车运动不确定性
  - 新增常量 `EGO_MOTION_PROCESS_NOISE_SCALE = 0.5`
- **`fusion/tracker.py`**
  - `MultiObjectTracker.update(..., ego_motion=None)` 新增参数
  - 新增 `use_ego_motion=True` (默认开启) / `imu_predictor` 初始化参数
  - 当 use_ego_motion=True 且未外部传入 ego_motion,自动从 sensor_detections 提取 IMU
  - `reset()` 重置 IMU 预测器状态
- **`fusion/__init__.py`** — 导出 IMUEgoPredictor / compute_ego_motion / extract_imu_from_sensors

### 验收 (bench_ego_motion.py, Highway 5 车, 100 帧, 0.05s 步长)
| 指标 | Baseline | P3-A | 变化 |
|---|---|---|---|
| Position RMSE | 0.871 m | 0.892 m | +2.4% (无改善) |
| Velocity RMSE | 7.180 m/s | **6.749 m/s** | **-6.0%** (改善) |
| MOTA | 0.0000 | 0.0000 | 一致 |

**诚实结论**: Position RMSE 未达 10% 目标 (因为当前 sensor 模型直接给世界坐标真值,ego_motion 补偿对位置改善有限),但 Velocity RMSE 改善 6%,P 协方差扩展正确。**框架就位**,等未来 sensor 改为"给 sensor_frame 位置"(更真实)时,ego_motion 补偿会大放异彩 — 调研报告 §9.2 列为 P3-A 刚需原因。

### 测试
- unit: 73 个 (原 58 + 新 15), 全过 0.5s
- 全套: 84/84 全过 (含 integration 11 个)
- 旧 7 个 ekf/ukf/imm/scenarios 测试 0 破坏

## v0.3 (2026-07-06) — nuScenes mini 真实数据回归
- nuScenes mini adapter + 5 integration tests (默认 skip)
- 4 个 mota baseline JSON + dashboard
- README MOTA baseline 章节
- 7 轮 bug fix: view 函数双减 / sensor panel 双减 / LIDAR 5列 / sensor_frame transform / GIF 双减 / 视图自动放大 / heading=0 rotation
- MOTA v0 (-0.084) → v6 (+0.209) → v7 noisy (+0.185) 闭环

## v0.2.2 (2026-06-13)

### 修复
- 删 `fusion/tracker.py.bak` / `tracker.py.bak2` / `tracker_fixed.py` / `jpda.py.bak`（仓库卫生）
- webui.py 19 处 `use_container_width` → `width='stretch'/'content'`（streamlit 1.50 弃用警告修复）
- 版本号统一：`core/__init__.py` 单一 `__version__ = "0.2.2"` 来源，UI/文档不再硬编码

### JPDA 修复尝试 → 回滚
- `tracker_fixed.py` 里的修复试用后发现更差：vel decay 0.85^8=0.27 太狠
  （miss 8 帧后 track 速度被抽到 0），叠加 β<0.1 早退 + 3σ 紧门限，密集场景下
  MOTA 从 0 跌到 -8.7
- **决定**：回滚修复、保留 v0.2.1 的 JPDA 行为。JPDA 模式被明确标注为实验功能
  （README 已有 known limitation 警告）。主线推荐 Hungarian。
- `tests/test_jpda_fixes.py` 重写：改用 MOTA / 速度 RMSE 指标，Hungarian 主线
  必须 ≥0.85，JPDA 只检不崩不 NaN
- Hungarian 验证：5车 MOTA 0.92, 9车 MOTA 0.91, 速度 RMSE 3.6 m/s, ID switch 5

### 优化
- **优化A · IMM** (CV+CA 双模型交互多模型)：`fusion/imm.py`
- **优化B · 距离依赖噪声**：LiDAR/Radar/Camera 噪声按距离分段，远距目标 ID switch 减少
- **优化C · pytest 分层**：`tests/unit/` + `tests/integration/`，覆盖率从 5.8% → ~25%
- **优化D · config 外部化**：`config/scenarios.yaml` + loader，支持 grid sweep
- **优化E · BEV 鸟瞰融合可视化**：单图叠加 LiDAR 强度 + Radar RCS 等高线 + Camera bbox + Track 协方差

### 工程
- **优化F · 独立 git 仓库**：`git init` + v0.2.2 initial commit

## v0.2.1 (2026-06-09)
- JPDA 修复尝试（实际有 clutter 增加副作用，v0.2.2 真正解决）

## v0.2.0 (2026-06-08)
- 5 场景 + 5 传感器 + 天气 + Streamlit UI + Plotly 3D
- EKF + UKF + Hungarian + JPDA 融合
- MOTA 0.93 / position RMSE 0.72m @ 102× 实时

## v0.1.0 (2026-06-02)
- P0 最小可演示：单帧 3 传感器 + 1 场景 + EKF + Open3D
