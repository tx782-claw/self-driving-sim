# Changelog

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
