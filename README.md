# 🚗 Self-Driving Sensor Simulation & Fusion Platform

> 本地运行的多传感器仿真 + 融合平台  
> **版本**: v0.2.2 · **日期**: 2026-06-13

---

## ✨ 特性

- 🎯 **多传感器仿真**: LiDAR (32线机械) / Radar (FMCW) / Camera (针孔) / IMU / GPS
- 🧠 **多目标跟踪**: EKF (CV/CA) · **IMM (CV+CA 交互多模型)** · UKF (无迹) + 贪心 N-to-1 数据关联
- 📡 **距离依赖传感器模型** (v0.2.2): LiDAR/Radar/Camera 噪声 + 漏检率随距离变化
- 🛣️ **丰富场景**: Highway / Urban / Dense Highway (24车) / Junction / Stop & Go
- 🌐 **可视化**: Plotly (web) · Open3D (本地) · **BEV 鸟瞰融合**（单图叠加 3 传感器）
- 🖥️ **Web UI**: Streamlit 一键启动，时间滑块回放，指标时序曲线
- 📊 **评估指标**: RMSE / Precision / Recall / ID 切换 / 时延
- 🎬 **导出**: 全场 JSON · track/GT CSV · 动画 GIF
- 🧪 **pytest 分层**: 64 个测试 (unit + integration) 全过

---

## 🚀 快速开始

### 安装

```bash
cd /Users/mac/.openclaw/workspace/self-driving-sim
pip install -r requirements.txt
```

### 启动 Web UI

```bash
streamlit run app/webui.py
```

浏览器打开 http://localhost:8501

### 命令行端到端测试

```bash
PYTHONPATH=. python tests/integration/test_e2e.py
```

### pytest 分层测试 (64 个)

```bash
# 跑全部 (unit + integration)
python3 -m pytest tests/

# 只跑 unit (快，快到0.5s)
python3 -m pytest tests/unit/

# 只跑 integration (慢，~2s)
python3 -m pytest tests/integration/
```

测试覆盖：
- `tests/unit/` — EKF/UKF/IMM/UKF 滤波 · Hungarian/JPDA 关联 · LiDAR/Radar/Camera 传感器 · RangeNoiseModel · BEV 可视化 · YAML loader
- `tests/integration/` — e2e 端到端 · JPDA 回归

---

## 📁 项目结构

```
self-driving-sim/
├── ARCHITECTURE.md       # 架构设计文档（P0/P1/P2/P3 优先级）
├── README.md
├── requirements.txt
├── core/                 # 仿真核心
│   ├── clock.py          # 时间步进器
│   ├── simulator.py      # 主仿真循环
│   └── data_types.py    # Detection / TrackedObject / SimFrame 等数据类
├── sensors/             # 传感器仿真
│   ├── base.py
│   ├── lidar.py         # 32线机械 LiDAR，射线投射
│   ├── radar.py         # FMCW Radar，距离+多普勒+角度
│   ├── camera.py        # 针孔相机，2D 目标检测框
│   └── imu_gps.py       # IMU (加速度/角速度) + GPS
├── scenarios/            # 场景生成
│   ├── base.py
│   ├── highway.py       # 高速：自车 + N 辆周围车辆
│   ├── urban.py         # 城市：自车 + 行人 + 交叉路口
│   ├── dense.py         # 密集路况
│   │   ├── DenseHighwayScenario  # 3车道×8车 = 24辆密集车队
│   │   ├── JunctionScenario       # 4方向汇入路口
│   │   └── StopAndGoScenario       # 走走停停拥堵
│   └── weather.py       # 天气预设（晴/雨/雾/夜）
├── fusion/              # 融合算法
│   ├── association.py   # 贪心 N-to-1 匈牙利数据关联
│   ├── ekf.py           # EKF（CV 匀速 / CA 匀加速）+ Track 生命周期
│   └── tracker.py       # MultiObjectTracker（候选→确认，墓碑复活机制）
├── visualization/       # 渲染
│   ├── plotly_viewer.py # 3D + 俯视 (Plotly)
│   └── panels/          # 各传感器独立面板
├── evaluation/          # 评估指标
│   └── metrics.py       # RMSE / MOTA / MOTP / per-frame 时序
├── app/
│   └── webui.py         # Streamlit Web UI
├── scripts/
│   └── export_session.py # 全场 JSON / track CSV / GT CSV 导出
└── tests/
    └── test_e2e.py     # 端到端测试
```

---

## 🗺️ 场景

| 场景 | 说明 |
|------|------|
| **Highway** | 高速：自车 + N 辆周围车辆 |
| **Urban** | 城市：自车 + 行人 + 交叉路口 |
| **Dense Highway** | 车队跟自车：3车道 × 8车 = 24辆密集车队，前车遮挡严重，考验关联鲁棒性 |
| **Junction** | 4方向汇入路口 |
| **Stop & Go** | 走走停停拥堵，自车加减速波 |

---

## 🧠 融合架构

### 当前实现（v0.2.2）

| 模块 | 实现 | 状态 |
|------|------|------|
| **关联（默认）** | 贪心 N-to-1 匈牙利 | ✅ 稳定（密集场景推荐） |
| **关联（实验）** | JPDA 联合概率数据关联 | ⚠️ 有限制：密集场景下 EKF 速度估计可能漂移 |
| **滤波（默认）** | EKF（CV + CA 双模型） | ✅ 稳定 |
| **滤波（实验）** | UKF 无迹卡尔曼 | ⚠️ Beta 阶段 |
| **跟踪器** | MultiObjectTracker | 候选→确认状态机 + 墓碑复活机制 |
| **IMM** | ❌ 未实现 | 规划 P3 |

### 关联模式选择

| 场景 | 推荐 | 说明 |
|------|------|------|
| 5 个以下目标 | Hungarian 或 JPDA | 两者表现接近 |
| 密集场景（5+ 个近距离目标） | **Hungarian** | JPDA 在密集场景下 track 速度估计会漂移 |
| 理论验证 / 教学 | JPDA | 能看到 β 概率分布 |

### 融合流程

```
每帧传感器检测 → Detection 列表
                        ↓
           关联算法（Hungarian / JPDA）
                        ↓
    ┌───────────────────┴────────────────────┐
    │            已匹配 track                  │
    │  EKF.predict() → 多检测 update()       │
    │  (同 timestamp 只 predict一次)          │
    └─────────────────────────────────────────┘
                        ↓
    ┌───────────────────┴────────────────────┐
    │          未匹配 Detection                │
    │  ① 先尝试从墓碑复活（时间+距离门限）    │
    │  ② 无则创建新 candidate track           │
    │  ③ candidate hits ≥ min_hits → 确认     │
    └─────────────────────────────────────────┘
                        ↓
              TrackedObject 列表
```

---

## 🛠️ 技术栈

| 类别 | 库 |
|------|-----|
| 数值计算 | NumPy / SciPy |
| 卡尔曼滤波 | FilterPy |
| 图像处理 | OpenCV / Pillow |
| 3D 渲染 | Open3D / Plotly |
| Web UI | Streamlit |
| 数据 | Pandas / JSON |

---

## 📊 已实现

- ✅ 5 个传感器仿真（LiDAR / Radar / Camera / IMU / GPS）
- ✅ EKF 跟踪（CV + CA 双模型）
- ✅ 贪心 N-to-1 匈牙利数据关联
- ✅ MultiObjectTracker（状态机 + 墓碑复活）
- ✅ 5 个场景（高速 / 城市 / 密集车队 / 路口 / Stop&Go）
- ✅ 天气预设（晴 / 雨 / 雾 / 夜）
- ✅ 3D + 俯视双视图（Plotly）
- ✅ 传感器独立面板（LiDAR BEV/3D · Radar 距离-多普勒 · Camera · IMU · GPS）
- ✅ 6 项顶部指标卡片
- ✅ 时间滑块回放（多媒体控制台 UI）
- ✅ 指标时序曲线（RMSE · 目标数量 · Precision/Recall）
- ✅ 导出（全场 JSON · track/GT CSV · GIF 动画）
- ✅ 端到端测试

---

## 🗺️ 开发路线图

详见 `ARCHITECTURE.md` 中的 P0/P1/P2/P3 优先级

| 阶段 | 内容 | 状态 |
|------|------|------|
| 🟢 **P0** | 最小可演示，跑通全链路 | ✅ |
| 🟡 **P1** | 完整 5 传感器 + 真实场景 + 天气 + 传感器面板 | ✅ |
| 🟠 **P2** | **UKF** ✅ · **JPDA** ✅ (有限制) · **IMM** ✅ · 距离依赖噪声 · 密集场景优化 | ✅ v0.2.2 完成 |
| 🔴 **P3** | 深度学习融合 · BEV · 占用网络 · ROS 集成 | ⏳ |

### P2 已完成（v0.2.2）

- ✅ `fusion/ukf.py` — UKF 跟踪器（CV + CA），sigma points + unscented transform
- ✅ `fusion/jpda.py` — JPDA 关联（标准 marginal 公式 + NN-限制 + 速度稳定）
- ✅ `fusion/imm.py` — **IMM (CV+CA 交互多模型)**，高沠 5车 MOTA 0.895→0.942
- ✅ `fusion/tracker.py` — 支持 `association_mode='jpda'|'hungarian'` 和 `use_ukf=True|False` `use_imm=True|False`
- ✅ `sensors/range_model.py` — 距离依赖噪声 + 漏检率（LiDAR/Radar/Camera 共享）
- ⚠️ **JPDA 限制**: 密集场景下 EKF 速度估计仍会漂移 (实测 5车 MOTA 负数)。
  2026-06-13 试过“3σ 紧门限 + β<0.1 miss + vel decay”修复，实测 MOTA 从
  0 跌到 -8.7（修复反而退化），已回滚。**生产环境请用 Hungarian**。
  JPDA 仅保留为 ≤4 个目标的理论验证/教学用。

---

## 📝 License

Personal project, aslan.