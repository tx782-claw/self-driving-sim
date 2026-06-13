# 🚗 Self-Driving Sensor Simulation & Fusion Platform

**版本**: v0.2.2
**日期**: 2026-06-13
**范围**: Option B (标准版)

---

## 1. 项目目标

构建一个**本地可运行**的自动驾驶感知层仿真平台，专注于：

- ✅ 多种车载传感器数据仿真（Camera / LiDAR / Radar / IMU / GPS）
- ✅ 多传感器数据融合（EKF + 目标跟踪 + 数据关联）
- ✅ 3D 可视化（实时点云 + 跟踪结果）
- ✅ Web UI 控制 + 回放
- ✅ 评估指标（位置/速度 RMSE / 跟踪精度 / 时延）

**非目标**（不在范围内）：
- 端到端驾驶决策/规划/控制
- 真实数据集训练
- ROS 集成（v2 再考虑）

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit Web UI  (app/webui.py)                           │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │ 场景选择  │ 参数调节  │ 实时3D  │  融合结果 │ 评估指标 │   │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘   │
└─────────────┬───────────────────────────────────────────────┘
              │ HTTP / WebSocket (实时流)
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Simulation Core  (core/simulator.py)                        │
│  ┌──────────────────────────────────────────────────┐        │
│  │ Clock / Stepper (固定步长 50ms / 20Hz)          │        │
│  └──────────────────────────────────────────────────┘        │
│       │           │            │            │                │
│       ▼           ▼            ▼            ▼                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│  │ Camera  │ │  LiDAR  │ │  Radar  │ │IMU+GPS  │            │
│  │(2D图像) │ │(3D点云) │ │(距离+多 │ │(轨迹+   │            │
│  │         │ │         │ │普勒+角) │ │  朝向)  │            │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘            │
│       │           │            │            │                │
│       └───────────┴────────────┴────────────┘                │
│                          ▼                                    │
│              ┌──────────────────────┐                        │
│              │  Fusion Pipeline     │                        │
│              │  ┌────────────────┐  │                        │
│              │  │ Preprocess     │  │                        │
│              │  │ (去噪/标定/对齐)│  │                        │
│              │  └────────┬───────┘  │                        │
│              │           ▼          │                        │
│              │  ┌────────────────┐  │                        │
│              │  │ Association    │  │  Hungarian / GNN      │
│              │  │ (数据关联)     │  │                        │
│              │  └────────┬───────┘  │                        │
│              │           ▼          │                        │
│              │  ┌────────────────┐  │                        │
│              │  │ EKF/UKF Track  │  │  卡尔曼滤波 + 生命周期│
│              │  │ (跟踪)         │  │                        │
│              │  └────────┬───────┘  │                        │
│              │           ▼          │                        │
│              │  ┌────────────────┐  │                        │
│              │  │ State Estimate │  │  位置/速度/朝向/ID    │
│              │  └────────────────┘  │                        │
│              └──────────────────────┘                        │
│                          ▼                                    │
│              ┌──────────────────────┐                        │
│              │  Evaluator           │  RMSE / 跟踪精度      │
│              └──────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 核心数据流

### 3.1 数据结构（统一格式）

所有传感器输出统一为 `Detection` 列表，每帧更新：

```python
@dataclass
class Detection:
    """统一检测结果（融合前）"""
    sensor_id: str          # 传感器ID: "lidar_top", "camera_front", ...
    timestamp: float        # 仿真时间 (s)
    position: np.ndarray    # (x, y, z) 世界坐标
    velocity: np.ndarray    # (vx, vy, vz) 速度
    object_id: int | None   # 真值ID（仿真用，融合时不用）
    confidence: float       # 0~1
    attributes: dict        # 传感器特定属性
        # radar:  {"rcs": ..., "range": ..., "doppler": ...}
        # lidar:  {"num_points": ..., "intensity_avg": ...}
        # camera: {"bbox_2d": (x1,y1,x2,y2), "class": "car", "depth_est": ...}
```

### 3.2 融合层输出

```python
@dataclass
class TrackedObject:
    """融合后的目标轨迹"""
    track_id: int           # 全局唯一跟踪ID
    timestamp: float
    position: np.ndarray    # EKF 估计位置
    velocity: np.ndarray    # EKF 估计速度
    covariance: np.ndarray  # 6x6 协方差矩阵
    age: int                # 跟踪持续帧数
    hits: int               # 关联命中次数
    miss_streak: int        # 连续未命中次数
    source_sensors: set     # 来源传感器集合
    class_label: str | None # 目标类别
```

### 3.3 仿真帧同步

```python
@dataclass
class SimFrame:
    timestamp: float
    ego_state: EgoState     # 自车位置/速度/朝向
    ground_truth: list[GroundTruthObj]  # 真实目标状态
    detections: dict[str, list[Detection]]  # 各传感器检测
    tracks: list[TrackedObject]            # 融合后跟踪
```

---

## 4. 模块设计

### 4.1 `sensors/` 传感器仿真

| 模块 | 物理模型 | 输出 | 复用 |
|------|---------|------|------|
| `camera.py` | 针孔相机 + 简单渲染 | 2D 图像 + 2D 目标检测框 | 全新 |
| `lidar.py` | 旋转扫描 + 射线投射 | 3D 点云 + 目标级检测 | 全新 |
| `radar.py` | 复用你现有的 `radar_simulation.py` FMCW 模型 | 距离/多普勒/角度 + 目标 | **复用 90%** |
| `imu_gps.py` | 运动学积分 + 噪声模型 | 加速度/角速度 + GPS 位置 | 全新 |

**关键设计**：
- 每个传感器独立采样率（Camera 10Hz, LiDAR 10Hz, Radar 20Hz, IMU 100Hz, GPS 1Hz）
- 时间戳统一，融合层做时间对齐
- 传感器外参标定（位置/朝向相对自车）

### 4.2 `scenarios/` 场景生成

简单起步（v0.1）：

```python
# scenarios/highway.py
class HighwayScenario:
    """高速公路场景：自车 + 4-6 个周围车辆"""
    def __init__(self, num_vehicles=6, weather="clear", duration=20.0):
        ...
    def step(self, t: float) -> list[GroundTruthObj]:
        """返回当前时刻所有真实目标状态"""
        ...
```

可扩展接口（v0.2+）：
- `UrbanScenario` - 城市道路 + 行人 + 红绿灯
- `WeatherVariation` - 雨/雾/夜间
- `AdversarialScenario` - 危险场景（cut-in / brake）

### 4.3 `fusion/` 融合算法

**v0.1 基础（先跑通）：**
- `preprocessing.py` - 坐标系转换 + 噪声过滤
- `association.py` - 匈牙利匹配（基于距离）
- `ekf.py` - 扩展卡尔曼滤波（匀速运动模型）
- `tracker.py` - 多目标跟踪器（生命周期管理）

**v0.2 增强：**
- `ukf.py` - 无迹卡尔曼滤波
- `jpda.py` - 联合概率数据关联
- `imm.py` - 交互多模型（匀速/匀加速切换）

### 4.4 `visualization/` 3D 可视化

**v0.1 用 Open3D**（最简单）：
- 自车模型（坐标系三色轴）
- LiDAR 点云（实时变色：地面/目标）
- 跟踪目标（3D 包围盒 + ID 标签）
- 摄像头视角（2D 俯视图 + 点云叠加）

**v0.2 可选 PyVista**（更炫）：
- 体积渲染
- 时间滑块

### 4.5 `app/webui.py` Web UI（Streamlit）

```python
# 主要页面布局
1. 侧边栏：
   - 场景选择（Highway / Urban）
   - 参数（车辆数 / 天气 / 时长）
   - 启动/暂停/重置按钮
2. 主区域：
   - 3D 可视化（嵌入 Open3D HTML）
   - 2D 俯视图（matplotlib）
   - 实时跟踪表（pandas dataframe）
3. 底部：
   - 评估指标（RMSE / 跟踪精度 / 时延）
   - 导出数据按钮（JSON / CSV）
```

### 4.6 `evaluation/` 评估

```python
def compute_metrics(frames: list[SimFrame]) -> dict:
    return {
        "position_rmse_m": ...,          # 位置均方根误差
        "velocity_rmse_mps": ...,        # 速度均方根误差
        "tracking_precision": ...,       # 跟踪准确率（命中/总数）
        "tracking_recall": ...,          # 跟踪召回率
        "false_track_rate": ...,         # 误跟踪率
        "id_switches": ...,              # ID 切换次数
        "avg_latency_ms": ...,           # 单帧处理时延
    }
```

---

## 5. 技术栈

```
核心仿真    : Python 3.11+ 
数值计算    : NumPy / SciPy
卡尔曼滤波  : FilterPy (v0.1 引入)
图像处理    : OpenCV (Camera 渲染) / Pillow
3D 可视化   : Open3D (v0.1) → PyVista (v0.2)
Web UI      : Streamlit
数据格式    : JSON (前后端传输) / Pickle (本地缓存)
测试        : Pytest
依赖管理    : requirements.txt
```

---

## 6. 工作量评估

| 模块 | 预计代码量 | 预计耗时 | 难度 |
|------|-----------|---------|------|
| 架构 + 数据类 | 200 行 | 0.5 天 | ★ |
| Radar（复用） | 200 行（精简） | 0.5 天 | ★ |
| LiDAR | 300 行 | 1 天 | ★★ |
| Camera | 250 行 | 1 天 | ★★ |
| IMU+GPS | 150 行 | 0.5 天 | ★ |
| 场景引擎 | 300 行 | 1 天 | ★★ |
| 融合层 (EKF + 关联) | 400 行 | 1.5 天 | ★★★ |
| 3D 可视化 | 300 行 | 1 天 | ★★ |
| Streamlit Web UI | 400 行 | 1.5 天 | ★★ |
| 评估指标 | 150 行 | 0.5 天 | ★★ |
| 端到端集成 + 测试 | 300 行 | 1 天 | ★★★ |
| 文档 + README | - | 0.5 天 | ★ |
| **合计** | **~2950 行** | **~10.5 人天** | - |

> 实际如果聚焦 MVP（先跑通单传感器 + 基础融合 + 3D + UI），4-5 天可以出第一版。

---

## 7. 开发优先级（P0 → P3）

### 🟢 P0 - 最小可演示（M2 任务，预计 2-3 小时）

**目标**：单帧全链路跑通，浏览器能看到 3D 跟踪结果

- [x] 项目结构 + 数据类
- [ ] Camera（最简单，1 个 2D 图像 + 1 个目标框）
- [ ] LiDAR（简化：射线投射生成点云 + 目标级聚类）
- [ ] Radar（简化：复用你的 50 行核心代码）
- [ ] 场景：3 辆车直线行驶
- [ ] 融合：EKF + 匈牙利
- [ ] 可视化：Open3D 静态 3D 渲染
- [ ] Web UI：Streamlit 单页面

**Demo 验收**：浏览器打开 `localhost:8501`，点"运行"看到 3 辆车在 3D 视图中被识别+跟踪

### 🟡 P1 - 真实仿真（M2 任务，1-2 天）

- [ ] 多帧时间步进（20Hz）
- [ ] 真实场景：高速路 + 城市道路
- [ ] 完整 LiDAR（多线 + 旋转）
- [ ] 完整 Radar（FMCW + CA-CFAR）
- [ ] 完整 Camera（针孔 + 简单渲染）
- [ ] IMU + GPS 通道
- [ ] 3D 实时渲染（Open3D 动画）
- [ ] 评估指标

### 🟠 P2 - 增强（M3 任务，2-3 天）

- [ ] UKF 滤波
- [ ] JPDA 数据关联
- [ ] 天气变化（雨/雾/夜）
- [ ] 多目标密集场景
- [ ] 数据回放与导出
- [ ] 性能优化（向量化）

### 🔴 P3 - 进阶（v2 再说）

- [ ] 深度学习融合
- [ ] BEV 生成
- [ ] 占用网络
- [ ] ROS 集成

---

## 8. 当前会话计划

我先完成 P0（最小可演示），分 3 个 checkpoints：

**Checkpoint 1（现在）**: 架构设计 + 项目骨架 + 数据类
**Checkpoint 2（接下来）**: Camera + LiDAR + Radar 简化版 + 场景 + EKF 融合 + Open3D 3D 渲染 → 本地脚本能跑出 3D 画面
**Checkpoint 3（最后）**: Streamlit Web UI 包装 + 端到端测试

每完成一个 checkpoint 我会汇报进展并展示效果。

---

## 9. 文件结构

```
self-driving-sim/
├── ARCHITECTURE.md        ← 本文档
├── README.md
├── requirements.txt
├── config/
│   ├── sensors.yaml       # 传感器参数
│   └── scenarios.yaml     # 场景配置
├── core/
│   ├── __init__.py
│   ├── simulator.py       # 主仿真循环
│   ├── data_types.py      # Detection / Track / SimFrame 等
│   └── clock.py           # 时间步进
├── sensors/
│   ├── __init__.py
│   ├── base.py            # 传感器基类
│   ├── camera.py
│   ├── lidar.py
│   ├── radar.py           # 复用 + 简化
│   └── imu_gps.py
├── scenarios/
│   ├── __init__.py
│   ├── base.py
│   ├── highway.py
│   └── urban.py
├── fusion/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── association.py     # 匈牙利
│   ├── ekf.py             # 卡尔曼
│   └── tracker.py         # 多目标跟踪
├── visualization/
│   ├── __init__.py
│   ├── open3d_viewer.py   # 3D
│   └── plot_2d.py         # 俯视图
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
├── app/
│   ├── webui.py           # Streamlit 入口
│   └── components/        # UI 组件
├── scenarios_data/        # 仿真输出
└── tests/
    ├── test_sensors.py
    ├── test_fusion.py
    └── test_e2e.py
```

---

**下一步：开始 Checkpoint 2 - 实现核心传感器 + 融合 + 3D 渲染**
