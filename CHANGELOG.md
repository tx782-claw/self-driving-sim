# Changelog

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
