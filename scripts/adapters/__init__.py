"""External dataset adapters for self-driving-sim.

把公开自动驾驶数据集 (nuScenes / KITTI / Waymo) 翻译成 self-driving-sim 的
SimFrame 数据结构, 用于真实数据回归测试, 验证 tracker / fusion 在真实
sensor noise + 非匀速运动下的表现.

设计原则:
- 不下载数据, 不重训练模型; 只做 schema 翻译 + 坐标系转换.
- GT 来自数据集标注 (ground truth), detection 来自标注 + 可选噪声叠加.
- ego frame 坐标系: x=前, y=左, z=上 (与 self-driving-sim 现有约定一致).
"""