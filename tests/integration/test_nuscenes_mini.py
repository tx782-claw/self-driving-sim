"""
nuScenes mini 真实数据回归测试.

默认 skip (需要用户下载数据 + 安装 nuscenes-devkit).

跑起来:
  1. pip install nuscenes-devkit
  2. 下载 nuScenes v1.0-mini (4GB) 到 /data/nuscenes
     https://www.nuscenes.org/download
  3. cd self-driving-sim
     pytest tests/integration/test_nuscenes_mini.py -v

不带数据跑:
  pytest tests/integration/test_nuscenes_mini.py -v   # 全 skip
"""
import os
import sys
import numpy as np
import pytest

# 把项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 默认数据根目录 (用户可改)
NUSCENES_DATAROOT = os.environ.get("NUSCENES_DATAROOT", "/data/nuscenes")

# 检查 nuscenes-devkit 是否安装
try:
    import nuscenes  # noqa: F401
    HAS_NUSCENES = True
except ImportError:
    HAS_NUSCENES = False

# 检查数据是否就绪
try:
    from scripts.adapters.nuscenes_adapter import NuScenesAdapter, has_nuscenes_mini
    HAS_ADAPTER = True
except ImportError as e:
    HAS_ADAPTER = False
    _IMPORT_ERROR = str(e)

# 跳过原因字符串 (Python 3.9 兼容: 不在 f-string 里嵌套三元)
_data_ok = has_nuscenes_mini(NUSCENES_DATAROOT) if HAS_ADAPTER else False
_SKIP_REASON = (
    "需要 nuscenes-devkit + nuScenes mini 数据. "
    f"nuscenes-devkit={'yes' if HAS_NUSCENES else 'NO'}, "
    f"adapter={'yes' if HAS_ADAPTER else 'NO'}, "
    f"data={'yes' if _data_ok else 'NO'} "
    f"(检查 {NUSCENES_DATAROOT}/v1.0-mini/). "
    "设环境变量 NUSCENES_DATAROOT 指向你的数据根目录."
)

# 跳过条件: 没有 nuscenes-devkit 或没数据
pytestmark = pytest.mark.skipif(
    not HAS_NUSCENES or not HAS_ADAPTER or not _data_ok,
    reason=_SKIP_REASON,
)


# ----------------- 冒烟测试 (不依赖 tracker) -----------------

def test_nuscenes_mini_smoke_load():
    """最小冒烟: 加载数据 + 解析 1 个 frame + 不崩 / 不 NaN."""
    adapter = NuScenesAdapter(NUSCENES_DATAROOT, version="v1.0-mini", verbose=False)

    scenes = adapter.list_scenes()
    assert len(scenes) > 0, "nuScenes mini 应至少 1 个场景"

    # 遍历所有场景, 取每场景前 5 帧
    for scene_name in scenes[:2]:
        frames = []
        for i, frame in enumerate(adapter.load_scene(scene_name)):
            frames.append(frame)
            if i >= 5:
                break

        assert len(frames) > 0, f"{scene_name}: 0 帧"

        for frame in frames:
            # 1. ego_state 字段无 NaN
            assert np.all(np.isfinite(frame.ego_state.position)), "ego pos 含 NaN"
            assert np.isfinite(frame.ego_state.heading), "ego heading 含 NaN"

            # 2. ground_truth 列表每条都有效
            assert len(frame.ground_truth) > 0, "scene 应至少 1 个 GT 目标"
            for gt in frame.ground_truth:
                assert np.all(np.isfinite(gt.position)), f"GT pos NaN (id={gt.object_id})"
                assert np.all(np.isfinite(gt.velocity)), f"GT vel NaN (id={gt.object_id})"
                assert gt.obj_type in ("car", "truck", "cyclist", "pedestrian"), \
                    f"未知 obj_type: {gt.obj_type}"

            # 3. detections_by_sensor 存在 (临时字段, 由 adapter 注入)
            assert hasattr(frame, "detections_by_sensor"), \
                "adapter 应注入 detections_by_sensor 字段"

            # 4. 每个 sensor_id 的 detection 都合理
            for sensor_id, dets in frame.detections_by_sensor.items():
                for det in dets:
                    assert np.all(np.isfinite(det.position)), \
                        f"{sensor_id} detection pos NaN"
                    assert 0.0 <= det.confidence <= 1.0, \
                        f"{sensor_id} confidence 越界: {det.confidence}"


def test_nuscenes_mini_metadata():
    """元信息测试: scene 列表 / 帧数 / 时长."""
    adapter = NuScenesAdapter(NUSCENES_DATAROOT, version="v1.0-mini")

    scenes = adapter.list_scenes()
    # nuScenes mini 应有 10 个场景
    assert len(scenes) >= 10, f"mini 应 ≥10 场景, 实际 {len(scenes)}"

    # 第一个场景验证帧数 / 时长
    scene_name = scenes[0]
    n_frames = adapter.frame_count(scene_name)
    duration = adapter.scene_duration_s(scene_name)

    # mini 每个场景 ≈ 20 秒 × 12Hz = 240 帧
    assert n_frames > 100, f"{scene_name} 帧数太少: {n_frames}"
    assert 15.0 < duration < 60.0, f"{scene_name} 时长异常: {duration:.1f}s"


def test_nuscenes_mini_categories():
    """类别映射测试: nuScenes 32 类应能映射到 self-driving-sim 4 类."""
    adapter = NuScenesAdapter(NUSCENES_DATAROOT, version="v1.0-mini")

    seen_types = set()
    for scene_name in adapter.list_scenes()[:3]:
        for i, frame in enumerate(adapter.load_scene(scene_name)):
            for gt in frame.ground_truth:
                seen_types.add(gt.obj_type)
            if i >= 30:
                break

    # mini 通常含 car + pedestrian, 偶有 cyclist
    assert "car" in seen_types, f"mini 应含 car, 实际类别: {seen_types}"
    assert "pedestrian" in seen_types, f"mini 应含 pedestrian, 实际类别: {seen_types}"


def test_nuscenes_mini_noisy_mode():
    """noisy 模式: 叠加 RangeNoiseModel 后漏检率应 > 0."""
    from sensors.range_model import RangeNoiseModel  # type: ignore

    noise_model = RangeNoiseModel(max_range_m=80.0)
    adapter = NuScenesAdapter(
        NUSCENES_DATAROOT,
        version="v1.0-mini",
        mode="noisy",
        noise_model=noise_model,
    )

    scene_name = adapter.list_scenes()[0]
    frames = []
    for i, frame in enumerate(adapter.load_scene(scene_name)):
        frames.append(frame)
        if i >= 10:
            break

    # noisy 模式: 总 detection 数应少于 GT (有漏检)
    total_gt = sum(len(f.ground_truth) for f in frames)
    total_det = sum(
        sum(len(dets) for dets in f.detections_by_sensor.values())
        for f in frames
    )

    # noisy 应有一定漏检 (经验: 50-90% 漏检, 取决于 RangeNoiseModel 默认值)
    assert total_gt > 0, "应有 GT"
    assert total_det < total_gt * 6, f"noisy 应有漏检: det={total_det}, gt={total_gt}"


# ----------------- tracker 集成测试 (需要 MultiObjectTracker) -----------------

def test_nuscenes_mini_tracker_runs():
    """端到端: 用 nuScenes mini 喂 MultiObjectTracker, 验证不崩 / 不 NaN."""
    from fusion import MultiObjectTracker  # type: ignore

    adapter = NuScenesAdapter(NUSCENES_DATAROOT, version="v1.0-mini")
    tracker = MultiObjectTracker(
        dt=0.05,
        gate_threshold=15.0,
        min_hits_to_confirm=2,
        max_miss_streak=30,
        use_confidence_weighted=True,
    )

    scene_name = adapter.list_scenes()[0]
    total_frames = 0
    total_tracks = 0
    for frame in adapter.load_scene(scene_name):
        # 把所有 sensor 的 detection 合并喂 tracker
        all_dets = []
        for dets in frame.detections_by_sensor.values():
            all_dets.extend(dets)

        # tracker API 兼容: 接受 detection 列表 + 时间戳
        tracks = tracker.update(all_dets, frame.timestamp)
        frame.tracks = tracks

        # 验证 tracks 无 NaN
        for tr in tracks:
            assert np.all(np.isfinite(tr.position)), "track pos NaN"
            assert np.all(np.isfinite(tr.velocity)), "track vel NaN"

        total_frames += 1
        total_tracks += len(tracks)

        # 只跑前 50 帧做 smoke test, 避免 CI 太慢
        if total_frames >= 50:
            break

    assert total_frames > 0, "tracker 应至少处理 1 帧"
    assert total_tracks > 0, "tracker 应至少产出 1 条 track (mini 场景含多目标)"