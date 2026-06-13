"""
关联算法单测 — Hungarian + JPDA 边界
"""
import numpy as np
import pytest
from core.data_types import Detection
from fusion.association import hungarian_associate
from fusion.jpda import auto_jpda
from fusion.ekf import EKFTrack


def make_det(x, y, z=0, sensor='lidar', conf=0.9):
    return Detection(
        sensor_id=sensor, timestamp=0.0,
        position=np.array([x, y, z]),
        velocity=np.zeros(3),
        object_id=None,
        confidence=conf,
        attributes={},
    )


def make_track(x, y, tid):
    """用 EKFTrack 包装一个位置（Hungarian/JPDA 都需要 .kf.x[:3]）"""
    t = EKFTrack(tid, np.array([x, y, 0]), np.zeros(3), dt=0.05, timestamp=0.0)
    return t


class TestHungarian:
    """贪心 Hungarian 关联"""

    def test_perfect_match(self):
        """一对一完美匹配 (greedy_multi=False 严格 1-to-1)"""
        dets = [make_det(0, 0), make_det(10, 0), make_det(20, 0)]
        tracks = [make_track(0, 0, 1), make_track(10, 0, 2), make_track(20, 0, 3)]
        matched, unmatched_d, unmatched_t = hungarian_associate(
            dets, tracks, gate_threshold=2.0, greedy_multi=False)
        assert len(matched) == 3
        assert len(unmatched_d) == 0
        assert len(unmatched_t) == 0

    def test_greedy_multi_all_match_same(self):
        """默认 greedy_multi=True：所有 det 可以匹配同一 track"""
        dets = [make_det(0, 0), make_det(1, 0), make_det(2, 0)]
        tracks = [make_track(0, 0, 1)]
        matched, unmatched_d, unmatched_t = hungarian_associate(
            dets, tracks, gate_threshold=5.0)
        # 3 个 det 都匹配 track 0
        assert len(matched) == 3
        assert len(unmatched_d) == 0
        assert len(unmatched_t) == 1

    def test_all_missed(self):
        """全部 track 都 miss（dets 都不在门限内）"""
        dets = [make_det(100, 100)]
        tracks = [make_track(0, 0, 1)]
        matched, unmatched_d, unmatched_t = hungarian_associate(
            dets, tracks, gate_threshold=2.0)
        assert len(matched) == 0
        assert len(unmatched_d) == 1
        assert len(unmatched_t) == 1

    def test_gate_threshold(self):
        """门限过滤"""
        dets = [make_det(0, 0), make_det(5, 0)]  # 第二个超出 2m 门限
        tracks = [make_track(0, 0, 1)]
        matched, unmatched_d, unmatched_t = hungarian_associate(
            dets, tracks, gate_threshold=2.0)
        # 应该只匹配 1 个，1 个 unmatched det
        assert len(matched) == 1
        assert len(unmatched_d) == 1

    def test_empty_inputs(self):
        """空输入不应该崩"""
        matched, unmatched_d, unmatched_t = hungarian_associate([], [], gate_threshold=10.0)
        assert len(matched) == 0
        assert len(unmatched_d) == 0
        assert len(unmatched_t) == 0

    def test_one_to_one_no_greedy(self):
        """greedy_multi=False：1-to-1 严格"""
        dets = [make_det(0, 0), make_det(1, 0)]
        tracks = [make_track(0, 0, 1), make_track(1, 0, 2)]
        matched, _, _ = hungarian_associate(
            dets, tracks, gate_threshold=5.0, greedy_multi=False)
        assert len(matched) == 2


class TestJPDA:
    """JPDA 关联"""

    def test_runs_without_crash(self):
        """基本调用不崩"""
        dets = [make_det(0, 0), make_det(10, 0), make_det(20, 0)]
        tracks = [make_track(0, 0, 1), make_track(10, 0, 2), make_track(20, 0, 3)]
        result = auto_jpda(dets, tracks, gate_threshold=10.0, use_mahalanobis=False)
        assert result is not None
        # result = (assoc_weights, unmatched_dets, unmatched_trks)
        assert len(result) == 3

    def test_empty_inputs(self):
        """空输入不崩"""
        result = auto_jpda([], [], gate_threshold=10.0, use_mahalanobis=False)
        assert result is not None
        assert len(result) == 3
