"""
BEV 融合可视化单测
"""
import numpy as np
import pytest
from visualization import render_bev_fusion
from core.data_types import GroundTruthObj, TrackedObject, Detection


@pytest.fixture
def sample_frame():
    """最小可用场景"""
    ego = np.array([0.0, 0.0, 0.0])
    gt = [
        GroundTruthObj(object_id=1, timestamp=0.0,
                       position=np.array([20, 0, 0]), velocity=np.array([20, 0, 0]),
                       size=np.array([4.5, 1.8, 1.5]), heading=0.0, obj_type='car'),
    ]
    trk = [
        TrackedObject(track_id=1, timestamp=0.0, position=np.array([20, 0, 0]),
                      velocity=np.array([20, 0, 0]), covariance=np.eye(6),
                      age=10, hits=10, miss_streak=0, source_sensors=set()),
    ]
    lidar = np.array([[20, 0, 0], [21, 0, 0], [22, 0, 0]], dtype=float)
    lidar_intens = np.array([0.5, 0.6, 0.7])
    return ego, gt, trk, lidar, lidar_intens


class TestBEV:
    """BEV 融合视图"""

    def test_basic_render(self, sample_frame, tmp_path):
        ego, gt, trk, lidar, intens = sample_frame
        out = str(tmp_path / "bev.html")
        fig = render_bev_fusion(ego, gt, trk, lidar, intens,
                                range_m=60, out_path=out)
        assert fig is not None
        assert len(fig.data) > 0
        import os
        assert os.path.exists(out)

    def test_no_lidar(self, sample_frame):
        """没 LiDAR 也能渲染"""
        ego, gt, trk, _, _ = sample_frame
        fig = render_bev_fusion(ego, gt, trk, None, None, range_m=60)
        assert fig is not None

    def test_with_radar(self, sample_frame):
        """带 Radar"""
        ego, gt, trk, lidar, intens = sample_frame
        radar = [Detection('radar', 0.0, np.array([20, 0, 0]),
                           np.array([20, 0, 0]), 1, 0.9,
                           {'rcs_dbsm': 10, 'range_m': 20})]
        fig = render_bev_fusion(ego, gt, trk, lidar, intens, radar_detections=radar, range_m=60)
        assert fig is not None

    def test_with_camera(self, sample_frame):
        """带 Camera"""
        ego, gt, trk, lidar, intens = sample_frame
        cam = [Detection('camera', 0.0, np.array([20, 0, 0]),
                         np.array([20, 0, 0]), 1, 0.85,
                         {'bbox_2d': (100, 100, 200, 200), 'size': np.array([4.5, 1.8])})]
        fig = render_bev_fusion(ego, gt, trk, lidar, intens, camera_detections=cam, range_m=60)
        assert fig is not None

    def test_no_nan_with_bad_cov(self, sample_frame):
        """协方差奇异时不应该崩"""
        ego, gt, trk, lidar, intens = sample_frame
        trk[0].covariance = np.eye(6) * 0  # 奇异
        fig = render_bev_fusion(ego, gt, trk, lidar, intens, range_m=60)
        assert fig is not None
