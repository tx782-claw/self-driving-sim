"""
nuScenes → self-driving-sim SimFrame 适配器.

nuScenes mini (4GB, 10 场景) 是当前首选:
  - 5 类传感器 1:1 对齐 self-driving-sim (1 LiDAR + 5 Radar + 6 Cam + IMU + GPS)
  - 完整 3D bbox 标注 + 32 类语义
  - 时间戳与 20Hz 主循环天然对齐 (LIDAR_TOP ≈ 12Hz, RADAR ≈ 13Hz)
  - License: CC BY-NC-SA 4.0 (非商用, 衍生作品需同协议)

数据流:
  nuScenes sample
    ├─ sample_annotation (3D bbox GT) ──→ Detection (按 sensor_id 字典)  ─→ tracker
    └─ ego_pose + calibrated_sensor      ──→ EgoState + 坐标系变换

两种模式:
  - 'gt' (默认): annotation 直接当 detection (无 sensor noise).
    用于验证 schema + tracker pipeline 在真实数据上不崩.
  - 'noisy': 在 GT 上叠加 RangeNoiseModel 模拟 sensor noise.
    用于验证 tracker 在 sensor noise 下的鲁棒性.

下载指引 (用户自己跑):
  1. 注册: https://www.nuscenes.org/sign-up
  2. 下载 mini: https://www.nuscenes.org/download (v1.0-mini, 4GB)
  3. 解压到 /data/nuscenes/, 应有 v1.0-mini/ + samples/ + sweeps/
  4. pip install nuscenes-devkit
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from core.data_types import (
    Detection,
    EgoState,
    GroundTruthObj,
    SimFrame,
)


# nuScenes category_name → self-driving-sim obj_type
# 只映射 self-driving-sim 支持的 4 类; 其它归 pedestrian
NUSCENES_CATEGORY_MAP = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "truck",
    "vehicle.bus.rigid": "truck",
    "vehicle.trailer": "truck",
    "vehicle.construction": "truck",
    "vehicle.motorcycle": "cyclist",
    "vehicle.bicycle": "cyclist",
    "vehicle.pedestrian": "pedestrian",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "animal": "pedestrian",  # 兜底
    "movable_object.barrier": "pedestrian",
    "movable_object.trafficcone": "pedestrian",
    "movable_object.pushable_pullable": "pedestrian",
    "movable_object.debris": "pedestrian",
    "static_object.bicycle_rack": "pedestrian",
}

# 每个 nuScenes sensor channel 在 self-driving-sim 中的 sensor_id 命名约定
SENSOR_CHANNEL_MAP = {
    "LIDAR_TOP": "lidar_top",
    "RADAR_FRONT": "radar_front",
    "RADAR_FRONT_LEFT": "radar_front_left",
    "RADAR_FRONT_RIGHT": "radar_front_right",
    "RADAR_BACK_LEFT": "radar_back_left",
    "RADAR_BACK_RIGHT": "radar_back_right",
    "CAM_FRONT": "camera_front",
    "CAM_FRONT_LEFT": "camera_front_left",
    "CAM_FRONT_RIGHT": "camera_front_right",
    "CAM_BACK": "camera_back",
    "CAM_BACK_LEFT": "camera_back_left",
    "CAM_BACK_RIGHT": "camera_back_right",
}


def _quat_to_yaw(q: List[float]) -> float:
    """nuScenes quaternion [w, x, y, z] → yaw (弧度, global frame)."""
    w, x, y, z = q
    # yaw 提取: 与全局 z 轴对齐的旋转角度
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _quat_to_rotmat(q: List[float]) -> np.ndarray:
    """nuScenes quaternion [w, x, y, z] → 3x3 旋转矩阵."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class NuScenesAdapter:
    """nuScenes 数据集 → self-driving-sim SimFrame 适配器.

    Examples:
        >>> adapter = NuScenesAdapter('/data/nuscenes', version='v1.0-mini')
        >>> print(adapter.list_scenes())  # 10 个 mini 场景
        >>> for frame in adapter.load_scene('scene-0061'):
        ...     tracks = tracker.update(frame.detections_by_sensor)
        ...     evaluate(frame, tracks)
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        mode: str = "gt",
        noise_model=None,
        verbose: bool = False,
        deduplicate: bool = False,
        load_sensor_data: bool = False,
    ):
        """
        Args:
            dataroot: nuScenes 解压根目录, 需含 v1.0-mini/ + samples/ + sweeps/
            version: 'v1.0-mini' (4GB, 10 场景) 或 'v1.0-trainval' (350GB)
            mode: 'gt' (无噪声) 或 'noisy' (叠加 RangeNoiseModel)
            noise_model: sensors.RangeNoiseModel 实例, mode='noisy' 时必传
            verbose: 是否打印加载进度
            deduplicate: False=保留原始 12 传感器通道 (默认, tracker 依赖多传感器冗余);
                       True=同 GT instance 只输出 1 个 detection (高级用户 / 自定义 tracker 用)
            load_sensor_data: False=仅返回 detections/GT (默认, 快速, 可跟踪);
                        True=加载 lidar_top 点云 + 6 个 camera jpg (慢,
                              但 webui 能呈现原始 sensor 数据; 默认 30 帧/n 场景用=True 合适)
        """
        if mode not in ("gt", "noisy"):
            raise ValueError(f"mode must be 'gt' or 'noisy', got {mode!r}")
        if mode == "noisy" and noise_model is None:
            raise ValueError("mode='noisy' requires a RangeNoiseModel instance")

        try:
            from nuscenes.nuscenes import NuScenes  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "需要 nuscenes-devkit: pip install nuscenes-devkit"
            ) from e

        from nuscenes.nuscenes import NuScenes

        dataroot = str(Path(dataroot).expanduser())
        if not os.path.isdir(dataroot):
            raise FileNotFoundError(f"dataroot 不存在: {dataroot}")
        if not os.path.isdir(os.path.join(dataroot, version)):
            raise FileNotFoundError(
                f"找不到 {version}/ 子目录, 请检查解压是否完整: {dataroot}"
            )

        self.nuscenes = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
        self.mode = mode
        self.noise_model = noise_model
        self.version = version
        self.dataroot = dataroot
        self.deduplicate = deduplicate
        self.load_sensor_data = load_sensor_data

    # ----------------- 场景元信息 -----------------

    def list_scenes(self) -> List[str]:
        """返回所有场景名称 (mini 通常 10 个, trainval 通常 850 个)."""
        return [s["name"] for s in self.nuscenes.scene]

    def scene_count(self) -> int:
        return len(self.nuscenes.scene)

    def frame_count(self, scene_name: str) -> int:
        """返回 scene 的 LIDAR_TOP 帧数 (≈ 12Hz × 时长)."""
        scene = self._get_scene(scene_name)
        count = 0
        token = scene["first_sample_token"]
        while token:
            sample = self.nuscenes.get("sample", token)
            if "LIDAR_TOP" in sample["data"]:
                count += 1
            token = sample["next"]
        return count

    def scene_duration_s(self, scene_name: str) -> float:
        """返回 scene 时长 (秒)."""
        scene = self._get_scene(scene_name)
        first_sample = self.nuscenes.get("sample", scene["first_sample_token"])
        last_sample = self.nuscenes.get("sample", scene["last_sample_token"])
        return (last_sample["timestamp"] - first_sample["timestamp"]) / 1e6

    # ----------------- 主入口 -----------------

    def load_scene(self, scene_name: str) -> Iterator[SimFrame]:
        """按 LIDAR_TOP 帧率 (≈12Hz) 遍历 scene, yield SimFrame.

        Yields:
            SimFrame, 含:
              - ego_state: EgoState (global 坐标, heading=yaw)
              - ground_truth: List[GroundTruthObj] (ego 坐标)
              - detections_by_sensor: Dict[sensor_id, List[Detection]]
                (每个 radar 通道一份, lidar_top 一份, 所有相机通道一份)
              - tracks: 空列表 (由调用方用 tracker.update() 填充)
        """
        scene = self._get_scene(scene_name)
        sample_token = scene["first_sample_token"]

        while sample_token:
            sample = self.nuscenes.get("sample", sample_token)
            frame = self._build_frame(sample)
            if frame is not None:
                yield frame
            sample_token = sample["next"]

    # ----------------- 帧构建 -----------------

    def _get_scene(self, scene_name: str) -> dict:
        for s in self.nuscenes.scene:
            if s["name"] == scene_name:
                return s
        available = ", ".join(self.list_scenes()[:5])
        raise KeyError(
            f"scene {scene_name!r} 不存在, 前 5 个可用: {available}..."
        )

    def _build_frame(self, sample: dict) -> Optional[SimFrame]:
        """从一个 nuScenes sample 构建 SimFrame.

        帧源: LIDAR_TOP 的 sample_data 时间戳 (≈ 12Hz).
        """
        if "LIDAR_TOP" not in sample["data"]:
            return None  # 无 lidar, 跳过

        lidar_sd = self.nuscenes.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = self.nuscenes.get("ego_pose", lidar_sd["ego_pose_token"])
        calibrated_sensor = self.nuscenes.get(
            "calibrated_sensor", lidar_sd["calibrated_sensor_token"]
        )

        timestamp_s = lidar_sd["timestamp"] / 1e6

        # 1. EgoState (global 坐标)
        ego_yaw_global = _quat_to_yaw(ego_pose["rotation"])
        ego_state = EgoState(
            timestamp=timestamp_s,
            position=np.array(ego_pose["translation"], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),  # mini 缺 ego velocity, 简化为 0
            acceleration=np.zeros(3, dtype=np.float32),
            heading=ego_yaw_global,
            angular_velocity=np.zeros(3, dtype=np.float32),
        )

        # 2. Ground truth (ego 坐标)
        ground_truth = self._build_ground_truth(
            sample["anns"], ego_pose, timestamp_s
        )

        # 3. Detections 按 sensor 分组
        detections_by_sensor = self._build_detections_by_sensor(
            sample["anns"], sample["data"], ego_pose, timestamp_s
        )

        # 4. SimFrame (用 .detections_by_sensor 字段, 额外注入)
        frame = SimFrame(
            timestamp=timestamp_s,
            ego_state=ego_state,
            ground_truth=ground_truth,
            tracks=[],
        )
        # SimFrame 没有原生 detections_by_sensor 字段, 用 all_sensors 暂存
        # 实际 tracker 调用应走 frame.all_sensors[...].detections
        # 这里把 detection 列表塞进一个临时映射, 由调用方决定怎么喂 tracker
        frame.detections_by_sensor = detections_by_sensor  # type: ignore[attr-defined]

        # 5. 原始 sensor 数据 (webui BEV fusion / camera panel 需要)
        #    仅当 load_sensor_data=True 时加载 (IO 成本, mini 每帧 ~40k LIDAR 点 + 1.4MB JPG)
        if self.load_sensor_data:
            all_sensors = self._build_sensor_data(sample, lidar_sd, timestamp_s)
            frame.all_sensors = all_sensors  # type: ignore[attr-defined]

        return frame

    def _build_sensor_data(
        self, sample: dict, lidar_sd: dict, timestamp_s: float
    ) -> dict:
        """加载原始 sensor 数据: lidar_top 点云 + 6 个 camera jpg.

        Returns: Dict[sensor_id, LidarScan | CameraImage | RadarTrack]
        """
        from core.data_types import LidarScan, CameraImage, RadarTrack
        from PIL import Image

        sensors = {}
        dataroot = self.dataroot

        # --- LiDAR_TOP 点云 (float32 x,y,z,intensity → (N,4) → 拆为 points + intensity) ---
        for nusc_channel, sensor_id in [
            ("LIDAR_TOP", "lidar_top"),
        ]:
            if nusc_channel not in sample["data"]:
                continue
            sd = self.nuscenes.get("sample_data", sample["data"][nusc_channel])
            filepath = os.path.join(dataroot, sd["filename"])
            if os.path.isfile(filepath):
                pts = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
                # nuScenes: x=前, y=左, z=上 (与 self-driving-sim 同)
                sensors[sensor_id] = LidarScan(
                    sensor_id=sensor_id,
                    timestamp=timestamp_s,
                    points=pts[:, :3].astype(np.float32),
                    intensity=pts[:, 3].astype(np.float32),
                )

        # --- 6 个 Camera JPG (顺序: front/front_left/front_right/back/back_left/back_right) ---
        for nusc_channel, sensor_id in [
            ("CAM_FRONT", "camera_front"),
            ("CAM_FRONT_LEFT", "camera_front_left"),
            ("CAM_FRONT_RIGHT", "camera_front_right"),
            ("CAM_BACK", "camera_back"),
            ("CAM_BACK_LEFT", "camera_back_left"),
            ("CAM_BACK_RIGHT", "camera_back_right"),
        ]:
            if nusc_channel not in sample["data"]:
                continue
            sd = self.nuscenes.get("sample_data", sample["data"][nusc_channel])
            filepath = os.path.join(dataroot, sd["filename"])
            if os.path.isfile(filepath):
                try:
                    img = np.array(Image.open(filepath).convert("RGB"))
                    sensors[sensor_id] = CameraImage(
                        sensor_id=sensor_id,
                        timestamp=timestamp_s,
                        image=img,
                    )
                except Exception:
                    pass  # 图损坏跳过, 不打破其他 sensor

        # --- Radar (nuScenes RADAR 存为 .pcd 但仅 ~6 dims, 加载逻辑跳过. user 可以后期扩展) ---
        # 为避免 webui "No Radar data" 警告, 造空 RadarTrack 记录存在.
        for nusc_channel, sensor_id in [
            ("RADAR_FRONT", "radar_front"),
            ("RADAR_FRONT_LEFT", "radar_front_left"),
            ("RADAR_FRONT_RIGHT", "radar_front_right"),
            ("RADAR_BACK_LEFT", "radar_back_left"),
            ("RADAR_BACK_RIGHT", "radar_back_right"),
        ]:
            if nusc_channel in sample["data"]:
                sensors[sensor_id] = RadarTrack(
                    sensor_id=sensor_id,
                    timestamp=timestamp_s,
                    detections=[],  # nuScenes RADAR raw 加载待扩展
                )

        return sensors

    def _build_ground_truth(
        self, ann_tokens: List[str], ego_pose: dict, timestamp_s: float
    ) -> List[GroundTruthObj]:
        """annotations → ego 坐标系 GroundTruthObj 列表."""
        ego_translation = np.array(ego_pose["translation"])
        ego_rotmat_inv = _quat_to_rotmat(ego_pose["rotation"]).T
        ego_yaw = _quat_to_yaw(ego_pose["rotation"])

        results: List[GroundTruthObj] = []
        for token in ann_tokens:
            ann = self.nuscenes.get("sample_annotation", token)
            global_pos = np.array(ann["translation"], dtype=np.float64)
            ego_pos = ego_rotmat_inv @ (global_pos - ego_translation)

            global_vel = np.array(ann.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float64)
            ego_vel = ego_rotmat_inv @ global_vel

            obj_type = NUSCENES_CATEGORY_MAP.get(
                ann.get("category_name", ""), "pedestrian"
            )

            # nuScenes size = [width, length, height] → 转 [length, width, height]
            w, l, h = ann["size"]
            size_lwh = np.array([l, w, h], dtype=np.float32)

            # annotation heading 转 ego frame
            ann_yaw_global = _quat_to_yaw(ann["rotation"])
            heading_ego = ann_yaw_global - ego_yaw
            heading_ego = (heading_ego + np.pi) % (2 * np.pi) - np.pi

            # 用 instance_token 末段哈希成稳定 int ID
            object_id = int(ann["instance_token"][:8], 16) % (10**6)

            results.append(
                GroundTruthObj(
                    object_id=object_id,
                    timestamp=timestamp_s,
                    obj_type=obj_type,
                    position=ego_pos.astype(np.float32),
                    velocity=ego_vel.astype(np.float32),
                    heading=float(heading_ego),
                    size=size_lwh,
                )
            )
        return results

    def _build_detections_by_sensor(
        self,
        ann_tokens: List[str],
        data_tokens: dict,
        ego_pose: dict,
        timestamp_s: float,
    ) -> dict:
        """每个传感器通道生成独立的 Detection 列表 (含 sensor_id).

        关键设计: 同一个目标在 lidar / radar / camera 三个通道里都生成 detection,
        因为 nuScenes 标注是 3D bbox 通用, 模拟 "传感器各自检测出同一个目标".
        """
        ego_translation = np.array(ego_pose["translation"])
        ego_rotmat_inv = _quat_to_rotmat(ego_pose["rotation"]).T

        # 先算 ego 坐标 + range (所有 sensor 共享)
        per_target_ego = []
        for token in ann_tokens:
            ann = self.nuscenes.get("sample_annotation", token)
            global_pos = np.array(ann["translation"], dtype=np.float64)
            ego_pos = ego_rotmat_inv @ (global_pos - ego_translation)
            global_vel = np.array(ann.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float64)
            ego_vel = ego_rotmat_inv @ global_vel
            range_m = float(np.linalg.norm(ego_pos))

            obj_type = NUSCENES_CATEGORY_MAP.get(
                ann.get("category_name", ""), "pedestrian"
            )
            confidence = float(ann.get("visibility_token", 1))  # 1~4, 简化为置信度
            confidence = min(1.0, 0.25 * confidence)  # 1→0.25, 4→1.0

            per_target_ego.append({
                "position": ego_pos.astype(np.float32),
                "velocity": ego_vel.astype(np.float32),
                "range_m": range_m,
                "obj_type": obj_type,
                "category_name": ann.get("category_name", ""),
                "confidence": confidence,
                "size": np.array(ann["size"], dtype=np.float32),  # w, l, h
            })

        # 按 sensor 通道生成 detection 列表
        result = {}
        if self.deduplicate:
            # 去重: 同一 GT instance 只输出 1 个 detection (lidar_top 通道)
            # 推荐 tracker 使用, 避免 12 通道 × GT 出现重复 track
            dets = []
            for tgt in per_target_ego:
                det = self._build_one_detection("lidar_top", timestamp_s, tgt)
                if det is not None:
                    dets.append(det)
            result["lidar_top"] = dets
        else:
            # 多传感器模式: 同一 GT 在 12 传感器通道都生成 detection (调试/融合研究用)
            for nusc_channel, sensor_id in SENSOR_CHANNEL_MAP.items():
                if nusc_channel not in data_tokens:
                    continue  # 该通道在当前 sample 无数据

                dets = []
                for tgt in per_target_ego:
                    det = self._build_one_detection(sensor_id, timestamp_s, tgt)
                    if det is not None:
                        dets.append(det)
                result[sensor_id] = dets

        return result

    def _build_one_detection(
        self, sensor_id: str, timestamp_s: float, tgt: dict
    ) -> Optional[Detection]:
        """根据 mode 生成单条 Detection. noisy 模式下按 range 概率漏检."""
        if self.mode == "noisy":
            detected = self.noise_model.sample(tgt["range_m"])
            if not detected:
                return None

        position = tgt["position"].copy()
        velocity = tgt["velocity"].copy()

        # noisy 模式: 叠加位置噪声
        if self.mode == "noisy":
            sigma, _ = self.noise_model.at(tgt["range_m"])
            position = position + np.random.normal(0, sigma, size=3).astype(np.float32)
            # 速度噪声: 按位置噪声比例放大 5× (速度积分误差)
            velocity = velocity + np.random.normal(
                0, sigma * 5.0, size=3
            ).astype(np.float32)

        # 传感器特定 attributes
        attributes = {
            "range_m": tgt["range_m"],
            "category": tgt["category_name"],
            "size_wlh": tgt["size"].tolist(),  # [w, l, h]
        }
        if sensor_id.startswith("radar"):
            # 径向速度分量 (Radar 只测径向)
            radial_v = float(np.dot(velocity, tgt["position"] / max(tgt["range_m"], 1e-6)))
            attributes["doppler_mps"] = radial_v
            attributes["rcs_dbsm"] = float(np.log10(max(np.prod(tgt["size"]), 0.01)) - 1.0)
        elif sensor_id.startswith("camera"):
            attributes["depth_m"] = tgt["range_m"]
            # bbox_2d 占位 (self-driving-sim Camera 不渲染真像素, 这里不生成 2D bbox)
        elif sensor_id.startswith("lidar"):
            attributes["num_points"] = max(10, int(200 / (1.0 + tgt["range_m"] / 10.0)))

        return Detection(
            sensor_id=sensor_id,
            timestamp=timestamp_s,
            position=position,
            velocity=velocity,
            object_id=None,  # detection 阶段不暴露 GT ID
            confidence=tgt["confidence"],
            attributes=attributes,
        )


def has_nuscenes_mini(dataroot: str) -> bool:
    """快速检查 nuScenes mini 是否就绪 (不实际加载)."""
    p = Path(dataroot).expanduser()
    return (p / "v1.0-mini").is_dir()