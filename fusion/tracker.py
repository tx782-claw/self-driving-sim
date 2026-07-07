"""
多目标跟踪器 - P2 增强版
- Track 合并去重 (复活最近死的 track)
- 候选/确认状态机
- 速度估计
- 置信度加权关联
- JPDA 关联模式（联合概率数据关联） - 实验功能
- UKF 滤波模式（无迹卡尔曼，可选）
- IMM 滤波模式（CV+CA 交互多模型，可选） - v0.2.2 新增
- P3-A 自车运动补偿 (v0.4 新增) - 接受 ego_motion 参数,补偿 IMU 不确定性到过程噪声
"""
import numpy as np
from collections import deque
from core.data_types import Detection
from .ekf import EKFTrack
from .ukf import UKFTrack
from .imm import IMMTrack
from .association import hungarian_associate
from .jpda import auto_jpda
from .imu_predict import IMUEgoPredictor, extract_imu_from_sensors
from .gnns import gnns_associate
from .iekf import IEKFTrack, make_iekf_from_ekf, DEFAULT_MAX_ITER, DEFAULT_TOL


class MultiObjectTracker:
    """多目标跟踪器 (P2 增强版)"""
    def __init__(self, dt: float = 0.05,
                 gate_threshold: float = 10.0,
                 min_hits_to_confirm: int = 2,
                 max_miss_streak: int = 20,
                 initial_velocity: np.ndarray = None,
                 use_confidence_weighted: bool = True,
                 min_init_confidence: float = 0.3,
                 # ===== P2 复活机制 =====
                 revive_window_s: float = 2.0,
                 revive_distance_m: float = 8.0,
                 graveyard_max_size: int = 100,
                 # ===== P2 关联/滤波模式 =====
                 association_mode: str = 'hungarian',  # 'gnns' | 'jpda' | 'hungarian' (默认 hungarian，密集场景下更稳定)
                 # P3-B 新增 'gnns' - Mahalanobis + chi2 门限 + 匈牙利
                 use_ukf: bool = False,            # True=UKFTrack，False=EKFTrack
                 use_imm: bool = False,            # True=IMMTrack (CV+CA) - v0.2.2 新增，优先级高于 use_ukf
                 # ===== P3-D IEKF 升级 (v0.4) =====
                 use_iekf: bool = False,           # True=IEKFTrack (迭代 EKF) - 优先级 use_imm > use_iekf > use_ukf
                 iekf_max_iter: int = DEFAULT_MAX_ITER,
                 iekf_tol: float = DEFAULT_TOL,
                 JPDA_PD: float = 0.9,            # 检测概率
                 JPDA_clut_density: float = 1e-5,  # 杂波密度
                 JPDA_max_miss: int = 10,        # JPDA 模式专用：幽灵 track 死亡阈值（比 Hungarian 更激进）
                 # ===== P3-A 自车运动补偿 (v0.4) =====
                 use_ego_motion: bool = True,    # True=从 sensor_detections 提取 IMU 计算 ego_motion
                 imu_predictor: IMUEgoPredictor = None,  # 外部传入可复用,None 时内部创建
                 # ===== P3-B GNNS 关联参数 =====
                 gnns_gate_chi2: float = 7.815,   # GNNS chi2 门限 (3D 95% 分位)
                 gnns_allow_n_to_1: bool = True, # GNNS N-to-1 允许多传感器同时更新同一 track
                 ):
        self.dt = dt
        self.gate_threshold = gate_threshold
        self.min_hits = min_hits_to_confirm
        self.max_miss = max_miss_streak
        self.default_vel = initial_velocity if initial_velocity is not None else np.zeros(3)
        self.use_confidence_weighted = use_confidence_weighted
        self.min_init_confidence = min_init_confidence

        # P2: 复活机制
        self.revive_window_s = revive_window_s
        self.revive_distance_m = revive_distance_m

        # P2: 关联 + 滤波模式
        self.association_mode = association_mode  # 'gnns' | 'jpda' | 'hungarian'
        self.use_ukf = use_ukf                    # True=UKFTrack, False=EKFTrack
        self.use_imm = use_imm                    # True=IMMTrack, 优先于 use_ukf
        self.use_iekf = use_iekf                  # P3-D: True=IEKFTrack, 优先于 use_ukf
        self.iekf_max_iter = iekf_max_iter
        self.iekf_tol = iekf_tol
        self.JPDA_PD = JPDA_PD
        self.JPDA_clut_density = JPDA_clut_density
        # v0.2.2: JPDA 模式幽灵 track 死亡更激进 (15 vs 30)，减少 clutter 拖尾
        self._jpda_max_miss = JPDA_max_miss if association_mode == 'jpda' else max_miss_streak

        # P3-A: 自车运动补偿
        self.use_ego_motion = use_ego_motion
        self._imu_predictor = imu_predictor if imu_predictor is not None else IMUEgoPredictor(dt=dt)

        # P3-B: GNNS 关联参数
        self.gnns_gate_chi2 = gnns_gate_chi2
        self.gnns_allow_n_to_1 = gnns_allow_n_to_1

        # Track 列表
        self.confirmed_tracks: list = []
        self.candidate_tracks: list = []
        self.all_tracks: list = []
        self.next_id = 1
        self.last_t = None
        self.graveyard: deque = deque(maxlen=graveyard_max_size)

    def reset(self):
        self.confirmed_tracks = []
        self.candidate_tracks = []
        self.all_tracks = []
        self.next_id = 1
        self.last_t = None
        self.graveyard.clear()
        # P3-A: 重置 IMU 预测器
        if self._imu_predictor is not None:
            self._imu_predictor.reset()

    def update(self, sensor_detections: dict, timestamp: float, ego_motion: dict = None) -> list:
        """
        主更新接口

        Args:
            sensor_detections: {sensor_id: [Detection, ...]}
            timestamp: 当前时间戳
            ego_motion: 自车运动补偿 (P3-A 新增,可选)
                - None: 内部从 sensor_detections 提取 IMU 自动计算
                - dict: {'delta_position': ..., 'delta_velocity': ..., 'delta_yaw': ...}
                - 也可调用 _zero_ego_motion() 关掉
        """
        all_dets = []
        for sid, dets in sensor_detections.items():
            if sid.startswith(('imu', 'gps')):
                continue
            for d in dets:
                all_dets.append(d)

        # P3-A: 计算 ego_motion (如果未外部传入且 use_ego_motion=True)
        if ego_motion is None and self.use_ego_motion:
            imu_det = extract_imu_from_sensors(sensor_detections)
            if imu_det is not None:
                ego_motion = self._imu_predictor.update(imu_det, dt=self.dt)
            # else: ego_motion 保持 None,各 track predict 不用补偿
        # 兼容旧调用: ego_motion=None 时不补偿

        tracks_for_assoc = self.all_tracks

        if self.association_mode == 'jpda':
            assoc_weights, unmatched_dets, unmatched_trks = auto_jpda(
                all_dets, tracks_for_assoc,
                gate_threshold=self.gate_threshold,
                use_mahalanobis=True, debug=False)

            # NN-限制：每个 detection 只能作为一个 track 的主更新
            # 避免多个 track 竞争同一 detection、谁都不死的问题
            candidates = []  # (beta, track_idx, det_idx)
            for trk_idx, wlist in assoc_weights.items():
                for det_idx, beta in wlist:
                    if beta > 0.01:
                        candidates.append((beta, trk_idx, det_idx))
            candidates.sort(key=lambda x: -x[0])
            used_dets = set()
            tracks_updated = set()
            for beta, trk_idx, det_idx in candidates:
                if det_idx in used_dets:
                    continue
                used_dets.add(det_idx)
                trk = tracks_for_assoc[trk_idx]
                det = all_dets[det_idx]
                # JPDA: 高 β update 加上有效 det_confidence
                # 低 β update 会污染 track 速度估计（已采用 NN 限制防止严重问题）
                effective_conf = det.confidence * beta
                # P3-A: 传入 ego_motion 到 update
                trk.update(det.position, det.sensor_id, timestamp,
                           det_confidence=effective_conf,
                           class_label=det.attributes.get('class'),
                           ego_motion=ego_motion)
                tracks_updated.add(trk_idx)
                # 记录本 track 还看过的其他检测
                for di2, _ in assoc_weights.get(trk_idx, []):
                    if di2 != det_idx:
                        trk.source_sensors.add(all_dets[di2].sensor_id)
                # JPDA 速度稳定：限制横向速度 |vy| <= 3 m/s（避免在密集场景下
                # track 被弱匹配检测拉偏横向速度而沿 y 轴走偏）
                if abs(trk.kf.x[4]) > 3.0:
                    trk.kf.x[4] = 0.0
                if abs(trk.kf.x[5]) > 3.0:
                    trk.kf.x[5] = 0.0

            # 其他有有效关联但没赢主检测的 track → miss
            for trk_idx in assoc_weights.keys():
                if trk_idx not in tracks_updated:
                    tracks_for_assoc[trk_idx].miss(timestamp)
            # 不在 assoc_weights 里的 track → miss
            for trk_idx in unmatched_trks:
                tracks_for_assoc[trk_idx].miss(timestamp)

        else:
            # ── 匈牙利模式 / GNNS 模式 ──────────────────────────
            if self.association_mode == 'gnns':
                # P3-B: GNNS 关联 (Mahalanobis + chi2 + 贪心)
                matched, unmatched_dets, unmatched_trks = gnns_associate(
                    all_dets, tracks_for_assoc,
                    gate_chi2=self.gnns_gate_chi2,
                    confidence_weighted=self.use_confidence_weighted,
                    allow_n_to_1=self.gnns_allow_n_to_1,
                    debug=False)
            else:
                # 标准匈牙利 (P0-P2 默认)
                matched, unmatched_dets, unmatched_trks = hungarian_associate(
                    all_dets, tracks_for_assoc,
                    gate_threshold=self.gate_threshold,
                    confidence_weighted=self.use_confidence_weighted,
                    debug=False, greedy_multi=True)

            for det_idx, trk_idx in matched:
                det = all_dets[det_idx]
                trk = tracks_for_assoc[trk_idx]
                # P3-A: 传入 ego_motion 到 update
                trk.update(
                    det.position, det.sensor_id, timestamp,
                    det_confidence=det.confidence,
                    class_label=det.attributes.get('class'),
                    ego_motion=ego_motion)

            for trk_idx in unmatched_trks:
                tracks_for_assoc[trk_idx].miss(timestamp)

        # ── 未匹配检测 → 复活墓碑或新建 ─────────────────────────
        new_tracks_this_frame = []
        for det_idx in unmatched_dets:
            det = all_dets[det_idx]
            if det.confidence < self.min_init_confidence:
                continue
            revived = self._try_revive(det, timestamp)
            if not revived:
                new_trk = self._create_track(det, timestamp)
                self.candidate_tracks.append(new_trk)
                self.all_tracks.append(new_trk)
                new_tracks_this_frame.append(new_trk)

        if new_tracks_this_frame:
            self._merge_close_tracks(new_tracks_this_frame)

        # ── 状态机提升/降级 ──────────────────────────────────────
        new_candidate = []
        for trk in self.candidate_tracks:
            if trk.hits >= self.min_hits:
                self.confirmed_tracks.append(trk)
            else:
                new_candidate.append(trk)
        self.candidate_tracks = new_candidate

        # ── 清理失效 tracks → 移入墓碑 ───────────────────────────
        # JPDA Fix: 用更短的 miss_streak 阈值（15 vs 30）加速幽灵 track 消亡
        max_miss = self._jpda_max_miss if self.association_mode == 'jpda' else self.max_miss
        alive_tracks = []
        for trk in self.all_tracks:
            if trk.is_alive(max_miss):
                alive_tracks.append(trk)
            else:
                if trk.hits >= self.min_hits:
                    self.graveyard.append((trk, timestamp, trk.kf.x[:3].copy()))
        self.all_tracks = alive_tracks
        self.confirmed_tracks = [t for t in self.confirmed_tracks if t.is_alive(max_miss)]
        self.candidate_tracks = [t for t in self.candidate_tracks if t.is_alive(max_miss)]

        while self.graveyard and (timestamp - self.graveyard[0][1]) > self.revive_window_s:
            self.graveyard.popleft()

        result = [trk.to_track(timestamp) for trk in self.all_tracks if trk.hits >= 1]
        self.last_t = timestamp
        return result

    def _try_revive(self, det: Detection, timestamp: float) -> bool:
        best = None
        best_d = self.revive_distance_m
        for trk, died_at, last_pos in self.graveyard:
            time_since_death = timestamp - died_at
            if time_since_death > self.revive_window_s:
                continue
            d = np.linalg.norm(det.position - last_pos)
            if d < best_d:
                adjusted_d = d + time_since_death * 5.0
                if adjusted_d < best_d * 1.5:
                    best = trk
                    best_d = d
        if best is None:
            return False

        best.miss_streak = 0
        best.hits += 1
        best.age += 1
        best.source_sensors.add(det.sensor_id)
        if best not in self.all_tracks:
            self.all_tracks.append(best)
            if best.hits >= self.min_hits:
                if best in self.candidate_tracks:
                    self.candidate_tracks.remove(best)
                if best not in self.confirmed_tracks:
                    self.confirmed_tracks.append(best)
            else:
                if best not in self.candidate_tracks:
                    self.candidate_tracks.append(best)
        dt = timestamp - best.last_t if timestamp > best.last_t else self.dt
        # 统一接口：UKF 和 EKF 都用 miss/update（不直接操作 kf.F / kf.predict）
        best.miss(timestamp, dt=dt)
        best.update(det.position, det.sensor_id, timestamp, dt=dt,
                    det_confidence=det.confidence)
        best.last_t = timestamp

        for i, (t, _, _) in enumerate(self.graveyard):
            if t is best:
                del self.graveyard[i]
                break
        return True

    def _create_track(self, det: Detection, timestamp: float):
        """根据 use_imm / use_iekf / use_ukf 选 IMMTrack / IEKFTrack / UKFTrack / EKFTrack
        优先级: use_imm > use_iekf > use_ukf > EKFTrack
        JPDA 模式下用更小的过程噪声 (0.3 vs 1.0)，防止 track 创建时速度被弱匹配拉偏
        """
        if self.association_mode == 'jpda':
            process_noise = 0.3
        else:
            process_noise = 1.0
        if self.use_imm:
            new_trk = IMMTrack(
                track_id=self.next_id,
                initial_pos=det.position,
                initial_vel=self.default_vel,
                dt=self.dt,
                timestamp=timestamp,
                process_noise_vel=process_noise,
            )
        elif self.use_iekf:
            # P3-D: IEKFTrack
            new_trk = IEKFTrack(
                track_id=self.next_id,
                initial_pos=det.position,
                initial_vel=self.default_vel,
                dt=self.dt,
                timestamp=timestamp,
                process_noise_vel=process_noise,
                iekf_max_iter=self.iekf_max_iter,
                iekf_tol=self.iekf_tol,
            )
        elif self.use_ukf:
            new_trk = UKFTrack(
                track_id=self.next_id,
                initial_pos=det.position,
                initial_vel=self.default_vel,
                dt=self.dt,
                timestamp=timestamp,
                process_noise_vel=process_noise,
            )
        else:
            new_trk = EKFTrack(
                track_id=self.next_id,
                initial_pos=det.position,
                initial_vel=self.default_vel,
                dt=self.dt,
                timestamp=timestamp,
                process_noise_vel=process_noise,
            )
        new_trk.hits = 1
        new_trk.source_sensors.add(det.sensor_id)
        if 'class' in det.attributes:
            new_trk.class_label = det.attributes['class']
        self.next_id += 1
        return new_trk

    def _merge_close_tracks(self, new_tracks: list):
        """合并同帧内位置接近的新 track"""
        if len(new_tracks) < 2:
            return
        merge_distance = 5.0
        merged = set()
        for i, t1 in enumerate(new_tracks):
            if i in merged:
                continue
            for j in range(i + 1, len(new_tracks)):
                if j in merged:
                    continue
                t2 = new_tracks[j]
                d = np.linalg.norm(t1.kf.x[:3] - t2.kf.x[:3])
                if d < merge_distance:
                    main, other = (t1, t2) if t1.age >= t2.age else (t2, t1)
                    main.source_sensors |= other.source_sensors
                    main.hits += other.hits
                    for lst in (self.candidate_tracks, self.all_tracks, self.confirmed_tracks):
                        if other in lst:
                            lst.remove(other)
                    merged.add(j)

    def get_all_tracks(self) -> list:
        return [trk.to_track(self.last_t) for trk in self.all_tracks]
