"""
core/event_builder.py

Assembles structured Kafka events from per-frame tracking data.

Two event types:
  vehicle_detected  — emitted for every active track every frame
  traffic_snapshot  — emitted every SNAPSHOT_EVERY frame (scene summary)

JSON schema is stable — adding new fields is backward compatible.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple

from traffic_analyzer.core.vehicle_metrics import VehicleMetrics
from traffic_analyzer.core.scene_metrics import TrafficMetrics, SNAPSHOT_EVERY
from traffic_analyzer.core.anomaly_detector import AnomalyDetector
from traffic_analyzer.utils.config_loader import AppConfig
from traffic_analyzer.visualization.colors import CLASS_LABELS

HEAVY_CLASSES = {5, 7}  # Bus=5, Truck=7

# Minimum frames a track must be observed before a vehicle_detected event is
# emitted. This guards against:
#   - Partially-entered bounding boxes (first 1-2 frames the bbox clips the edge)
#   - ByteTrack's minimum_consecutive_frames warm-up period
#   - Lane assignment returning None due to centroid still outside all polygons
# Value must be > tracker's minimum_consecutive_frames (default=3) so the track
# is confirmed AND its centroid is stable before we send anything to Kafka.
MIN_FRAMES_BEFORE_EVENT = 5


class EventBuilder:
    """
    Owns a VehicleMetrics instance and produces JSON-serialisable
    event dicts every frame.

    Usage (inside Analyzer._process_frame):
        events = self._event_builder.update(frame_id, frame_h, active_tracks)
        for e in events:
            self._producer.send(e)
    """

    def __init__(self, config: AppConfig):
        self._cfg = config
        self._vm = VehicleMetrics()
        self._roi_area = TrafficMetrics.roi_total_area(config.lanes)
        self._anomaly_detector = AnomalyDetector(self._vm)

    @property
    def vm(self) -> VehicleMetrics:
        """Expose VehicleMetrics so Analyzer can update it directly,
        keeping a single source of truth for all kinematic state."""
        return self._vm

    # ── MAIN ENTRY POINT ─────────────────────────────────────────────────────

    def update(self, frame_id: int, frame_height: int,
               active_tracks: List[Dict]) -> List[Dict]:
        """
        Call once per frame.

        active_tracks format:
            [{"tid": int, "cls_id": int, "box": (x1, y1, x2, y2)}, ...]

        Returns list of event dicts (vehicle events + optional snapshot).
        """
        events = []
        active_ids = set()
        valid_snapshot_tracks = []

        for track in active_tracks:
            tid = track["tid"]
            cls_id = track["cls_id"]
            box = track["box"]
            active_ids.add(tid)

            # Note: vm.update() is called by Analyzer before this method,
            # so kinematic state is already current for this frame.
            vehicle_event = self._build_vehicle_event(tid, cls_id, box, frame_id)

            # Gate 1 — bbox not yet stable (partially entered frame / tracker warmup)
            if vehicle_event["residence"]["frames_in_roi"] < MIN_FRAMES_BEFORE_EVENT:
                continue

            # Gate 2 — centroid still outside all lane polygons
            # This is a safety net; Gate 1 should eliminate most of these cases.
            # If lane is still None after MIN_FRAMES_BEFORE_EVENT frames the vehicle
            # is genuinely outside all defined lanes — skip it.
            if vehicle_event["vehicle"]["lane"] is None:
                continue

            events.append(vehicle_event)
            valid_snapshot_tracks.append(track)

        if frame_id % SNAPSHOT_EVERY == 0:
            events.append(
                self._build_snapshot(frame_id, valid_snapshot_tracks, frame_height)
            )

        self._vm.cleanup(active_ids)
        return events

    # ── VEHICLE EVENT ────────────────────────────────────────────────────────

    def _build_vehicle_event(self, tid: int, cls_id: int,
                             box: Tuple, frame_id: int) -> Dict:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        speed = self._vm.get_speed(tid)
        direction = self._vm.get_direction(tid)
        stopped = self._vm.is_stopped(tid)
        residence = self._vm.get_residence(tid, frame_id)
        lane = TrafficMetrics.get_lane(cx, cy, self._cfg.lanes)

        anomaly = self._anomaly_detector.detect(tid)

        return {
            "event_type": "vehicle_detected",
            "timestamp": round(time.time(), 3),
            "camera_id": self._cfg.camera.source_id,
            "frame_id": frame_id,

            "vehicle": {
                "id": tid,
                "class": CLASS_LABELS.get(cls_id, "Vehicle"),
                "class_id": cls_id,
                "is_heavy": cls_id in HEAVY_CLASSES,
                "bbox": list(box),
                "lane": lane,
            },

            "kinematics": {
                "speed_px_per_sec": speed,
                "is_slow": 0 < speed < 35.0,
                "is_stopped": stopped,
                "direction": direction,
            },

            "residence": residence,

            "anomaly": {
                "is_anomaly": anomaly.is_anomaly,
                "type": anomaly.type,
                "stop_seconds": anomaly.stop_seconds,
            },
        }

    # ── SNAPSHOT EVENT ───────────────────────────────────────────────────────

    def _build_snapshot(self, frame_id: int,
                        active_tracks: List[Dict],
                        frame_height: int) -> Dict:
        if not active_tracks:
            return {
                "event_type": "traffic_snapshot",
                "timestamp": round(time.time(), 3),
                "camera_id": self._cfg.camera.source_id,
                "frame_id": frame_id,
                "counts": {"total": 0, "car": 0, "motorcycle": 0,
                           "bus": 0, "truck": 0, "heavy_vehicle_ratio": 0.0},
                "speed": {"average_px": 0.0, "min_px": 0.0, "max_px": 0.0},
                "density": {"status": "FREE", "occupancy_ratio": 0.0},
                "lane_counts": {lane.name: 0 for lane in self._cfg.lanes},
                "anomalies": [],
            }

        class_counts = {2: 0, 3: 0, 5: 0, 7: 0}
        lane_counts = {lane.name: 0 for lane in self._cfg.lanes}
        boxes = []
        speeds = []
        anomalies = []

        for track in active_tracks:
            tid = track["tid"]
            cls_id = track["cls_id"]
            box = track["box"]
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            boxes.append(box)

            spd = self._vm.get_speed(tid)
            speeds.append(spd)

            lane = TrafficMetrics.get_lane(cx, cy, self._cfg.lanes)
            if lane and lane in lane_counts:
                lane_counts[lane] += 1

            # Anomaly collection (deduplicated by priority)
            if self._vm.is_stopped(tid):
                anomalies.append({
                    "vehicle_id": tid,
                    "class": CLASS_LABELS.get(cls_id, "Vehicle"),
                    "type": "stopped_vehicle",
                    "stop_seconds": self._vm.get_stop_duration(tid),
                    "lane": lane,
                })
            elif self._vm.is_sudden_slowdown(tid):
                anomalies.append({
                    "vehicle_id": tid,
                    "class": CLASS_LABELS.get(cls_id, "Vehicle"),
                    "type": "sudden_slowdown",
                    "speed_px": spd,
                    "lane": lane,
                })

        total = len(active_tracks)
        heavy = class_counts[5] + class_counts[7]
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        occupancy = TrafficMetrics.occupancy_ratio(boxes, self._roi_area)
        status = TrafficMetrics.traffic_status(avg_speed, total, occupancy)

        return {
            "event_type": "traffic_snapshot",
            "timestamp": round(time.time(), 3),
            "camera_id": self._cfg.camera.source_id,
            "frame_id": frame_id,

            "counts": {
                "total": total,
                "car": class_counts[2],
                "motorcycle": class_counts[3],
                "bus": class_counts[5],
                "truck": class_counts[7],
                "heavy_vehicle_ratio": round(heavy / total, 3),
            },

            "speed": {
                "average_px": round(avg_speed, 2),
                "min_px": round(min(speeds), 2),
                "max_px": round(max(speeds), 2),
            },

            "density": {
                "status": status,
                "occupancy_ratio": occupancy,
            },

            "lane_counts": lane_counts,
            "anomalies": anomalies,
        }

