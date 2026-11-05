"""
application/frame_processor.py

Processes video frames end-to-end (RF-DETR detection -> ByteTrack -> EventBuilder)
and produces domain events + immutable OverlaySnapshot for rendering.
"""

import time
from typing import List, Optional, Tuple

from traffic_analyzer.application.event_builder import EventBuilder
from traffic_analyzer.domain.models import AppConfig, FramePacket, OverlaySnapshot, TrackOverlay
from traffic_analyzer.domain.ports import IDetector, ITracker
from traffic_analyzer.services.ghost_track_manager import GhostTrackManager


class FrameProcessor:
    """
    Processes a single video frame end-to-end without performing any UI drawing.
    Produces domain events and an immutable OverlaySnapshot.
    """

    def __init__(
        self,
        detector: IDetector,
        tracker: ITracker,
        event_builder: EventBuilder,
        ghost_manager: GhostTrackManager,
        config: AppConfig,
    ):
        self._detector = detector
        self._tracker = tracker
        self._event_builder = event_builder
        self._ghost_manager = ghost_manager
        self._cfg = config

    @property
    def config(self) -> AppConfig:
        return self._cfg

    def process_packet(
        self, packet: FramePacket
    ) -> Tuple[List[dict], OverlaySnapshot, float]:
        """
        Execute detector, tracker, and event builder sequentially for `packet`,
        returning `(events, overlay_snapshot, inference_fps)`.
        """
        t0 = time.perf_counter()
        frame = packet.frame
        frame_id = packet.frame_id
        frame_ts = packet.timestamp
        frame_h = frame.shape[0]

        results = self._detector.detect(frame)
        detections = self._tracker.update(results, frame)

        active_tracks = []
        active_ids = set()

        for i in range(len(detections)):
            tid = int(detections.tracker_id[i])
            cls_id = int(detections.class_id[i])
            if cls_id == -1:
                continue
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            box = (x1, y1, x2, y2)
            conf_arr = getattr(detections, "confidence", None)
            conf = float(conf_arr[i]) if conf_arr is not None and len(conf_arr) > i else 1.0

            active_ids.add(tid)
            self._ghost_manager.update(tid, box, cls_id, frame_id)
            active_tracks.append({"tid": tid, "cls_id": cls_id, "box": box, "confidence": conf})

        events = self._event_builder.update(
            frame_id,
            frame_h,
            active_tracks,
            frame_timestamp=frame_ts,
        )

        speed_by_tid = {}
        anomaly_by_tid = {}
        stable_cls_by_tid = {}
        for ev in events:
            if ev.get("event_type") in ("vehicle_detected", "vehicle_update"):
                veh = ev.get("vehicle", {})
                tid = int(veh.get("id", ev.get("track_id", -1)))
                if "class_id" in veh:
                    stable_cls_by_tid[tid] = int(veh["class_id"])
                speed_by_tid[tid] = float(ev.get("kinematics", {}).get("speed_px_per_sec", 0.0))
                anom = ev.get("anomaly", {})
                anomaly_by_tid[tid] = anom.get("type") if anom.get("is_anomaly") else None

        overlays: List[TrackOverlay] = []
        for track in active_tracks:
            tid = track["tid"]
            x1, y1, x2, y2 = track["box"]
            overlays.append(
                TrackOverlay(
                    track_id=tid,
                    class_id=stable_cls_by_tid.get(tid, track["cls_id"]),
                    bbox=(x1, y1, x2, y2),
                    speed_px=speed_by_tid.get(tid, 0.0),
                    anomaly_type=anomaly_by_tid.get(tid),
                    is_ghost=False,
                )
            )

        for ghost in self._ghost_manager.get_ghosts(frame_id, active_ids):
            gx1, gy1, gx2, gy2 = ghost["box"]
            overlays.append(
                TrackOverlay(
                    track_id=int(ghost["tid"]),
                    class_id=int(ghost["cls_id"]),
                    bbox=(int(gx1), int(gy1), int(gx2), int(gy2)),
                    speed_px=0.0,
                    anomaly_type=None,
                    is_ghost=True,
                )
            )

        self._ghost_manager.cleanup(frame_id)
        elapsed = max(1e-4, time.perf_counter() - t0)
        inf_fps = 1.0 / elapsed

        snapshot = OverlaySnapshot(
            frame_id=frame_id,
            timestamp=frame_ts,
            tracks=tuple(overlays),
        )
        return events, snapshot, inf_fps

    def process(
        self, frame, frame_id: int, frame_timestamp: Optional[float] = None
    ) -> Tuple[List[dict], OverlaySnapshot]:
        """
        Process a single frame and return `(events, overlay_snapshot)` without drawing.
        """
        ts = float(frame_timestamp) if frame_timestamp is not None else (float(frame_id) / 25.0)
        packet = FramePacket(
            frame_id=frame_id,
            timestamp=ts,
            frame=frame,
            is_live=False,
        )
        events, snapshot, _ = self.process_packet(packet)
        return events, snapshot
