"""
core/frame_processor.py

Processes a single video frame end-to-end and returns the domain events produced.

Pipeline:
  1. Detect objects (Detector)
  2. Assign and maintain track IDs (Tracker)
  3. Update per-vehicle kinematics (VehicleMetrics via EventBuilder)
  4. Build structured domain events (EventBuilder)
  5. Render visual overlays in-place — lanes, vehicles, ghosts, legend (Renderer)

Part of a three-component pipeline:
  FrameProcessor  — single-frame processing
  VideoLoop       — video I/O and display
  Analyzer        — top-level orchestration
"""

import supervision as sv

from traffic_analyzer.core.detector import BaseDetector
from traffic_analyzer.core.tracker import BaseTracker
from traffic_analyzer.core.event_builder import EventBuilder
from traffic_analyzer.core.ghost_track_manager import GhostTrackManager
from traffic_analyzer.visualization.renderer import Renderer
from traffic_analyzer.utils.config_loader import AppConfig


class FrameProcessor:
    """
    Processes a single video frame end-to-end.

    Returns the list of domain events produced for that frame.
    The frame numpy array is modified in-place with rendered overlays.
    """

    def __init__(self,
                 detector: BaseDetector,
                 tracker: BaseTracker,
                 event_builder: EventBuilder,
                 renderer: Renderer,
                 ghost_manager: GhostTrackManager,
                 config: AppConfig):
        self._detector      = detector
        self._tracker       = tracker
        self._event_builder = event_builder
        self._renderer      = renderer
        self._ghost_manager = ghost_manager
        self._cfg           = config

    def process(self, frame, frame_id: int) -> list:
        """
        Detect → Track → Metrics → Events → Render.

        Args:
            frame:    BGR numpy array (modified in-place for rendering).
            frame_id: Monotonically increasing frame counter.

        Returns:
            List of event dicts ready to be sent to IEventPublisher.
        """
        frame_h = frame.shape[0]

        results    = self._detector.detect(frame)
        detections = sv.Detections.from_ultralytics(results)
        detections = self._tracker.update(detections, frame)

        active_tracks = []
        active_ids    = set()

        for i in range(len(detections)):
            tid    = int(detections.tracker_id[i])
            cls_id = int(detections.class_id[i])
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            box = (x1, y1, x2, y2)

            active_ids.add(tid)
            # Single source of truth: VehicleMetrics lives inside EventBuilder.
            self._event_builder.vm.update(tid, box, frame_id, frame_h, cls_id)
            self._ghost_manager.update(tid, box, cls_id, frame_id)
            active_tracks.append({"tid": tid, "cls_id": cls_id, "box": box})

        # Produce domain events — no publishing here, caller decides transport.
        events = self._event_builder.update(frame_id, frame_h, active_tracks)

        # ── Render overlays (in-place) ────────────────────────────────────
        self._renderer.draw_lanes(frame, self._cfg.lanes)

        for track in active_tracks:
            self._renderer.draw_vehicle(frame, *track["box"],
                                        track["tid"], track["cls_id"])

        for ghost in self._ghost_manager.get_ghosts(frame_id, active_ids):
            self._renderer.draw_vehicle(frame, *ghost["box"],
                                        ghost["tid"], ghost["cls_id"], ghost=True)

        self._renderer.draw_legend(frame)
        self._ghost_manager.cleanup(frame_id)

        return events
