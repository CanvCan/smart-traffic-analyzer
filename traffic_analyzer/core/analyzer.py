import cv2
import supervision as sv
from traffic_analyzer.core.detector import BaseDetector
from traffic_analyzer.core.tracker import BaseTracker
from traffic_analyzer.utils.config_loader import AppConfig
from traffic_analyzer.visualization.renderer import Renderer, GHOST_FRAMES
from traffic_analyzer.core.event_builder import EventBuilder
from traffic_analyzer.core.ghost_track_manager import GhostTrackManager
from kafka_layer.kafka_producer import TrafficProducer


class Analyzer:
    def __init__(self, config: AppConfig,
                 detector: BaseDetector,
                 tracker: BaseTracker):
        self._cfg = config
        self._detector = detector
        self._tracker = tracker
        self._renderer = Renderer()
        self._event_builder = EventBuilder(config)
        self._producer = TrafficProducer()
        self._frame_count = 0
        self._paused = False
        self._ghost_manager = GhostTrackManager(ghost_frames=GHOST_FRAMES)

    def run(self) -> None:
        cap = cv2.VideoCapture(self._cfg.camera.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {self._cfg.camera.video_path}")
            return

        print(f"[Analyzer] Processing: {self._cfg.camera.video_path}")
        print("[Analyzer] q -> quit  |  space -> pause/resume")

        dw = self._cfg.camera.display_width

        try:
            while True:
                if not self._paused:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    self._frame_count += 1
                    try:
                        self._process_frame(frame)
                    except Exception as e:
                        print(f"[ERROR] Frame {self._frame_count}: {e}")
                        import traceback
                        traceback.print_exc()

                    h, w = frame.shape[:2]
                    cv2.imshow("Smart Traffic Analyzer",
                               cv2.resize(frame, (dw, int(dw * h / w))))

                # Use longer wait when paused to avoid CPU spin
                wait_ms = 100 if self._paused else 1
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self._paused = not self._paused
                    print(f"[Analyzer] {'PAUSED' if self._paused else 'RESUMED'}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._producer.close()
            print("[Analyzer] Shutdown complete.")

    def _process_frame(self, frame) -> None:
        frame_h = frame.shape[0]

        results = self._detector.detect(frame)
        detections = sv.Detections.from_ultralytics(results)
        detections = self._tracker.update(detections, frame)

        active_tracks = []
        active_ids = set()

        for i in range(len(detections)):
            tid = int(detections.tracker_id[i])
            cls_id = int(detections.class_id[i])
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            box = (x1, y1, x2, y2)

            active_ids.add(tid)
            # Single source of truth: VehicleMetrics inside EventBuilder
            self._event_builder.vm.update(tid, box, self._frame_count, frame_h)
            self._ghost_manager.update(tid, box, cls_id, self._frame_count)
            active_tracks.append({"tid": tid, "cls_id": cls_id, "box": box})

        # Produce events
        events = self._event_builder.update(
            self._frame_count, frame_h, active_tracks
        )
        for event in events:
            self._producer.send(event)

        # Visualisation
        self._renderer.draw_lanes(frame, self._cfg.lanes)

        for track in active_tracks:
            self._renderer.draw_vehicle(frame, *track["box"],
                                        track["tid"], track["cls_id"])

        # Ghost rendering for recently lost tracks
        for ghost in self._ghost_manager.get_ghosts(self._frame_count, active_ids):
            self._renderer.draw_vehicle(frame, *ghost["box"],
                                        ghost["tid"], ghost["cls_id"], ghost=True)

        self._renderer.draw_legend(frame)

        # Cleanup stale ghost state
        self._ghost_manager.cleanup(self._frame_count)
