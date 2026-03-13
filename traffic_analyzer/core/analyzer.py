import cv2
import supervision as sv
from traffic_analyzer.core.detector import BaseDetector
from traffic_analyzer.core.tracker import BaseTracker
from traffic_analyzer.utils.memory import TrackMemory
from traffic_analyzer.utils.config_loader import AppConfig
from traffic_analyzer.visualization.renderer import Renderer, GHOST_FRAMES
from traffic_analyzer.core.event_builder import EventBuilder


class Analyzer:
    def __init__(self, config: AppConfig,
                 detector: BaseDetector,
                 tracker: BaseTracker):
        self._cfg           = config
        self._detector      = detector
        self._tracker       = tracker
        self._memory        = TrackMemory(history_size=15, max_lost_frames=150)
        self._renderer      = Renderer()
        self._event_builder = EventBuilder(config)
        self._frame_count   = 0
        self._paused        = False

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
                        import traceback; traceback.print_exc()

                    h, w = frame.shape[:2]
                    cv2.imshow("Smart Traffic Analyzer",
                               cv2.resize(frame, (dw, int(dw * h / w))))

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self._paused = not self._paused
                    print(f"[Analyzer] {'PAUSED' if self._paused else 'RESUMED'}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[Analyzer] Shutdown complete.")

    def _process_frame(self, frame) -> None:
        frame_h = frame.shape[0]

        results    = self._detector.detect(frame)
        detections = sv.Detections.from_ultralytics(results)
        detections = self._tracker.update(detections, frame)

        active_tracks = []
        active_ids    = set()

        for i in range(len(detections)):
            tid             = int(detections.tracker_id[i])
            cls_id          = int(detections.class_id[i])
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            box             = (x1, y1, x2, y2)

            active_ids.add(tid)
            self._memory.update(tid, cls_id, box, self._frame_count)
            active_tracks.append({"tid": tid, "cls_id": cls_id, "box": box})

        # Produce events (console until Kafka is wired)
        events = self._event_builder.update(
            self._frame_count, frame_h, active_tracks
        )
        for event in events:
            if event["event_type"] == "traffic_snapshot":
                print(f"[SNAPSHOT] {self._event_builder.to_json(event)}")

        # Visualisation
        self._renderer.draw_lanes(frame, self._cfg.lanes)

        for tid in active_ids:
            box = self._memory.get_box(tid)
            cls = self._memory.get_cls(tid)
            if box:
                self._renderer.draw_vehicle(frame, *box, tid, cls)

        for tid in self._memory.all_ids():
            lost = self._memory.get_lost_frames(tid, self._frame_count)
            if 0 < lost <= GHOST_FRAMES:
                box = self._memory.get_box(tid)
                cls = self._memory.get_cls(tid)
                if box:
                    self._renderer.draw_vehicle(frame, *box, tid, cls or 2, ghost=True)

        self._renderer.draw_legend(frame)
        self._memory.collect_garbage(self._frame_count)
