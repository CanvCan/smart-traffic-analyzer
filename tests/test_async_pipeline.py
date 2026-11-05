import unittest
import numpy as np

from traffic_analyzer.adapters.detector import BaseDetector
from traffic_analyzer.adapters.tracker import BaseTracker
from traffic_analyzer.application.event_builder import EventBuilder
from traffic_analyzer.application.frame_processor import FrameProcessor
from traffic_analyzer.domain.models import (
    AppConfig,
    CameraConfig,
    FramePacket,
    LaneConfig,
    ModelConfig,
)
from traffic_analyzer.infrastructure.video_loop import AtomicLatestSlot, VideoLoop
from traffic_analyzer.services.ghost_track_manager import GhostTrackManager
from traffic_analyzer.visualization.frame_renderer import FrameRenderer, _alpha_rect


class _DummyDetections:
    def __init__(self) -> None:
        self.tracker_id = np.array([101], dtype=np.int64)
        self.class_id = np.array([1], dtype=np.int64)
        self.xyxy = np.array([[120, 140, 220, 240]], dtype=np.int64)

    def __len__(self) -> int:
        return 1


class _DummyDetector(BaseDetector):
    def detect(self, frame):
        return _DummyDetections()


class _DummyTracker(BaseTracker):
    def update(self, detections, frame=None):
        return detections


class TestAsyncPipeline(unittest.TestCase):
    def test_atomic_latest_slot_overwrites_stale_frames(self) -> None:
        """AtomicLatestSlot retains only the latest frame in O(1) time."""
        slot = AtomicLatestSlot()
        for i in range(1, 6):
            slot.put(FramePacket(frame_id=i, timestamp=i * 0.04, frame=None, is_live=True))

        popped = slot.pop_latest()
        self.assertIsNotNone(popped)
        self.assertEqual(popped.frame_id, 5)
        self.assertIsNone(slot.pop_latest())

    def test_frame_processor_packet_produces_immutable_snapshot(self) -> None:
        """process_packet produces events, OverlaySnapshot, and inference FPS without mutating frame."""
        cfg = AppConfig(
            camera=CameraConfig(source_id="TEST_CAM", video_path="test.mp4", display_width=960),
            model=ModelConfig(model_path="best.pt", threshold=0.3, resolution=880, device="cpu"),
            lanes=[
                LaneConfig(
                    name="LANE_1",
                    roi=[50, 50, 400, 400],
                    points=[[50, 50], [400, 50], [400, 400], [50, 400]],
                )
            ],
        )
        renderer = FrameRenderer()
        processor = FrameProcessor(
            detector=_DummyDetector(),
            tracker=_DummyTracker(),
            event_builder=EventBuilder(cfg),
            ghost_manager=GhostTrackManager(ghost_frames=15),
            config=cfg,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        events = []
        snapshot = None
        inf_fps = 0.0
        for fid in range(1, 7):
            pkt = FramePacket(frame_id=fid, timestamp=fid * 0.04, frame=frame, is_live=False)
            events, snapshot, inf_fps = processor.process_packet(pkt)

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.frame_id, 6)
        self.assertEqual(len(snapshot.tracks), 1)
        self.assertEqual(snapshot.tracks[0].track_id, 101)
        self.assertGreater(inf_fps, 0.0)
        self.assertTrue(any(e["event_type"] == "vehicle_detected" for e in events))

        # FrameProcessor must not draw on the input frame
        self.assertEqual(int(frame.sum()), 0)

        # Render snapshot onto canvas using FrameRenderer
        canvas = frame.copy()
        renderer.draw_snapshot(canvas, cfg.lanes, snapshot)
        self.assertGreater(int(canvas.sum()), 0)

    def test_sub_roi_alpha_rect_clips_safely(self) -> None:
        """_alpha_rect blends only the clipped sub-region without mutating outside pixels."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _alpha_rect(img, 10, 10, 30, 30, (0, 200, 100), 0.5)
        self.assertGreater(int(img[15, 15].sum()), 0)
        self.assertEqual(int(img[5, 5].sum()), 0)

    def test_live_vs_file_hud_and_pause_guard(self) -> None:
        """Live streams show '🔴 LIVE | Q: Quit' and disable Space pause; local files allow pause."""
        renderer = FrameRenderer()
        live_hud = renderer.format_hud_text(fps=25.0, vehicle_count=4, is_live=True)
        self.assertIn("🔴 LIVE | Q: Quit", live_hud)
        self.assertNotIn("Space: Pause", live_hud)

        file_hud = renderer.format_hud_text(fps=25.0, vehicle_count=4, is_live=False)
        self.assertIn("Space: Pause | Q: Quit", file_hud)

        # Space key is ignored for live streams, toggles pause for local video files
        self.assertIsNone(VideoLoop._key_to_action(ord(" "), is_live=True))
        self.assertEqual(VideoLoop._key_to_action(ord(" "), is_live=False), "toggle_pause")

        live_loop = VideoLoop("https://example.com/cam.mjpeg", display_width=960, is_live=True)
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        _, paused_live, _ = live_loop._handle_action("toggle_pause", dummy, is_live=True, paused=False)
        self.assertFalse(paused_live)

        file_loop = VideoLoop("test_video.mp4", display_width=960, is_live=False)
        _, paused_file, _ = file_loop._handle_action("toggle_pause", dummy, is_live=False, paused=False)
        self.assertTrue(paused_file)


if __name__ == "__main__":
    unittest.main()
