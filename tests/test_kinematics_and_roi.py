import tempfile
import unittest
from pathlib import Path

from traffic_analyzer.infrastructure.config_loader import (
    compute_video_source_id,
    load_camera_lanes,
    resolve_local_video_source_id,
    save_camera_lanes,
)
from traffic_analyzer.services.vehicle_metrics import VehicleMetrics


class TestKinematicsAndROI(unittest.TestCase):
    def test_speed_invariance_under_inference_stall(self) -> None:
        """Speed depends only on injected frame timestamps, not wall-clock delay."""
        vm_fast = VehicleMetrics()
        vm_slow = VehicleMetrics()

        for frame_id in range(1, 15):
            ts = frame_id / 30.0
            box = (100 + frame_id * 6, 200, 150 + frame_id * 6, 260)
            vm_fast.update(1, box, frame_id=frame_id, frame_height=720, cls_id=1, frame_timestamp=ts)
            vm_slow.update(1, box, frame_id=frame_id, frame_height=720, cls_id=1, frame_timestamp=ts)

        self.assertAlmostEqual(vm_fast.get_speed(1), vm_slow.get_speed(1), places=5)
        self.assertGreater(vm_fast.get_speed(1), 0.0)

    def test_physical_time_direction_regression(self) -> None:
        """Direction regression uses physical timestamps (px/s) to estimate travel direction."""
        vm = VehicleMetrics()
        frames = [1, 2, 5, 8, 12]
        for fid in frames:
            ts = fid * 0.04
            x = int(100 + 60.0 * ts)
            vm.update(7, (x, 200, x + 50, 250), frame_id=fid, frame_height=720, cls_id=1, frame_timestamp=ts)

        self.assertEqual(vm.get_direction(7), "left_to_right")

    def test_stopped_detection_frame_rate_independent(self) -> None:
        """Stopped detection requires both minimum observations and physical elapsed duration."""
        vm = VehicleMetrics()
        for i in range(10):
            vm.update(3, (300, 300, 350, 360), frame_id=i + 1, frame_height=720, cls_id=1, frame_timestamp=i * 0.04)
        self.assertFalse(vm.is_stopped(3))

        for i in range(10, 30):
            vm.update(3, (300, 300, 350, 360), frame_id=i + 1, frame_height=720, cls_id=1, frame_timestamp=i * 0.05)
        self.assertTrue(vm.is_stopped(3))
        self.assertGreater(vm.get_stop_duration(3), 0.0)

    def test_video_source_id_isolation_and_scaling(self) -> None:
        """Distinct video files receive distinct fingerprinted IDs and scale coordinates cleanly."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            vid1 = tmp_dir / "test_video1.mp4"
            vid2 = tmp_dir / "test_video2.mp4"
            vid1.write_bytes(b"A" * 2048)
            vid2.write_bytes(b"B" * 4096)

            id1 = compute_video_source_id(vid1)
            id2 = compute_video_source_id(vid2)
            self.assertNotEqual(id1, id2)
            self.assertTrue(id1.startswith("test_video1_"))
            self.assertTrue(id2.startswith("test_video2_"))

            cameras_dir = tmp_dir / "cameras"
            save_camera_lanes(
                id1,
                cameras_dir,
                {"SOL_SERIT": {"roi": [100, 200, 400, 600], "points": [[100, 200], [400, 200], [400, 600]]}},
                resolution=[1920, 1080],
                display_name="Test Video 1",
            )

            lanes_v2 = load_camera_lanes(id2, cameras_dir)
            self.assertEqual(lanes_v2, [])

            lanes_v1_scaled = load_camera_lanes(id1, cameras_dir, target_resolution=(960, 540))
            self.assertEqual(len(lanes_v1_scaled), 1)
            self.assertEqual(lanes_v1_scaled[0].roi, [50, 100, 200, 300])

    def test_dynamic_local_video_source_id_and_isolated_json(self) -> None:
        """Local video files map to isolated 'local_<stem>' IDs without 'local.json' fallback."""
        with tempfile.TemporaryDirectory() as tmp:
            cameras_dir = Path(tmp) / "cameras"
            cameras_dir.mkdir(parents=True, exist_ok=True)

            self.assertEqual(
                resolve_local_video_source_id("C:/videos/test_video1.mp4", cameras_dir),
                "local_test_video1",
            )
            self.assertEqual(
                resolve_local_video_source_id("C:/videos/test_video4.mp4", cameras_dir),
                "local_test_video4",
            )

            save_camera_lanes(
                "local_test_video1",
                cameras_dir,
                {"Lane_A": {"points": [[10, 10], [100, 10], [100, 100]]}},
                resolution=[1280, 720],
            )
            self.assertTrue((cameras_dir / "local_test_video1.json").exists())
            self.assertEqual(len(load_camera_lanes("local_test_video1", cameras_dir)), 1)
            self.assertEqual(load_camera_lanes("local_test_video2", cameras_dir), [])


if __name__ == "__main__":
    unittest.main()
