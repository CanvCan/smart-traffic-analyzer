import inspect
import tempfile
import unittest
from pathlib import Path

from launcher.tabs.settings_tab import SETTINGS_METADATA
from launcher.validation import (
    DISPLAY_WIDTH_BOUNDS,
    snap_resolution_to_block40,
)
from traffic_analyzer.adapters.detector import BaseDetector
from traffic_analyzer.adapters.influx_publisher import InfluxPublisher
from traffic_analyzer.adapters.kafka_producer import TrafficProducer
from traffic_analyzer.adapters.tracker import BaseTracker
from traffic_analyzer.domain.models import (
    AppConfig,
    CameraConfig,
    LanePolygon,
    ModelConfig,
)
from traffic_analyzer.domain.ports import (
    CompositePublisher,
    IDetector,
    IEventPublisher,
    ITracker,
    NullPublisher,
)
from traffic_analyzer.domain.text import normalize_search_key, slugify_ascii
from traffic_analyzer.infrastructure.config_loader import load_config, save_config
from traffic_analyzer.infrastructure.docker_logs import (
    DockerLogStreamer,
    resolve_compose_container_name,
)
from traffic_analyzer.services.scene_metrics import TrafficMetrics
from traffic_analyzer.visualization.roi_renderer import ROIRenderer
from traffic_analyzer.visualization.roi_selector import ROISelectorWindow, run as roi_run
from launcher.app import SmartTrafficLauncherApp
from launcher.ipc_runner import GuiEventQueuePublisher
from launcher.tabs.logs_tab import format_telemetry_event


class TestLauncherAndROI(unittest.TestCase):
    def test_turkish_slugify_and_search_normalization(self) -> None:
        """Turkish characters transliterate to strict ASCII IDs and normalized search keys."""
        self.assertEqual(slugify_ascii("ÜÇYOL KAVŞAĞI"), "UCYOL_KAVSAGI")
        self.assertEqual(slugify_ascii("BORNOVA HÜKÜMET KONAĞI"), "BORNOVA_HUKUMET_KONAGI")
        self.assertEqual(slugify_ascii("İSLAM ENSTİTÜSÜ"), "ISLAM_ENSTITUSU")
        self.assertEqual(normalize_search_key("İzmir Üçyol"), "izmir ucyol")

    def test_resolution_block40_snapping_and_display_bounds(self) -> None:
        """Resolution snaps to multiples of 40 and display_width bounds match expected limits."""
        self.assertEqual(DISPLAY_WIDTH_BOUNDS, (480, 3840))
        snapped, modified = snap_resolution_to_block40(900)
        self.assertEqual(snapped % 40, 0)
        self.assertTrue(modified)

    def test_hexagonal_ports_and_domain_lane_polygon(self) -> None:
        """Adapters implement domain ports and abstract interfaces enforce method contracts."""
        self.assertTrue(inspect.isabstract(IDetector))
        self.assertTrue(inspect.isabstract(ITracker))
        self.assertTrue(inspect.isabstract(IEventPublisher))

        self.assertTrue(issubclass(BaseDetector, IDetector))
        self.assertTrue(issubclass(BaseTracker, ITracker))
        self.assertTrue(issubclass(TrafficProducer, IEventPublisher))
        self.assertTrue(issubclass(InfluxPublisher, IEventPublisher))
        self.assertTrue(issubclass(CompositePublisher, IEventPublisher))
        self.assertTrue(issubclass(NullPublisher, IEventPublisher))

        with self.assertRaises(TypeError):
            IDetector()  # type: ignore[abstract]
        with self.assertRaises(TypeError):
            ITracker()  # type: ignore[abstract]
        with self.assertRaises(TypeError):
            IEventPublisher()  # type: ignore[abstract]

        lane = LanePolygon(
            name="SOL_SERIT",
            roi=(0, 0, 100, 100),
            points=((0, 0), (100, 0), (100, 100), (0, 100)),
            display_name="SOL ŞERİT",
        )
        self.assertEqual(TrafficMetrics.get_lane(50, 50, [lane]), "SOL_SERIT")
        self.assertAlmostEqual(TrafficMetrics.roi_total_area([lane]), 10000.0)

    def test_roi_renderer_lane_selection_and_vertex_hit_testing(self) -> None:
        """ROIRenderer accurately hit-tests saved lane polygons and vertices for in-place editing."""
        renderer = ROIRenderer(frame_w=640, frame_h=480)
        renderer.update_projection(canvas_w=640, canvas_h=480)
        lanes = [
            {
                "name": "Lane 1",
                "points": [[100, 100], [300, 100], [300, 300], [100, 300]],
                "expected_direction": "DOWN",
            },
            {
                "name": "Lane 2",
                "points": [[350, 100], [550, 100], [550, 300], [350, 300]],
                "expected_direction": "UP",
            },
        ]
        # Inside Lane 1
        self.assertEqual(renderer.find_containing_lane(200, 200, lanes), 0)
        # Inside Lane 2
        self.assertEqual(renderer.find_containing_lane(450, 200, lanes), 1)
        # Outside any lane
        self.assertIsNone(renderer.find_containing_lane(50, 50, lanes))

        # Vertex hit-test on Lane 2 vertex 0 (350, 100)
        self.assertEqual(renderer.find_lane_vertex(353, 98, lanes, preferred_lane_idx=1), (1, 0))
        self.assertIsNone(renderer.find_lane_vertex(200, 200, lanes))

    def test_settings_metadata_and_atomic_persistence(self) -> None:
        """All SETTINGS_METADATA keys map 1-to-1 to AppConfig dataclass attributes and persist atomically."""
        cfg = AppConfig(camera=CameraConfig(), model=ModelConfig())
        for key, _label, _desc, _val_type, _min_v, _max_v in SETTINGS_METADATA:
            sec_name, attr_name = key.split(".", 1)
            self.assertTrue(hasattr(cfg, sec_name), f"Missing section {sec_name}")
            sec_obj = getattr(cfg, sec_name)
            self.assertTrue(hasattr(sec_obj, attr_name), f"Missing field {key}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_cfg = Path(tmp_dir) / "config.json"
            cfg.model.resolution = 1040
            cfg.kinematics.ema_tau = 0.45
            cfg.scene_thresholds.occ_flow = 0.42
            save_config(cfg, tmp_cfg)

            reloaded = load_config(str(tmp_cfg))
            self.assertEqual(reloaded.model.resolution, 1040)
            self.assertAlmostEqual(reloaded.kinematics.ema_tau, 0.45)
            self.assertAlmostEqual(reloaded.scene_thresholds.occ_flow, 0.42)

    def test_docker_logs_container_resolution(self) -> None:
        """DockerLogStreamer resolves container_name from docker-compose.yml and includes fallback candidates."""
        compose_kafka = resolve_compose_container_name("kafka")
        self.assertEqual(compose_kafka, "kafka_layer")
        streamer = DockerLogStreamer(container_candidates=["kafka_layer", "kafka"], service_name="kafka")
        self.assertIn("kafka_layer", streamer.candidates)
        self.assertIn("kafka", streamer.candidates)

    def test_telemetry_event_formatting_and_gui_queue_publisher(self) -> None:
        """Telemetry events format cleanly into structured [EVENT]/[ANOMALY]/[SNAPSHOT] lines."""
        import queue

        ev_q: "queue.Queue[dict]" = queue.Queue()
        pub = GuiEventQueuePublisher(ev_q)  # type: ignore[arg-type]
        vehicle_ev = {
            "event_type": "vehicle_detected",
            "timestamp": "03:48:52.120",
            "confidence": 0.91,
            "vehicle": {
                "id": 42,
                "class": "Car",
                "lane": "RIGHT_LANE",
                "confidence": 0.91,
            },
            "kinematics": {
                "speed_px_per_sec": 74.36,
            },
            "anomaly": {
                "is_anomaly": True,
                "type": "wrong_way",
            },
        }
        pub.send(vehicle_ev)
        popped = ev_q.get_nowait()
        formatted = format_telemetry_event(popped)
        self.assertEqual(len(formatted), 2)
        self.assertEqual(
            formatted[0][0].strip(),
            "[03:48:52.120] [EVENT] ID: #42 | Class: Car | Lane: RIGHT_LANE | Speed: 74.4 px/s | Conf: 0.91",
        )
        self.assertEqual(formatted[0][1], "EVENT")
        self.assertIn("[ANOMALY] ID: #42 | Type: WRONG_WAY", formatted[1][0])
        self.assertEqual(formatted[1][1], "ANOMALY")

    def test_logs_tab_freeze_and_settings_reset_to_defaults(self) -> None:
        """LogsTab freeze state suppresses queue inserts and SettingsTab reset restores AppConfig factory defaults."""
        from launcher.tabs.logs_tab import LogsTab
        from launcher.tabs.settings_tab import SettingsTab
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            logs = LogsTab(root)
            self.assertIsInstance(logs.paned_window, tk.PanedWindow)
            self.assertEqual(logs.paned_window.cget("sashwidth"), 6)
            self.assertEqual(logs.paned_window.cget("sashcursor"), "sb_h_double_arrow")

            self.assertFalse(logs.is_stream_paused)
            logs.toggle_stream_pause()
            self.assertTrue(logs.is_stream_paused)
            self.assertEqual(logs._pause_btn.cget("text"), "▶ Resume")

            logs.push_telemetry_event({"event_type": "vehicle_detected", "vehicle": {"id": 1, "class": "Bus"}})
            logs._drain_telemetry_events()
            # Since stream is paused, the event is not added to textbox
            content = logs._kafka_log_box.get("1.0", "end")
            self.assertNotIn("Bus", content)

            logs.toggle_stream_pause()
            self.assertFalse(logs.is_stream_paused)
            self.assertEqual(logs._pause_btn.cget("text"), "⏸ Pause")

            logs.clear_kafka_console()
            self.assertEqual(logs._kafka_log_box.get("1.0", "end").strip(), "")

            # Spark console pause and clear
            self.assertFalse(logs.is_spark_paused)
            logs.toggle_spark_pause()
            self.assertTrue(logs.is_spark_paused)
            self.assertEqual(logs._spark_pause_btn.cget("text"), "▶ Resume")
            logs.toggle_spark_pause()
            self.assertFalse(logs.is_spark_paused)
            self.assertEqual(logs._spark_pause_btn.cget("text"), "⏸ Pause")

            logs.clear_spark_console()
            logs.stop_streamers()

            with tempfile.TemporaryDirectory() as tmp_dir:
                cfg_path = Path(tmp_dir) / "config.json"
                custom_cfg = AppConfig(
                    camera=CameraConfig(display_width=1280),
                    model=ModelConfig(resolution=640, threshold=0.6),
                )
                custom_cfg.tracking.lost_track_buffer = 999
                save_config(custom_cfg, cfg_path)

                settings = SettingsTab(
                    root,
                    project_root=Path(tmp_dir),
                    config_path=cfg_path,
                    on_status_message=lambda _msg, _col: None,
                )
                self.assertEqual(settings._field_vars["tracking.lost_track_buffer"].get(), "999")
                self.assertEqual(settings._field_vars["model.batch_size"].get(), "1")

                settings.reset_to_defaults()
                # Factory default for lost_track_buffer is 150, resolution is 880
                self.assertEqual(settings._field_vars["tracking.lost_track_buffer"].get(), "150")
                self.assertEqual(settings._custom_res_var.get(), "880")
                self.assertEqual(settings._res_choice_var.get(), "880 (Recommended)")

                # Save and reload from disk
                saved_ok = settings.save_settings_to_disk(quiet=True)
                self.assertTrue(saved_ok)
                reloaded = load_config(str(cfg_path))
                self.assertEqual(reloaded.tracking.lost_track_buffer, 150)
                self.assertEqual(reloaded.model.resolution, 880)
        finally:
            root.destroy()

    def test_roi_selector_canvas_image_lifecycle_and_toplevel(self) -> None:
        """ROISelectorWindow uses parent master, retains canvas.image reference, and updates itemconfig."""
        import tkinter as tk
        import numpy as np

        root = tk.Tk()
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                cfg_path = Path(tmp_dir) / "config.json"
                save_config(AppConfig(camera=CameraConfig(), model=ModelConfig()), cfg_path)
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

                editor = ROISelectorWindow(
                    parent=root,
                    frame=frame,
                    config_path=str(cfg_path),
                    source_id="test_cam",
                    cameras_dir=Path(tmp_dir),
                )
                self.assertFalse(editor._owns_root)
                self.assertIsNone(editor._hidden_root)

                # Ensure canvas has dimensions and render
                editor._canvas.configure(width=640, height=480)
                editor._canvas.update()
                editor._render_canvas()

                # Verify PhotoImage master and GC reference
                self.assertIsNotNone(editor._photo)
                self.assertIsNotNone(editor._canvas.image)
                self.assertEqual(editor._photo, editor._canvas.image)
                first_img_id = editor._image_id
                self.assertIsNotNone(first_img_id)

                # Simulate mouse movement and re-render
                fake_event = tk.Event()
                fake_event.x = 200
                fake_event.y = 150
                editor._on_mouse_move(fake_event)

                # _image_id should remain the exact same item on canvas (updated in place via itemconfig)
                self.assertEqual(editor._image_id, first_img_id)
                self.assertIsNotNone(editor._canvas.image)

                editor.destroy()
        finally:
            root.destroy()

    def test_roi_selector_timeline_slider_and_frame_seeking(self) -> None:
        """ROISelectorWindow initializes timeline slider for video, skips intro, and seeks without dropping lanes."""
        import tkinter as tk

        test_vid = Path("traffic_analyzer/test_video.mp4")
        if not test_vid.exists():
            return

        root = tk.Tk()
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                cfg_path = Path(tmp_dir) / "config.json"
                save_config(AppConfig(camera=CameraConfig(), model=ModelConfig()), cfg_path)

                editor = ROISelectorWindow(
                    parent=root,
                    config_path=str(cfg_path),
                    source_id="test_vid_cam",
                    cameras_dir=Path(tmp_dir),
                    video_path=str(test_vid),
                )
                self.assertIsNotNone(editor._cap)
                self.assertGreater(editor._total_frames, 100)
                # Intro skip: default frame should be ~3s * fps
                expected_start = int(editor._fps * 3)
                self.assertEqual(editor._current_frame_idx, expected_start)

                # Timeline UI presence
                self.assertTrue(hasattr(editor, "_timeline_slider"))
                self.assertEqual(editor._timeline_slider.get(), expected_start)
                self.assertIn("Time:", editor._timeline_time_lbl.cget("text"))

                # Add a dummy lane to verify it is NOT dropped on seek
                editor._lanes.append({
                    "name": "Lane 1",
                    "points": [[100, 100], [200, 100], [200, 200], [100, 200]],
                    "expected_direction": "DOWN",
                })

                # Seek forward to frame 300
                editor._seek_frame(300)
                self.assertEqual(editor._current_frame_idx, 300)
                self.assertEqual(len(editor._lanes), 1)
                self.assertEqual(editor._lanes[0]["name"], "Lane 1")

                # Relative seek by +1 second
                prev_idx = editor._current_frame_idx
                editor._seek_relative(int(editor._fps * 1))
                self.assertEqual(editor._current_frame_idx, prev_idx + int(editor._fps * 1))

                editor._close_window()
                self.assertIsNone(editor._cap)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
