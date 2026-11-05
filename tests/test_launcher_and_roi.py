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


if __name__ == "__main__":
    unittest.main()
