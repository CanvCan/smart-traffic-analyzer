"""
traffic_analyzer/main.py

Composition Root — the single place where all dependencies are created and injected.

Clean Architecture rule: the answer to "which interface maps to which implementation?"
lives only here.

Layer responsibilities:
  Infrastructure  : YOLODetector, ByteTracker, TrafficProducer  (external world)
  Application     : EventBuilder, Renderer, GhostTrackManager   (orchestration)
  Composition     : FrameProcessor, VideoLoop, Analyzer         (wiring)
"""

import torch

from traffic_analyzer.utils.config_loader import load_config
from traffic_analyzer.core.detector import YOLODetector
from traffic_analyzer.core.tracker import ByteTracker
from traffic_analyzer.core.event_builder import EventBuilder
from traffic_analyzer.core.frame_processor import FrameProcessor
from traffic_analyzer.core.video_loop import VideoLoop
from traffic_analyzer.core.analyzer import Analyzer
from traffic_analyzer.core.ghost_track_manager import GhostTrackManager
from traffic_analyzer.visualization.renderer import Renderer, GHOST_FRAMES
from kafka_layer.kafka_producer import TrafficProducer


def main():
    # ── 1. Configuration ──────────────────────────────────────────────────────
    cfg    = load_config('config.json')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 2. Infrastructure components ──────────────────────────────────────────
    #    Adapters that talk to the outside world (YOLO model, ByteTrack, Kafka).
    detector = YOLODetector(
        model_path=cfg.model.yolo_path,
        device=device,
        conf=cfg.model.conf_threshold,
        imgsz=cfg.model.imgsz,
        half=cfg.model.half,
        iou=cfg.model.iou,
        agnostic_nms=cfg.model.agnostic_nms,
        target_classes=[2, 3, 5, 7],
    )
    tracker   = ByteTracker(
        lost_track_buffer=150,
        min_matching_threshold=0.8,
        min_consecutive_frames=3,
    )
    publisher = TrafficProducer()   # IEventPublisher implementation

    # ── 3. Application components ─────────────────────────────────────────────
    #    Objects that orchestrate business logic and visualisation.
    event_builder = EventBuilder(cfg)
    renderer      = Renderer()
    ghost_manager = GhostTrackManager(ghost_frames=GHOST_FRAMES)

    # ── 4. Composition ────────────────────────────────────────────────────────
    #    Components are wired together via constructor injection.
    #    No class creates its own dependencies internally.
    frame_processor = FrameProcessor(
        detector=detector,
        tracker=tracker,
        event_builder=event_builder,
        renderer=renderer,
        ghost_manager=ghost_manager,
        config=cfg,
    )
    video_loop = VideoLoop(
        video_path=cfg.camera.video_path,
        display_width=cfg.camera.display_width,
    )
    analyzer = Analyzer(
        video_loop=video_loop,
        frame_processor=frame_processor,
        publisher=publisher,
    )

    # ── 5. Run ────────────────────────────────────────────────────────────────
    analyzer.run()


if __name__ == "__main__":
    main()
