"""
core/analyzer.py

Top-level orchestrator for the traffic analysis pipeline.

Wires FrameProcessor, VideoLoop, and IEventPublisher together following the
Dependency Inversion Principle — each dependency is injected, never constructed here.

Responsibilities:
  - Start and stop the video loop
  - Forward each frame to FrameProcessor
  - Publish resulting domain events via IEventPublisher
  - Guarantee clean shutdown (flush / close) on exit

All business logic lives in the injected components:
  FrameProcessor  — detection, tracking, metrics, event building, rendering
  VideoLoop       — video I/O, display, keyboard handling
  IEventPublisher — event transport (Kafka, console, or any adapter)
"""

from traffic_analyzer.core.frame_processor import FrameProcessor
from traffic_analyzer.core.video_loop import VideoLoop
from traffic_analyzer.domain.ports import IEventPublisher


class Analyzer:
    """
    Top-level orchestrator for the traffic analysis pipeline.

    Inject all dependencies from main.py (Composition Root).
    """

    def __init__(self,
                 video_loop: VideoLoop,
                 frame_processor: FrameProcessor,
                 publisher: IEventPublisher):
        self._loop      = video_loop
        self._processor = frame_processor
        self._publisher = publisher

    def run(self) -> None:
        """Start the pipeline. Blocks until the video ends or user quits."""
        try:
            self._loop.run(
                frame_callback=self._processor.process,
                on_events=self._publish_all,
            )
        finally:
            self._publisher.close()
            print("[Analyzer] Shutdown complete.")

    def _publish_all(self, events: list) -> None:
        for event in events:
            self._publisher.send(event)
