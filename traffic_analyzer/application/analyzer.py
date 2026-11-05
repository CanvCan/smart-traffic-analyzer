"""
application/analyzer.py

Top-level orchestrator for the traffic analysis pipeline.
"""

import threading

from traffic_analyzer.application.frame_processor import FrameProcessor
from traffic_analyzer.infrastructure.video_loop import VideoLoop
from traffic_analyzer.domain.ports import IEventPublisher


class Analyzer:
    """Top-level orchestrator for the traffic analysis pipeline."""

    def __init__(self,
                 video_loop: VideoLoop,
                 frame_processor: FrameProcessor,
                 publisher: IEventPublisher):
        self._loop      = video_loop
        self._processor = frame_processor
        self._publisher = publisher

    def run(self) -> None:
        try:
            self._loop.run(
                frame_callback=self._processor.process,
                on_events=self._publish_all,
            )
        finally:
            threading.Thread(
                target=self._publisher.close, daemon=True, name="publisher-close"
            ).start()
            print("[Analyzer] Shutdown complete.")

    def _publish_all(self, events: list) -> None:
        for event in events:
            self._publisher.send(event)
