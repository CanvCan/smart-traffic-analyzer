from abc import ABC, abstractmethod
from typing import Any


class BaseTracker(ABC):
    """
    Strategy interface for multi-object trackers.
    Swap ByteTrack for BoTSORT or any other tracker
    by implementing this interface.
    """

    @abstractmethod
    def update(self, detections, frame) -> Any:
        ...


class ByteTracker(BaseTracker):
    """Supervision ByteTrack wrapper."""

    def __init__(self, lost_track_buffer: int = 150,
                 min_matching_threshold: float = 0.8,
                 min_consecutive_frames: int = 3):
        import supervision as sv
        self._tracker = sv.ByteTrack(
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=min_matching_threshold,
            minimum_consecutive_frames=min_consecutive_frames,
        )
        print(f"[Tracker] ByteTrack | buffer={lost_track_buffer} "
              f"| min_frames={min_consecutive_frames}")

    def update(self, detections, frame=None):
        return self._tracker.update_with_detections(detections)
