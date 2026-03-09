from collections import defaultdict, deque
from typing import Dict, Optional, Tuple


class TrackMemory:
    """
    Manages per-track state: class voting history, last known
    bounding box, last seen frame, and voted class.

    Implements a garbage collector to release stale track data.
    """

    def __init__(self, history_size: int = 15, max_lost_frames: int = 150):
        self.max_lost_frames = max_lost_frames
        self._class_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._last_seen:  Dict[int, int]             = {}
        self._last_box:   Dict[int, Tuple[int,...]]  = {}
        self._last_cls:   Dict[int, int]             = {}

    def update(self, tid: int, cls_id: int,
               box: Tuple[int, int, int, int], frame_count: int) -> int:
        """Update track state and return the majority-voted class."""
        self._last_seen[tid]  = frame_count
        self._last_box[tid]   = box
        self._class_history[tid].append(cls_id)

        history = self._class_history[tid]
        voted   = max(set(history), key=history.count)
        self._last_cls[tid] = voted
        return voted

    def get_box(self, tid: int) -> Optional[Tuple[int, int, int, int]]:
        return self._last_box.get(tid)

    def get_cls(self, tid: int) -> Optional[int]:
        return self._last_cls.get(tid)

    def get_lost_frames(self, tid: int, frame_count: int) -> int:
        return frame_count - self._last_seen.get(tid, frame_count)

    def all_ids(self):
        return list(self._last_seen.keys())

    def collect_garbage(self, frame_count: int) -> None:
        """Remove tracks that have not been seen for max_lost_frames."""
        stale = [
            tid for tid, last in self._last_seen.items()
            if (frame_count - last) > self.max_lost_frames
        ]
        for tid in stale:
            self._class_history.pop(tid, None)
            self._last_seen.pop(tid, None)
            self._last_box.pop(tid, None)
            self._last_cls.pop(tid, None)
