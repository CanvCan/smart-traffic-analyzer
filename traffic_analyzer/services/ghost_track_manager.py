"""
services/ghost_track_manager.py

Manages ghost track state for recently-lost tracks during short occlusions.
A ghost track is a vehicle that has temporarily disappeared from the detector
but whose last known bounding box and class are retained for up to `ghost_frames` frames.
"""

from __future__ import annotations

DEFAULT_GHOST_FRAMES = 30
"""Default number of frames a lost track is retained as a ghost."""


class GhostTrackManager:
    """
    Tracks the last known position and class of each vehicle so that
    recently-lost tracks can be represented during brief occlusions.
    """

    def __init__(self, ghost_frames: int = DEFAULT_GHOST_FRAMES):
        self.ghost_frames = ghost_frames
        self._last_box: dict = {}
        self._last_cls: dict = {}
        self._last_seen: dict = {}

    def update(self, tid: int, box: tuple, cls_id: int, frame_count: int) -> None:
        """Record the latest observation for an active track."""
        self._last_box[tid] = box
        self._last_cls[tid] = cls_id
        self._last_seen[tid] = frame_count

    def get_ghosts(self, frame_count: int, active_ids: set) -> list:
        """
        Return ghost entries for tracks that are lost but within the
        ghost_frames window.

        Each entry is a dict: {"tid": int, "box": tuple, "cls_id": int}.
        """
        ghosts = []
        for tid, last_frame in self._last_seen.items():
            lost = frame_count - last_frame
            if 0 < lost <= self.ghost_frames and tid not in active_ids:
                box = self._last_box.get(tid)
                if box:
                    ghosts.append({
                        "tid": tid,
                        "box": box,
                        "cls_id": self._last_cls.get(tid, 1),
                    })
        return ghosts

    def cleanup(self, frame_count: int) -> None:
        """Remove stale entries that are beyond the ghost_frames window."""
        stale = [tid for tid, lf in self._last_seen.items()
                 if frame_count - lf > self.ghost_frames]
        for tid in stale:
            self._last_box.pop(tid, None)
            self._last_cls.pop(tid, None)
            self._last_seen.pop(tid, None)
