"""
visualization/ghost_track_manager.py

Re-exports GhostTrackManager from traffic_analyzer.services.ghost_track_manager
for backward compatibility.
"""

from traffic_analyzer.services.ghost_track_manager import (
    DEFAULT_GHOST_FRAMES,
    GhostTrackManager,
)

__all__ = ["DEFAULT_GHOST_FRAMES", "GhostTrackManager"]
