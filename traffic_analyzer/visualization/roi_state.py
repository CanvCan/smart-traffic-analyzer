"""
traffic_analyzer/visualization/roi_state.py

Mutable editor state for the ROI lane selector.
Isolated from both UI and rendering so each layer can be read and tested
independently.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LaneEditorState:
    """All mutable state for one ROI editing session.

    Attributes:
        polygons:    Confirmed lane polygons (each is a dict with keys
                     ``name``, ``pts``, ``expected_direction``, ``label_pt``).
        current_pts: Points placed for the polygon currently being drawn
                     (original-frame coordinates).
        drag_idx:    Index into ``current_pts`` of the point being dragged,
                     or None.
        hover_idx:   Index into ``current_pts`` of the point under the cursor,
                     or None.
        mode:        Active editing step — ``"drawing"``, ``"naming"``, or
                     ``"direction"``.
        temp_name:   Lane name typed in the naming panel (not yet committed).
        temp_dir:    Direction value selected in the direction panel (not yet
                     committed).
    """

    polygons:       list[dict]             = field(default_factory=list)
    current_pts:    list[tuple[int, int]]  = field(default_factory=list)
    drag_idx:       Optional[int]          = None
    hover_idx:      Optional[int]          = None
    mode:           str                    = "drawing"
    temp_name:      str                    = ""
    temp_dir:       str                    = ""
    label_dragging: bool                   = False
