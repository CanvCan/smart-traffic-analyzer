import cv2
import numpy as np
from typing import Tuple, List
from traffic_analyzer.visualization.colors import CLASS_COLORS, CLASS_LABELS
from traffic_analyzer.utils.config_loader import LaneConfig

GHOST_FRAMES = 12  # Frames to show ghost box after occlusion


class Renderer:
    """
    Handles all OpenCV drawing operations.
    Keeps visualization logic fully decoupled from detection/tracking.
    """

    def draw_lanes(self, frame, lanes: List[LaneConfig]) -> None:
        overlay = frame.copy()
        for lane in lanes:
            r, c = lane.roi, tuple(lane.color)
            cv2.rectangle(overlay, (r[0], r[1]), (r[2], r[3]), c, -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        for lane in lanes:
            r, c = lane.roi, tuple(lane.color)
            cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), c, 2)
            (tw, th), _ = cv2.getTextSize(lane.name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(frame, (r[0], r[1] - th - 8), (r[0] + tw + 6, r[1]), c, -1)
            cv2.putText(frame, lane.name, (r[0] + 3, r[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    def draw_vehicle(self, frame, x1: int, y1: int, x2: int, y2: int,
                     track_id: int, cls_id: int, ghost: bool = False) -> None:
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label = f"ID:{track_id} {CLASS_LABELS.get(cls_id, 'Vehicle')}"
        c     = tuple(int(v * 0.5) for v in color) if ghost else color
        clen  = min(int((x2 - x1) * 0.25), int((y2 - y1) * 0.25), 20)

        if ghost:
            self._draw_dashed_box(frame, x1, y1, x2, y2, c)
        else:
            self._draw_corner_box(frame, x1, y1, x2, y2, c, clen)

        self._draw_label(frame, label, x1, y1, x2, y2, c, ghost)

    def draw_legend(self, frame) -> None:
        x, y  = 12, 12
        gap   = 26
        w     = 175
        h     = 12 + gap * len(CLASS_COLORS)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)

        for i, (cls_id, color) in enumerate(CLASS_COLORS.items()):
            cy = y + 6 + gap * i + gap // 2
            cv2.rectangle(frame, (x + 8, cy - 7), (x + 24, cy + 7), color, -1)
            cv2.putText(frame, CLASS_LABELS[cls_id], (x + 32, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_corner_box(self, frame, x1, y1, x2, y2,
                         color: Tuple, clen: int) -> None:
        for pts in [
            ((x1, y1+clen), (x1, y1), (x1+clen, y1)),
            ((x2-clen, y1), (x2, y1), (x2, y1+clen)),
            ((x1, y2-clen), (x1, y2), (x1+clen, y2)),
            ((x2-clen, y2), (x2, y2), (x2, y2-clen)),
        ]:
            cv2.polylines(frame, [np.array(pts)], False, color, 3, cv2.LINE_AA)

    def _draw_dashed_box(self, frame, x1, y1, x2, y2, color: Tuple) -> None:
        for i in range(x1, x2, 10):
            cv2.line(frame, (i, y1), (min(i+5, x2), y1), color, 1)
            cv2.line(frame, (i, y2), (min(i+5, x2), y2), color, 1)
        for i in range(y1, y2, 10):
            cv2.line(frame, (x1, i), (x1, min(i+5, y2)), color, 1)
            cv2.line(frame, (x2, i), (x2, min(i+5, y2)), color, 1)

    def _draw_label(self, frame, label: str, x1, y1, x2, y2,
                    color: Tuple, ghost: bool) -> None:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        lx = x1
        ly = y1 - 6 if y1 - 6 - th - 4 >= 0 else y2 + th + 6
        cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 4, ly), color, -1)
        cv2.putText(frame, label, (lx + 2, ly - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255) if not ghost else (180, 180, 180),
                    1, cv2.LINE_AA)
