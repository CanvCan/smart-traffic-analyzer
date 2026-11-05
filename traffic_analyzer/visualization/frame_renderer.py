"""
visualization/frame_renderer.py

Live-frame renderer with resolution-proportional (frame_height / 720) scaling
for bounding boxes, corner markers, vehicle speed/anomaly labels, lane polygons, and compact HUD.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from traffic_analyzer.domain.models import AnomalyType, LaneConfig, OverlaySnapshot
from traffic_analyzer.visualization.colors import CLASS_COLORS, CLASS_LABELS, LANE_PALETTE

_ANOMALY_DISPLAY: Dict[str, Tuple[str, Tuple[int, int, int]]] = {
    AnomalyType.STOPPED_VEHICLE.value: ("STOPPED", (50, 50, 235)),
    "STOPPED": ("STOPPED", (50, 50, 235)),
    AnomalyType.SUDDEN_SLOWDOWN.value: ("SLOWDOWN", (0, 140, 255)),
    "SLOWDOWN": ("SLOWDOWN", (0, 140, 255)),
    AnomalyType.WRONG_WAY.value: ("WRONG_WAY", (40, 40, 240)),
    "WRONG_WAY": ("WRONG_WAY", (40, 40, 240)),
}


def _alpha_rect(
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        alpha: float,
) -> None:
    """Blend a filled BGR rectangle onto a clipped sub-ROI slice without full-frame copy."""
    h, w = img.shape[:2]
    cx1 = max(0, min(w, int(x1)))
    cy1 = max(0, min(h, int(y1)))
    cx2 = max(0, min(w, int(x2)))
    cy2 = max(0, min(h, int(y2)))
    if cx2 <= cx1 or cy2 <= cy1:
        return
    roi = img[cy1:cy2, cx1:cx2]
    fill = np.empty_like(roi)
    fill[:] = color
    cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0.0, roi)


class FrameRenderer:
    """
    Handles all OpenCV drawing operations for the live analysis pipeline.
    Scales typography, stroke thickness, and corner lengths dynamically
    relative to `frame_height / 720.0`.
    """

    def __init__(self, is_live: bool = False) -> None:
        self.is_live: bool = is_live
        self.show_lanes: bool = True
        self.show_legend: bool = False
        self.show_ghosts: bool = True
        self.show_labels: bool = True

        self._lane_cache_key: Optional[Tuple] = None
        self._lane_fill_layer: Optional[np.ndarray] = None
        self._lane_line_mask: Optional[np.ndarray] = None
        self._lane_line_layer: Optional[np.ndarray] = None
        self._last_vehicle_count: int = 0

    @staticmethod
    def _scale_factor(frame_height: int) -> float:
        return max(0.5, float(frame_height) / 720.0)

    def _build_lane_cache(self, frame_shape: Tuple[int, int, int], lanes: List[LaneConfig]) -> None:
        h, w = frame_shape[:2]
        scale = self._scale_factor(h)
        line_thick = max(1, int(round(1.8 * scale)))
        font_scale = max(0.40, round(0.46 * scale, 2))

        fill_layer = np.zeros((h, w, 3), dtype=np.uint8)
        line_layer = np.zeros((h, w, 3), dtype=np.uint8)

        for i, lane in enumerate(lanes):
            if not getattr(lane, "visible", True):
                continue
            c = LANE_PALETTE[i % len(LANE_PALETTE)]
            if lane.points:
                pts = np.array(lane.points, dtype=np.int32)
                cv2.fillPoly(fill_layer, [pts], c)
                cv2.polylines(line_layer, [pts], isClosed=True, color=c, thickness=line_thick, lineType=cv2.LINE_AA)
            else:
                r = lane.roi
                cv2.rectangle(fill_layer, (r[0], r[1]), (r[2], r[3]), c, -1)
                cv2.rectangle(line_layer, (r[0], r[1]), (r[2], r[3]), c, line_thick)

            if lane.label_pt and len(lane.label_pt) == 2:
                lx, ly = int(lane.label_pt[0]), int(lane.label_pt[1])
                display_txt = getattr(lane, "display_name", "") or lane.name
                cv2.putText(
                    line_layer,
                    display_txt,
                    (lx + 1, ly + 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (12, 12, 18),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    line_layer,
                    display_txt,
                    (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    c,
                    1,
                    cv2.LINE_AA,
                )

        self._lane_fill_layer = fill_layer
        self._lane_line_layer = line_layer
        self._lane_line_mask = np.any(line_layer > 0, axis=2)

    def draw_lanes(self, frame: np.ndarray, lanes: List[LaneConfig]) -> None:
        if not self.show_lanes or not lanes:
            return

        cache_key = (
            frame.shape[:2],
            tuple(
                (
                    lane.name,
                    getattr(lane, "display_name", ""),
                    getattr(lane, "visible", True),
                    tuple(lane.roi),
                    tuple(tuple(p) for p in lane.points),
                    tuple(lane.label_pt),
                )
                for lane in lanes
            ),
        )
        if cache_key != self._lane_cache_key:
            self._build_lane_cache(frame.shape, lanes)
            self._lane_cache_key = cache_key

        if self._lane_fill_layer is not None:
            fill_mask = np.any(self._lane_fill_layer > 0, axis=2)
            if np.any(fill_mask):
                blended = cv2.addWeighted(self._lane_fill_layer, 0.10, frame, 0.90, 0)
                frame[fill_mask] = blended[fill_mask]

        if self._lane_line_layer is not None and self._lane_line_mask is not None:
            frame[self._lane_line_mask] = self._lane_line_layer[self._lane_line_mask]

    def draw_vehicle(
            self,
            frame: np.ndarray,
            x1: int,
            y1: int,
            x2: int,
            y2: int,
            track_id: int,
            cls_id: int,
            speed_px: float = 0.0,
            anomaly_type: Optional[str] = None,
            ghost: bool = False,
    ) -> None:
        if ghost and not self.show_ghosts:
            return
        scale = self._scale_factor(frame.shape[0])
        thickness = max(1, int(round(1.8 * scale)))
        max_corner = max(8, int(round(14.0 * scale)))

        cls_name = CLASS_LABELS.get(cls_id, "Vehicle")
        base_color = CLASS_COLORS.get(cls_id, (200, 200, 200))

        anom_key = str(getattr(anomaly_type, "value", anomaly_type) or "")
        anom_info = _ANOMALY_DISPLAY.get(anom_key) if (anom_key and not ghost) else None

        if anom_info is not None:
            anom_tag, box_color = anom_info
            label = f"#{track_id} {cls_name} {int(round(speed_px))} px/s [{anom_tag}]"
        elif not ghost and speed_px > 0.0:
            box_color = base_color
            label = f"#{track_id} {cls_name} {int(round(speed_px))} px/s"
        else:
            box_color = base_color
            label = f"#{track_id} {cls_name}"

        c = tuple(int(v * 0.5) for v in box_color) if ghost else box_color
        clen = min(int((x2 - x1) * 0.20), int((y2 - y1) * 0.20), max_corner)

        if ghost:
            self._draw_dashed_box(frame, x1, y1, x2, y2, c, max(1, thickness - 1))
        else:
            self._draw_corner_box(frame, x1, y1, x2, y2, c, clen, thickness)

        if self.show_labels:
            self._draw_label(frame, label, x1, y1, x2, y2, c, ghost, scale)

    def draw_snapshot(
            self,
            frame: np.ndarray,
            lanes: List[LaneConfig],
            snapshot: Optional[OverlaySnapshot],
    ) -> None:
        self.draw_lanes(frame, lanes)
        active_count = 0
        if snapshot is not None:
            for trk in snapshot.tracks:
                if not trk.is_ghost:
                    active_count += 1
                x1, y1, x2, y2 = trk.bbox
                self.draw_vehicle(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    trk.track_id,
                    trk.class_id,
                    speed_px=trk.speed_px,
                    anomaly_type=trk.anomaly_type,
                    ghost=trk.is_ghost,
                )
        self._last_vehicle_count = active_count
        if self.show_legend:
            self.draw_legend(frame)

    def format_hud_text(
            self,
            fps: float,
            vehicle_count: Optional[int] = None,
            paused: bool = False,
            is_live: Optional[bool] = None,
    ) -> str:
        """Build the top-left HUD string dynamically based on whether the stream is live."""
        live = self.is_live if is_live is None else bool(is_live)
        count = self._last_vehicle_count if vehicle_count is None else vehicle_count
        if live:
            return f"FPS: {fps:4.1f}  |  Vehicles: {count}  |  🔴 LIVE | Q: Quit"
        status_part = "  |  PAUSED" if paused else ""
        return f"FPS: {fps:4.1f}  |  Vehicles: {count}{status_part}  |  Space: Pause | Q: Quit"

    def draw_hud(
            self,
            frame: np.ndarray,
            fps: float,
            vehicle_count: Optional[int] = None,
            paused: bool = False,
            is_live: Optional[bool] = None,
    ) -> None:
        """Render a compact top-left HUD bar scaled to frame resolution."""
        h, _w = frame.shape[:2]
        scale = self._scale_factor(h)
        font_scale = max(0.38, round(0.44 * scale, 2))
        live = self.is_live if is_live is None else bool(is_live)
        text = self.format_hud_text(fps=fps, vehicle_count=vehicle_count, paused=paused, is_live=live)

        measure_text = text.replace("🔴 ", "O ")
        (tw, th), _ = cv2.getTextSize(measure_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        pad_x = max(8, int(round(10 * scale)))
        pad_y = max(6, int(round(7 * scale)))
        x1, y1 = 12, 12
        x2, y2 = x1 + tw + pad_x * 2, y1 + th + pad_y * 2

        _alpha_rect(frame, x1, y1, x2, y2, (16, 19, 42), 0.78)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (58, 66, 110), 1)

        baseline_y = y1 + pad_y + th - 1
        if live and "🔴 " in text:
            prefix, suffix = text.split("🔴 ", 1)
            cv2.putText(
                frame,
                prefix,
                (x1 + pad_x, baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (225, 230, 248),
                1,
                cv2.LINE_AA,
            )
            (pw, _), _ = cv2.getTextSize(prefix, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            (ow, _), _ = cv2.getTextSize("O ", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            dot_r = max(4, int(round(4.5 * scale)))
            dot_cx = x1 + pad_x + pw + dot_r + 1
            dot_cy = baseline_y - th // 2
            cv2.circle(frame, (dot_cx, dot_cy), dot_r, (50, 50, 235), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                frame,
                suffix,
                (x1 + pad_x + pw + ow, baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (225, 230, 248),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                text,
                (x1 + pad_x, baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (225, 230, 248),
                1,
                cv2.LINE_AA,
            )

    def draw_legend(self, frame: np.ndarray) -> None:
        if not self.show_legend:
            return
        scale = self._scale_factor(frame.shape[0])
        font_scale = max(0.38, round(0.44 * scale, 2))
        gap = max(18, int(round(22 * scale)))
        w = max(135, int(round(155 * scale)))
        x, y = 12, int(round(48 * scale))
        h = 10 + gap * len(CLASS_COLORS)

        _alpha_rect(frame, x, y, x + w, y + h, (16, 19, 42), 0.72)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (58, 66, 110), 1)

        for i, (cls_id, color) in enumerate(CLASS_COLORS.items()):
            cy = y + 5 + gap * i + gap // 2
            box_r = max(4, int(round(5 * scale)))
            cv2.rectangle(frame, (x + 8, cy - box_r), (x + 8 + box_r * 2, cy + box_r), color, -1)
            cv2.putText(
                frame,
                CLASS_LABELS[cls_id],
                (x + 14 + box_r * 2, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (220, 224, 240),
                1,
                cv2.LINE_AA,
            )

    def _draw_corner_box(
            self,
            frame: np.ndarray,
            x1: int,
            y1: int,
            x2: int,
            y2: int,
            color: Tuple[int, int, int],
            clen: int,
            thickness: int,
    ) -> None:
        for pts in [
            ((x1, y1 + clen), (x1, y1), (x1 + clen, y1)),
            ((x2 - clen, y1), (x2, y1), (x2, y1 + clen)),
            ((x1, y2 - clen), (x1, y2), (x1 + clen, y2)),
            ((x2 - clen, y2), (x2, y2), (x2, y2 - clen)),
        ]:
            cv2.polylines(frame, [np.array(pts)], False, color, thickness, cv2.LINE_AA)

    def _draw_dashed_box(
            self,
            frame: np.ndarray,
            x1: int,
            y1: int,
            x2: int,
            y2: int,
            color: Tuple[int, int, int],
            thickness: int,
    ) -> None:
        for i in range(x1, x2, 10):
            cv2.line(frame, (i, y1), (min(i + 5, x2), y1), color, thickness)
            cv2.line(frame, (i, y2), (min(i + 5, x2), y2), color, thickness)
        for i in range(y1, y2, 10):
            cv2.line(frame, (x1, i), (x1, min(i + 5, y2)), color, thickness)
            cv2.line(frame, (x2, i), (x2, min(i + 5, y2)), color, thickness)

    def _draw_label(
            self,
            frame: np.ndarray,
            label: str,
            x1: int,
            y1: int,
            x2: int,
            y2: int,
            color: Tuple[int, int, int],
            ghost: bool,
            scale: float,
    ) -> None:
        font_scale = max(0.34, round(0.40 * scale, 2))
        pad_x = max(3, int(round(4 * scale)))
        pad_y = max(2, int(round(3 * scale)))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        lx = x1
        ly = y1 - 3 if y1 - th - pad_y * 2 - 3 >= 0 else y2 + th + pad_y + 3
        _alpha_rect(
            frame,
            lx,
            ly - th - pad_y * 2,
            lx + tw + pad_x * 2,
            ly + 1,
            color,
            0.78,
        )
        text_color = (255, 255, 255) if not ghost else (185, 185, 185)
        cv2.putText(
            frame,
            label,
            (lx + pad_x, ly - pad_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )
