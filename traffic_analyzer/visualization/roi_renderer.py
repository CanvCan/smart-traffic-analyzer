"""
traffic_analyzer/visualization/roi_renderer.py

Pure OpenCV / PIL rendering for the ROI lane editor.

Responsibilities:
  - Convert LaneEditorState + a BGR base frame into an ImageTk.PhotoImage
    ready to be placed on a tk.Canvas.
  - Own all coordinate-space conversions (canvas px ↔ original-frame px).
  - Provide a ``nearest()`` helper for mouse-snap detection.

No CTk / tkinter dependency — only cv2, NumPy, and PIL.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from traffic_analyzer.visualization.colors     import LANE_PALETTE
from traffic_analyzer.visualization.roi_state  import LaneEditorState
from traffic_analyzer.visualization._pil_utils import pil_font as _pil_font, pil_text_size as _text_size


_SNAP_RADIUS = 12


def _np_to_photo(bgr: np.ndarray) -> ImageTk.PhotoImage:
    return ImageTk.PhotoImage(Image.fromarray(bgr[:, :, ::-1]))


def _lane_hex(idx: int) -> str:
    b, g, r = LANE_PALETTE[idx % len(LANE_PALETTE)]
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Renderer ──────────────────────────────────────────────────────────────────

class ROIRenderer:
    """
    Stateless canvas renderer for the ROI lane editor.

    All rendering logic (OpenCV overlays, PIL text) is centralised here so
    that ROISelectorWindow stays focused on UI layout and event handling.

    Args:
        scale: Ratio of displayed canvas pixels to original frame pixels.
    """

    def __init__(self, scale: float) -> None:
        self._scale = scale

    # ── Public interface ──────────────────────────────────────────────────────

    def render(self, base: np.ndarray,
               state: LaneEditorState) -> ImageTk.PhotoImage:
        """Produce a PhotoImage from the current editor state.

        Args:
            base:  BGR base frame (scaled to canvas size, never mutated).
            state: Current mutable editor state.

        Returns:
            PhotoImage suitable for ``canvas.itemconfig(img_id, image=...)``.
        """
        display  = base.copy()
        label_drag_idx = len(state.polygons) - 1 if state.mode == "label_drag" else -1
        deferred = self._draw_polygons(display, state.polygons, label_drag_idx)
        if state.current_pts:
            self._draw_active_polygon(
                display, state.current_pts,
                len(state.polygons), state.hover_idx,
            )
        self._draw_hud(display, state.mode, state.current_pts, len(state.polygons))
        if deferred:
            display = self._apply_labels(display, deferred)
        return _np_to_photo(display)

    def to_orig(self, cx: int, cy: int) -> tuple[int, int]:
        """Canvas pixel → original-frame coordinates."""
        return int(cx / self._scale), int(cy / self._scale)

    def to_canvas(self, ox: int, oy: int) -> tuple[int, int]:
        """Original-frame coordinates → canvas pixel."""
        return int(ox * self._scale), int(oy * self._scale)

    def nearest(self, cx: int, cy: int,
                current_pts: list[tuple[int, int]]) -> Optional[int]:
        """Return the index of the current point closest to (cx, cy) within
        the snap radius, or None if no point is close enough.
        """
        for i, (ox, oy) in enumerate(current_pts):
            px, py = self.to_canvas(ox, oy)
            if (px - cx) ** 2 + (py - cy) ** 2 <= _SNAP_RADIUS ** 2:
                return i
        return None

    # ── Private drawing helpers ───────────────────────────────────────────────

    def _draw_polygons(self, display: np.ndarray,
                       polygons: list[dict],
                       label_drag_idx: int = -1) -> list[tuple]:
        """Draw all confirmed polygons; return deferred PIL label data."""
        deferred: list[tuple[str, int, int, tuple]] = []

        for i, poly in enumerate(polygons):
            bgr  = LANE_PALETTE[i % len(LANE_PALETTE)]
            spts = np.array([self.to_canvas(*p) for p in poly["pts"]], np.int32)

            ov = display.copy()
            cv2.fillPoly(ov, [spts], bgr)
            cv2.addWeighted(ov, 0.22, display, 0.78, 0, display)
            cv2.polylines(display, [spts], True, bgr, 2, cv2.LINE_AA)

            for sp in spts:
                sp = tuple(sp)
                cv2.circle(display, sp, 5, bgr, -1, cv2.LINE_AA)
                cv2.circle(display, sp, 5, (255, 255, 255), 1, cv2.LINE_AA)

            if "label_pt" in poly:
                is_drag_target = (i == label_drag_idx)
                ox, oy = int(poly["label_pt"][0]), int(poly["label_pt"][1])
                sx, sy = self.to_canvas(ox, oy)
                tw, th = _text_size(poly["name"], 13)
                pad    = 7
                bx1    = max(sx - pad, 0)
                by1    = max(sy - pad, 0)
                bx2    = min(sx + tw + pad, display.shape[1] - 1)
                by2    = min(sy + th + pad, display.shape[0] - 1)
                ov2    = display.copy()
                cv2.rectangle(ov2, (bx1, by1), (bx2, by2), (10, 8, 18), -1)
                cv2.addWeighted(ov2, 0.88, display, 0.12, 0, display)
                border_clr = (80, 220, 130) if is_drag_target else bgr
                border_w   = 2             if is_drag_target else 1
                cv2.rectangle(display, (bx1, by1), (bx2, by2),
                              border_clr, border_w, cv2.LINE_AA)
                if is_drag_target:
                    # Corner drag handles
                    hs = 4
                    for hx, hy in [(bx1, by1), (bx2, by1), (bx1, by2), (bx2, by2)]:
                        cv2.rectangle(display,
                                      (hx - hs, hy - hs), (hx + hs, hy + hs),
                                      (80, 220, 130), -1)
                    # "move" hint icon (crosshair arrows)
                    mx, my = (bx1 + bx2) // 2, by1 - 14
                    if my > 8:
                        cv2.putText(display, "+", (mx - 5, my + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                                    (80, 220, 130), 1, cv2.LINE_AA)
                deferred.append((poly["name"], sx, sy, bgr))

        return deferred

    def _draw_active_polygon(self, display: np.ndarray,
                             current_pts: list[tuple[int, int]],
                             n_polygons: int,
                             hover_idx: Optional[int]) -> None:
        """Draw the in-progress polygon currently being placed."""
        bgr  = LANE_PALETTE[n_polygons % len(LANE_PALETTE)]
        spts = [self.to_canvas(*p) for p in current_pts]

        for j in range(len(spts) - 1):
            cv2.line(display, spts[j], spts[j + 1], bgr, 2, cv2.LINE_AA)

        if len(spts) >= 3:
            dim = tuple(max(0, int(c * 0.35)) for c in bgr)
            cv2.line(display, spts[-1], spts[0], dim, 1, cv2.LINE_AA)

        for j, sp in enumerate(spts):
            hover    = (j == hover_idx)
            r        = _SNAP_RADIUS + 4 if hover else _SNAP_RADIUS
            stroke_w = 2 if hover else 1

            glow = display.copy()
            cv2.circle(glow, sp, r + 6, bgr, -1, cv2.LINE_AA)
            blurred = cv2.GaussianBlur(glow, (0, 0), 4)
            cv2.addWeighted(blurred, 0.28, display, 0.72, 0, display)

            cv2.circle(display, sp, r, bgr, -1, cv2.LINE_AA)
            cv2.circle(display, sp, r, (240, 240, 255), stroke_w, cv2.LINE_AA)
            cv2.putText(display, str(j + 1), (sp[0] + r + 4, sp[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (220, 220, 240), 1, cv2.LINE_AA)

    def _draw_hud(self, display: np.ndarray, mode: str,
                  current_pts: list, n_polygons: int) -> None:
        """Render the glassmorphism HUD bar at the bottom of the canvas."""
        h, w = display.shape[:2]
        n    = len(current_pts)

        if mode == "drawing":
            if n == 0:
                msg = "Left-click to place points  ·  Enter: confirm  ·  S: save"
            elif n < 3:
                msg = f"{n} pt{'s' if n > 1 else ''} placed  ·  Need {3 - n} more  ·  R-Click: undo"
            else:
                msg = f"{n} pts ready  ·  Enter: confirm  ·  Drag to adjust  ·  R-Click: undo"
        elif mode == "naming":
            msg = "Type a name in the sidebar  ·  Enter: next  ·  Esc: cancel"
        else:
            msg = "Select traffic direction in the sidebar  ·  Esc: cancel"

        bar_h  = 38
        region = display[h - bar_h:h, 0:w]
        if region.size:
            blurred = cv2.GaussianBlur(region, (0, 0), 14)
            tint    = np.full_like(blurred, (8, 7, 14), dtype=np.uint8)
            mixed   = cv2.addWeighted(blurred, 0.55, tint, 0.45, 0)
            display[h - bar_h:h, 0:w] = mixed

        cv2.line(display, (0, h - bar_h), (w, h - bar_h), (28, 26, 48), 1)
        cv2.putText(display, msg, (16, h - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (140, 140, 185), 1, cv2.LINE_AA)

        if n_polygons:
            label   = f"{n_polygons} lane{'s' if n_polygons > 1 else ''}"
            bw, bh  = _text_size(label, 11)
            bw     += 16
            overlay = display.copy()
            cv2.rectangle(overlay, (10, 10), (10 + bw, 10 + bh + 8), (10, 8, 20), -1)
            cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
            cv2.rectangle(display, (10, 10), (10 + bw, 10 + bh + 8),
                          (40, 80, 100), 1, cv2.LINE_AA)
            cv2.putText(display, label, (18, 10 + bh + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (91, 189, 204), 1, cv2.LINE_AA)

    def _apply_labels(self, display: np.ndarray,
                      deferred: list[tuple]) -> np.ndarray:
        """Overlay PIL-rendered lane-name labels on the frame."""
        pil  = Image.fromarray(display[:, :, ::-1])
        draw = ImageDraw.Draw(pil)
        font = _pil_font(13)
        for text, sx, sy, bgr in deferred:
            rgb = (bgr[2], bgr[1], bgr[0])
            draw.text((sx + 1, sy + 1), text, font=font, fill=(0, 0, 0, 180))
            draw.text((sx, sy),         text, font=font, fill=rgb)
        return np.array(pil)[:, :, ::-1]
