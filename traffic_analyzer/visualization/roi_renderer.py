from typing import List, Optional, Tuple
import math
import cv2
import numpy as np
from PIL import Image, ImageTk


VERTEX_HIT_RADIUS_PX: int = 12
CLOSE_SNAP_RADIUS_PX: int = 15

LANE_COLORS_BGR = [
    (229, 70, 79),    # Indigo
    (129, 185, 16),   # Emerald
    (11, 158, 245),   # Amber
    (246, 92, 139),   # Purple
    (233, 165, 14),   # Sky Blue
    (153, 72, 236),   # Pink
    (166, 184, 20),   # Teal
]

SELECTED_BORDER_BGR = (0, 230, 255)   # Bright yellow/cyan highlight in BGR
SELECTED_VERTEX_BGR = (255, 255, 0)   # Bright cyan vertex handle in BGR

DIR_ARROWS = {
    "DOWN":  (0,  1, "DOWN"),
    "UP":    (0, -1, "UP"),
    "LEFT":  (-1, 0, "LEFT"),
    "RIGHT": (1,  0, "RIGHT"),
}


class ROIRenderer:
    """Letterbox coordinate projection and OpenCV overlay renderer for the ROI editor."""

    def __init__(self, frame_w: int, frame_h: int) -> None:
        self.frame_w = max(1, frame_w)
        self.frame_h = max(1, frame_h)
        self._scale: float = 1.0
        self._offset_x: int = 0
        self._offset_y: int = 0

    def update_projection(self, canvas_w: int, canvas_h: int) -> None:
        scale_w = canvas_w / self.frame_w
        scale_h = canvas_h / self.frame_h
        self._scale = min(scale_w, scale_h)
        scaled_w = int(self.frame_w * self._scale)
        scaled_h = int(self.frame_h * self._scale)
        self._offset_x = (canvas_w - scaled_w) // 2
        self._offset_y = (canvas_h - scaled_h) // 2

    def canvas_to_frame(self, cx: int, cy: int) -> Tuple[int, int]:
        if self._scale <= 0:
            return 0, 0
        fx = int((cx - self._offset_x) / self._scale)
        fy = int((cy - self._offset_y) / self._scale)
        fx = max(0, min(self.frame_w - 1, fx))
        fy = max(0, min(self.frame_h - 1, fy))
        return fx, fy

    def frame_to_canvas(self, fx: int, fy: int) -> Tuple[int, int]:
        cx = int(fx * self._scale) + self._offset_x
        cy = int(fy * self._scale) + self._offset_y
        return cx, cy

    def find_nearest_vertex(
        self,
        cx: int,
        cy: int,
        current_pts: List[List[int]],
    ) -> Optional[int]:
        if not current_pts:
            return None
        best_idx: Optional[int] = None
        best_dist: float = float(VERTEX_HIT_RADIUS_PX)
        for idx, pt in enumerate(current_pts):
            vx, vy = self.frame_to_canvas(pt[0], pt[1])
            dist = math.hypot(cx - vx, cy - vy)
            if dist <= best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def find_lane_vertex(
        self,
        cx: int,
        cy: int,
        lanes: List[dict],
        preferred_lane_idx: Optional[int] = None,
    ) -> Optional[Tuple[int, int]]:
        """Return (lane_idx, vertex_idx) if canvas point (cx, cy) hits a saved lane vertex."""
        if not lanes:
            return None

        order: List[int] = list(range(len(lanes)))
        if preferred_lane_idx is not None and 0 <= preferred_lane_idx < len(lanes):
            order.remove(preferred_lane_idx)
            order.insert(0, preferred_lane_idx)

        best_match: Optional[Tuple[int, int]] = None
        best_dist: float = float(VERTEX_HIT_RADIUS_PX)
        for l_idx in order:
            pts = lanes[l_idx].get("points", [])
            for v_idx, pt in enumerate(pts):
                vx, vy = self.frame_to_canvas(pt[0], pt[1])
                dist = math.hypot(cx - vx, cy - vy)
                if dist <= best_dist:
                    best_dist = dist
                    best_match = (l_idx, v_idx)
        return best_match

    def find_containing_lane(
        self,
        cx: int,
        cy: int,
        lanes: List[dict],
    ) -> Optional[int]:
        """Return the index of the lane polygon containing canvas point (cx, cy), or None."""
        if not lanes:
            return None
        fx, fy = self.canvas_to_frame(cx, cy)
        # Iterate in reverse so topmost drawn lane wins on overlap
        for idx in range(len(lanes) - 1, -1, -1):
            pts = lanes[idx].get("points", [])
            if len(pts) < 3:
                continue
            contour = np.array(pts, dtype=np.int32)
            inside = cv2.pointPolygonTest(contour, (float(fx), float(fy)), False)
            if inside >= 0:
                return idx
        return None

    def render_photo(
        self,
        base_frame: np.ndarray,
        canvas_w: int,
        canvas_h: int,
        lanes: List[dict],
        current_pts: List[List[int]],
        hover_pt: Optional[Tuple[int, int]],
        hover_idx: Optional[int],
        drag_idx: Optional[int],
        close_snap_active: bool,
        polygon_closed: bool,
        selected_lane_idx: Optional[int] = None,
    ) -> ImageTk.PhotoImage:
        self.update_projection(canvas_w, canvas_h)
        scaled_w = max(1, int(self.frame_w * self._scale))
        scaled_h = max(1, int(self.frame_h * self._scale))

        img = self._compose_frame(
            base_frame=base_frame,
            lanes=lanes,
            current_pts=current_pts,
            hover_pt=hover_pt,
            hover_idx=hover_idx,
            drag_idx=drag_idx,
            close_snap_active=close_snap_active,
            polygon_closed=polygon_closed,
            selected_lane_idx=selected_lane_idx,
        )
        resized = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        canvas_img = np.full((canvas_h, canvas_w, 3), (20, 13, 10), dtype=np.uint8)
        y1 = max(0, self._offset_y)
        x1 = max(0, self._offset_x)
        y2 = min(canvas_h, y1 + scaled_h)
        x2 = min(canvas_w, x1 + scaled_w)
        canvas_img[y1:y2, x1:x2] = resized[: y2 - y1, : x2 - x1]

        rgb = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _compose_frame(
        self,
        base_frame: np.ndarray,
        lanes: List[dict],
        current_pts: List[List[int]],
        hover_pt: Optional[Tuple[int, int]],
        hover_idx: Optional[int],
        drag_idx: Optional[int],
        close_snap_active: bool,
        polygon_closed: bool,
        selected_lane_idx: Optional[int] = None,
    ) -> np.ndarray:
        canvas = base_frame.copy()
        overlay = base_frame.copy()

        for i, lane in enumerate(lanes):
            pts = np.array(lane["points"], np.int32).reshape((-1, 1, 2))
            is_selected = (i == selected_lane_idx)
            color = SELECTED_BORDER_BGR if is_selected else LANE_COLORS_BGR[i % len(LANE_COLORS_BGR)]
            cv2.fillPoly(overlay, [pts], color)

        cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)

        for i, lane in enumerate(lanes):
            pts = np.array(lane["points"], np.int32).reshape((-1, 1, 2))
            is_selected = (i == selected_lane_idx)
            color = SELECTED_BORDER_BGR if is_selected else LANE_COLORS_BGR[i % len(LANE_COLORS_BGR)]
            thickness = 3 if is_selected else 2
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

            for pt in lane["points"]:
                if is_selected:
                    cv2.circle(canvas, tuple(pt), 7, (15, 23, 42), -1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, tuple(pt), 5, SELECTED_VERTEX_BGR, -1, lineType=cv2.LINE_AA)
                else:
                    cv2.circle(canvas, tuple(pt), 4, color, -1, lineType=cv2.LINE_AA)

            M = cv2.moments(pts)
            if M["m00"] != 0:
                cx_m = int(M["m10"] / M["m00"])
                cy_m = int(M["m01"] / M["m00"])
            else:
                cx_m, cy_m = lane["points"][0]

            direction = lane.get("expected_direction")
            self._draw_lane_badge(
                canvas,
                lane["name"],
                direction,
                cx_m,
                cy_m,
                color,
                is_selected=is_selected,
            )

            if direction and direction in DIR_ARROWS:
                dx, dy, _ = DIR_ARROWS[direction]
                arrow_len = 38
                end_pt = (cx_m + dx * arrow_len, cy_m + dy * arrow_len + 26)
                start_pt = (cx_m - dx * (arrow_len // 2), cy_m - dy * (arrow_len // 2) + 26)
                cv2.arrowedLine(
                    canvas,
                    start_pt,
                    end_pt,
                    (255, 255, 255),
                    3,
                    line_type=cv2.LINE_AA,
                    tipLength=0.35,
                )

        if current_pts:
            if polygon_closed and len(current_pts) >= 3:
                active_overlay = canvas.copy()
                pts_arr = np.array(current_pts, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(active_overlay, [pts_arr], (129, 185, 16))
                cv2.addWeighted(active_overlay, 0.24, canvas, 0.76, 0, canvas)
                cv2.polylines(
                    canvas,
                    [pts_arr],
                    isClosed=True,
                    color=(129, 185, 16),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            else:
                for i in range(len(current_pts) - 1):
                    cv2.line(
                        canvas,
                        tuple(current_pts[i]),
                        tuple(current_pts[i + 1]),
                        (50, 220, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )

                if hover_pt is not None and drag_idx is None:
                    line_color = (129, 185, 16) if close_snap_active else (180, 220, 255)
                    cv2.line(
                        canvas,
                        tuple(current_pts[-1]),
                        hover_pt,
                        line_color,
                        1,
                        lineType=cv2.LINE_AA,
                    )
                    if len(current_pts) >= 2:
                        cv2.line(
                            canvas,
                            hover_pt,
                            tuple(current_pts[0]),
                            (100, 100, 255),
                            1,
                            lineType=cv2.LINE_AA,
                        )

            for i, pt in enumerate(current_pts):
                pt_t = tuple(pt)
                is_first = (i == 0)
                is_hovered = (i == hover_idx) or (i == drag_idx)

                if is_first and close_snap_active and len(current_pts) >= 3 and not polygon_closed:
                    cv2.circle(canvas, pt_t, 12, (129, 185, 16), 2, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, pt_t, 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                elif is_hovered:
                    cv2.circle(canvas, pt_t, 9, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, pt_t, 5, (0, 215, 255), -1, lineType=cv2.LINE_AA)
                else:
                    v_col = (0, 140, 255) if is_first else (0, 215, 255)
                    cv2.circle(canvas, pt_t, 6, (15, 23, 42), -1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, pt_t, 4, v_col, -1, lineType=cv2.LINE_AA)

                cv2.putText(
                    canvas,
                    str(i + 1),
                    (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        self._draw_bottom_hud(
            canvas,
            current_pts,
            close_snap_active,
            polygon_closed,
            selected_lane_idx=selected_lane_idx,
        )
        return canvas

    def _draw_lane_badge(
        self,
        img: np.ndarray,
        name: str,
        direction: Optional[str],
        cx: int,
        cy: int,
        color: Tuple[int, int, int],
        is_selected: bool = False,
    ) -> None:
        label = name
        if direction and direction in DIR_ARROWS:
            label = f"{name} [{DIR_ARROWS[direction][2]}]"
        if is_selected:
            label = f"* {label} (EDITING)"

        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.52
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)

        pad_x, pad_y = 10, 6
        x1 = max(4, cx - tw // 2 - pad_x)
        y1 = max(4, cy - th // 2 - pad_y)
        x2 = min(self.frame_w - 4, cx + tw // 2 + pad_x)
        y2 = min(self.frame_h - 4, cy + th // 2 + pad_y)

        bg_col = (30, 41, 59) if is_selected else (15, 23, 42)
        border_thickness = 2 if is_selected else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), bg_col, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, border_thickness, lineType=cv2.LINE_AA)
        cv2.putText(
            img,
            label,
            (x1 + pad_x, y2 - pad_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_bottom_hud(
        self,
        img: np.ndarray,
        current_pts: List[List[int]],
        close_snap_active: bool,
        polygon_closed: bool,
        selected_lane_idx: Optional[int] = None,
    ) -> None:
        h, w = img.shape[:2]
        bar_h = 32
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 10, 7), -1)
        cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

        n_pts = len(current_pts)
        if selected_lane_idx is not None and n_pts == 0:
            msg = "Edit Mode: Change name/direction in sidebar, drag vertices, or click empty space to deselect."
            col = (0, 230, 255)
        elif polygon_closed:
            msg = f"Polygon closed ({n_pts} pts) — Press 'Save Lane' in sidebar or drag vertices to refine."
            col = (129, 185, 16)
        elif close_snap_active:
            msg = "Click first vertex (P1) to snap & close polygon."
            col = (129, 185, 16)
        elif n_pts == 0:
            msg = "Left-click empty space to draw a lane, or click an existing lane to edit it in-place."
            col = (210, 200, 190)
        elif n_pts < 3:
            msg = f"{n_pts} point(s) placed — Add at least {3 - n_pts} more point(s) to form a polygon."
            col = (50, 220, 255)
        else:
            msg = f"{n_pts} points placed — Click P1 or double-click to close polygon, or press Enter to save."
            col = (50, 220, 255)

        cv2.putText(
            img,
            msg,
            (14, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            col,
            1,
            cv2.LINE_AA,
        )
