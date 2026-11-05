from pathlib import Path
from typing import Dict, List, Optional, Tuple
import copy
import json
import logging
import math
import os
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import customtkinter as ctk

from traffic_analyzer.domain.theme import (
    BG_PRIMARY as _BG,
    BG_SECONDARY as _SURFACE,
    BG_TERTIARY as _CARD,
    BG_ELEVATED as _CARD_HOVER,
    BORDER_SUBTLE as _BORDER,
    ACCENT_PRIMARY as _ACCENT,
    ACCENT_HOVER as _ACCENT_HOVER,
    ACCENT_SUCCESS as _EMERALD,
    ACCENT_SUCCESS_HOVER as _EMERALD_HOVER,
    ACCENT_WARNING as _AMBER,
    ACCENT_DANGER as _RED,
    ACCENT_DANGER_HOVER as _RED_HOVER,
    TEXT_PRIMARY as _TEXT,
    TEXT_SECONDARY as _TEXT_SEC,
    TEXT_MUTED as _TEXT_MUTED,
)
from traffic_analyzer.infrastructure.config_loader import (
    resolve_camera_json_path,
    save_camera_lanes,
)
from traffic_analyzer.visualization.roi_renderer import (
    CLOSE_SNAP_RADIUS_PX,
    LANE_COLORS_BGR,
    ROIRenderer,
    VERTEX_HIT_RADIUS_PX,
)

logger = logging.getLogger(__name__)

_MODEL_TO_UI_DIR: Dict[str, Optional[str]] = {
    "top_to_bottom": "DOWN",
    "bottom_to_top": "UP",
    "right_to_left": "LEFT",
    "left_to_right": "RIGHT",
    "DOWN": "DOWN",
    "UP": "UP",
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
}

_UI_TO_MODEL_DIR: Dict[str, str] = {
    "DOWN": "top_to_bottom",
    "UP": "bottom_to_top",
    "LEFT": "right_to_left",
    "RIGHT": "left_to_right",
    "top_to_bottom": "top_to_bottom",
    "bottom_to_top": "bottom_to_top",
    "right_to_left": "right_to_left",
    "left_to_right": "left_to_right",
}


def _bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


class ROISelectorWindow(ctk.CTkToplevel):
    """Interactive polygon ROI editor with in-place lane editing and dirty-state protection."""

    def __init__(
        self,
        parent: Optional[tk.Misc],
        frame: np.ndarray,
        config_path: str = "config.json",
        source_id: str = "default",
        cameras_dir: Optional[os.PathLike] = None,
    ) -> None:
        self._owns_root = parent is None
        if self._owns_root:
            ctk.set_appearance_mode("Dark")
            self._hidden_root = ctk.CTk()
            self._hidden_root.withdraw()
            super().__init__(self._hidden_root)
        else:
            self._hidden_root = None
            super().__init__(parent)

        self.title(f"Smart Traffic Analyzer — Interactive ROI Editor [{source_id}]")
        self.geometry("1360x820")
        self.minsize(1024, 660)
        self.configure(fg_color=_BG)

        if parent is not None:
            self.transient(parent)
        self.grab_set()
        self.focus_force()

        self._base_frame: np.ndarray = frame.copy()
        self._frame_h, self._frame_w = self._base_frame.shape[:2]
        self._renderer = ROIRenderer(self._frame_w, self._frame_h)
        self._config_path = config_path
        self._source_id = source_id
        self._cameras_dir = Path(cameras_dir) if cameras_dir is not None else self._default_cameras_dir()

        self._lanes: List[dict] = []
        self._initial_lanes: List[dict] = []
        self._current_pts: List[List[int]] = []
        self._hover_pt: Optional[Tuple[int, int]] = None
        self._hover_vertex_idx: Optional[int] = None
        self._dragging_vertex_idx: Optional[int] = None
        self._dragging_lane_vertex: Optional[Tuple[int, int]] = None
        self._drag_moved: bool = False
        self._close_snap_active: bool = False
        self._polygon_closed: bool = False
        self._selected_dir: Optional[str] = "DOWN"
        self._selected_lane_idx: Optional[int] = None
        self._updating_ui_from_selection: bool = False
        self.is_dirty: bool = False
        self._saved: bool = False
        self._photo = None

        self._load_existing_lanes()
        self._initial_lanes = copy.deepcopy(self._lanes)
        self._build_ui()
        self._bind_shortcuts()
        self.after(60, self._render_canvas)

    @property
    def selected_lane_idx(self) -> Optional[int]:
        return self._selected_lane_idx

    @selected_lane_idx.setter
    def selected_lane_idx(self, value: Optional[int]) -> None:
        self._select_lane(value)

    def _default_cameras_dir(self) -> Path:
        cfg_parent = Path(os.path.abspath(self._config_path)).parent
        if cfg_parent.name == "traffic_analyzer":
            cam_dir = cfg_parent / "cameras"
        else:
            cam_dir = cfg_parent / "traffic_analyzer" / "cameras"
        cam_dir.mkdir(parents=True, exist_ok=True)
        return cam_dir

    def _get_camera_json_path(self) -> str:
        self._cameras_dir.mkdir(parents=True, exist_ok=True)
        resolved = resolve_camera_json_path(self._source_id, self._cameras_dir)
        if resolved is not None:
            return str(resolved)
        return str(self._cameras_dir / f"{self._source_id}.json")

    def _load_existing_lanes(self) -> None:
        resolved_path = resolve_camera_json_path(self._source_id, self._cameras_dir)
        if resolved_path is None or not resolved_path.exists():
            # First time configuring ROI for this source: open with a clean empty template
            self._lanes = []
            return

        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_lanes = data.get("lanes", {})
            if isinstance(raw_lanes, dict):
                for name, item in raw_lanes.items():
                    if not isinstance(item, dict):
                        continue
                    pts = item.get("points", [])
                    if len(pts) >= 3:
                        raw_dir = str(item.get("expected_direction") or "")
                        self._lanes.append({
                            "name": str(item.get("display_name") or name),
                            "points": [[int(p[0]), int(p[1])] for p in pts],
                            "expected_direction": _MODEL_TO_UI_DIR.get(raw_dir),
                        })
            elif isinstance(raw_lanes, list):
                for item in raw_lanes:
                    if not isinstance(item, dict):
                        continue
                    pts = item.get("points", [])
                    if len(pts) >= 3:
                        raw_dir = str(item.get("expected_direction") or "")
                        self._lanes.append({
                            "name": str(item.get("name", f"Lane {len(self._lanes) + 1}")),
                            "points": [[int(p[0]), int(p[1])] for p in pts],
                            "expected_direction": _MODEL_TO_UI_DIR.get(raw_dir, item.get("expected_direction")),
                        })
        except Exception as exc:
            logger.warning("Could not load camera lanes from %s: %s", resolved_path, exc)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # Left Canvas Area
        canvas_container = ctk.CTkFrame(
            self,
            fg_color=_SURFACE,
            corner_radius=12,
            border_width=1,
            border_color=_BORDER,
        )
        canvas_container.grid(row=0, column=0, sticky="nsew", padx=(14, 7), pady=14)
        canvas_container.grid_rowconfigure(1, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)

        # Top Canvas Header Bar
        top_bar = ctk.CTkFrame(canvas_container, fg_color="transparent", height=42)
        top_bar.grid(row=0, column=0, sticky="ew", padx=16, pady=(10, 4))
        top_bar.grid_columnconfigure(1, weight=1)

        title_lbl = ctk.CTkLabel(
            top_bar,
            text=f"CAMERA: {self._source_id.upper()}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_TEXT,
        )
        title_lbl.grid(row=0, column=0, sticky="w")

        self._coord_lbl = ctk.CTkLabel(
            top_bar,
            text=f"Resolution: {self._frame_w}×{self._frame_h} px  |  Cursor: (—, —)",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color=_TEXT_SEC,
        )
        self._coord_lbl.grid(row=0, column=2, sticky="e")

        # Main Interactive tk.Canvas
        self._canvas = tk.Canvas(
            canvas_container,
            bg="#0A0D14",
            highlightthickness=0,
            cursor="crosshair",
        )
        self._canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind("<ButtonPress-1>", self._on_left_press)
        self._canvas.bind("<B1-Motion>", self._on_left_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self._canvas.bind("<Double-Button-1>", self._on_double_click)
        self._canvas.bind("<ButtonPress-3>", self._on_right_click)
        self._canvas.bind("<Motion>", self._on_mouse_move)

        # Right Control Sidebar
        sidebar = ctk.CTkFrame(
            self,
            width=350,
            fg_color=_SURFACE,
            corner_radius=12,
            border_width=1,
            border_color=_BORDER,
        )
        sidebar.grid(row=0, column=1, sticky="nsew", padx=(7, 14), pady=14)
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(3, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)

        # Section 1: Active / Selected Lane Configuration
        cfg_card = ctk.CTkFrame(
            sidebar,
            fg_color=_CARD,
            corner_radius=10,
            border_width=1,
            border_color=_BORDER,
        )
        cfg_card.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        cfg_card.grid_columnconfigure(0, weight=1)

        self._mode_header_lbl = ctk.CTkLabel(
            cfg_card,
            text="NEW LANE CONFIGURATION",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=_TEXT_SEC,
        )
        self._mode_header_lbl.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))

        ctk.CTkLabel(
            cfg_card,
            text="Lane Name",
            font=ctk.CTkFont(size=12),
            text_color=_TEXT,
        ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 4))

        self._name_var = tk.StringVar(value=f"Lane {len(self._lanes) + 1}")
        self._name_var.trace_add("write", self._on_name_var_changed)
        self._name_entry = ctk.CTkEntry(
            cfg_card,
            textvariable=self._name_var,
            height=34,
            fg_color=_SURFACE,
            border_color=_BORDER,
            text_color=_TEXT,
            font=ctk.CTkFont(size=13),
        )
        self._name_entry.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        self._name_entry.bind("<Return>", lambda _e: self._on_name_entry_return())

        ctk.CTkLabel(
            cfg_card,
            text="Traffic Direction",
            font=ctk.CTkFont(size=12),
            text_color=_TEXT,
        ).grid(row=3, column=0, sticky="w", padx=14, pady=(0, 6))

        dir_frame = ctk.CTkFrame(cfg_card, fg_color="transparent")
        dir_frame.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 12))
        for col_i in range(5):
            dir_frame.grid_columnconfigure(col_i, weight=1)

        self._dir_buttons = {}
        dir_specs = [
            ("DOWN", "↓"),
            ("UP", "↑"),
            ("LEFT", "←"),
            ("RIGHT", "→"),
            (None, "None"),
        ]
        for idx, (code, label) in enumerate(dir_specs):
            btn = ctk.CTkButton(
                dir_frame,
                text=label,
                height=32,
                width=50,
                corner_radius=6,
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda c=code: self._select_direction(c),
            )
            btn.grid(row=0, column=idx, padx=2, sticky="ew")
            self._dir_buttons[code] = btn
        self._refresh_direction_buttons()

        # Polygon Action Buttons
        poly_btn_frame = ctk.CTkFrame(cfg_card, fg_color="transparent")
        poly_btn_frame.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 14))
        poly_btn_frame.grid_columnconfigure(0, weight=1)
        poly_btn_frame.grid_columnconfigure(1, weight=1)

        self._undo_pt_btn = ctk.CTkButton(
            poly_btn_frame,
            text="Undo Point",
            height=34,
            fg_color=_SURFACE,
            hover_color=_CARD_HOVER,
            border_width=1,
            border_color=_BORDER,
            text_color=_TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._undo_last_point,
        )
        self._undo_pt_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        self._save_lane_btn = ctk.CTkButton(
            poly_btn_frame,
            text="Save Lane (Enter)",
            height=34,
            fg_color=_ACCENT,
            hover_color=_ACCENT_HOVER,
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._commit_current_lane,
        )
        self._save_lane_btn.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        # Section 2: Saved Lanes Header
        lanes_hdr = ctk.CTkFrame(sidebar, fg_color="transparent")
        lanes_hdr.grid(row=2, column=0, sticky="ew", padx=16, pady=(6, 4))
        lanes_hdr.grid_columnconfigure(0, weight=1)

        self._lanes_count_lbl = ctk.CTkLabel(
            lanes_hdr,
            text=f"SAVED LANES ({len(self._lanes)})",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=_TEXT_SEC,
        )
        self._lanes_count_lbl.grid(row=0, column=0, sticky="w")

        clear_all_btn = ctk.CTkButton(
            lanes_hdr,
            text="Clear All",
            width=70,
            height=24,
            fg_color="transparent",
            hover_color=_CARD,
            text_color=_RED,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._clear_all_lanes,
        )
        clear_all_btn.grid(row=0, column=1, sticky="e")

        # Section 3: Scrollable Saved Lanes List
        self._lanes_scroll = ctk.CTkScrollableFrame(
            sidebar,
            fg_color=_CARD,
            corner_radius=10,
            border_width=1,
            border_color=_BORDER,
        )
        self._lanes_scroll.grid(row=3, column=0, sticky="nsew", padx=14, pady=(0, 8))
        self._lanes_scroll.grid_columnconfigure(0, weight=1)
        self._refresh_lanes_list()

        # Section 4: Shortcuts Hint Card
        hint_card = ctk.CTkFrame(sidebar, fg_color=_CARD, corner_radius=8)
        hint_card.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 8))
        hints_text = (
            "• Click Polygon / List Row: Edit lane in-place\n"
            "• Left-Click Empty: Add point / Deselect lane\n"
            "• Drag Vertex: Move point  |  Right-Click: Undo\n"
            "• Click 1st Point / Enter: Close & save polygon"
        )
        ctk.CTkLabel(
            hint_card,
            text=hints_text,
            justify="left",
            font=ctk.CTkFont(size=11),
            text_color=_TEXT_SEC,
        ).pack(anchor="w", padx=12, pady=8)

        # Section 5: Bottom Action Buttons (Save & Close + Cancel / Exit)
        footer_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        footer_frame.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 14))
        footer_frame.grid_columnconfigure(0, weight=1)
        footer_frame.grid_columnconfigure(1, weight=1)

        self._save_close_btn = ctk.CTkButton(
            footer_frame,
            text="Save & Close",
            height=42,
            corner_radius=8,
            fg_color=_EMERALD,
            hover_color=_EMERALD_HOVER,
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._save_and_close,
        )
        self._save_close_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self._cancel_exit_btn = ctk.CTkButton(
            footer_frame,
            text="Cancel / Exit",
            height=42,
            corner_radius=8,
            fg_color=_CARD,
            hover_color=_RED_HOVER,
            border_width=1,
            border_color=_BORDER,
            text_color=_TEXT,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_close_request,
        )
        self._cancel_exit_btn.grid(row=0, column=1, padx=(5, 0), sticky="ew")

    def _bind_shortcuts(self) -> None:
        self.bind("<Return>", lambda _e: self._on_enter_key())
        self.bind("<Escape>", lambda _e: self._on_escape_key())
        self.bind("<BackSpace>", lambda _e: self._on_backspace())
        self.bind("<Control-z>", lambda _e: self._undo_last_point())
        self.bind("<Control-s>", lambda _e: self._save_and_close())
        self.protocol("WM_DELETE_WINDOW", self._on_close_request)

    def _on_backspace(self) -> None:
        if self.focus_get() == self._name_entry._entry:
            return
        self._undo_last_point()

    def _on_enter_key(self) -> None:
        if self.focus_get() == self._name_entry._entry:
            self._on_name_entry_return()
            return
        if self._selected_lane_idx is not None and not self._current_pts:
            self._select_lane(None)
            return
        self._commit_current_lane()

    def _on_name_entry_return(self) -> None:
        if self._selected_lane_idx is not None and not self._current_pts:
            self._select_lane(None)
        elif len(self._current_pts) >= 3:
            self._commit_current_lane()

    def _on_escape_key(self) -> None:
        if self._current_pts:
            self._cancel_current_polygon()
        elif self._selected_lane_idx is not None:
            self._select_lane(None)
        else:
            self._on_close_request()

    # ── In-Place Selection & Editing ──────────────────────────────────────────

    def _select_lane(self, idx: Optional[int]) -> None:
        if idx is not None and not (0 <= idx < len(self._lanes)):
            idx = None

        self._selected_lane_idx = idx
        self._updating_ui_from_selection = True
        try:
            if idx is not None:
                lane = self._lanes[idx]
                self._name_var.set(lane.get("name", f"Lane {idx + 1}"))
                self._selected_dir = lane.get("expected_direction")
                self._mode_header_lbl.configure(
                    text=f"EDITING LANE #{idx + 1} (IN-PLACE)",
                    text_color=_AMBER,
                )
                self._save_lane_btn.configure(
                    text="Done Editing",
                    fg_color=_AMBER,
                    hover_color=_ACCENT_HOVER,
                )
            else:
                self._name_var.set(f"Lane {len(self._lanes) + 1}")
                self._mode_header_lbl.configure(
                    text="NEW LANE CONFIGURATION",
                    text_color=_TEXT_SEC,
                )
                self._save_lane_btn.configure(
                    text="Save Lane (Enter)",
                    fg_color=_ACCENT,
                    hover_color=_ACCENT_HOVER,
                )
            self._refresh_direction_buttons()
        finally:
            self._updating_ui_from_selection = False

        self._refresh_lanes_list()
        self._render_canvas()

    def _on_name_var_changed(self, *_args) -> None:
        if self._updating_ui_from_selection:
            return
        if self._selected_lane_idx is not None and 0 <= self._selected_lane_idx < len(self._lanes):
            new_name = self._name_var.get().strip() or f"Lane {self._selected_lane_idx + 1}"
            if self._lanes[self._selected_lane_idx].get("name") != new_name:
                self._lanes[self._selected_lane_idx]["name"] = new_name
                self.is_dirty = True
                self._refresh_lanes_list()
                self._render_canvas()

    def _select_direction(self, direction_code: Optional[str]) -> None:
        self._selected_dir = direction_code
        self._refresh_direction_buttons()

        if self._selected_lane_idx is not None and 0 <= self._selected_lane_idx < len(self._lanes):
            if self._lanes[self._selected_lane_idx].get("expected_direction") != direction_code:
                self._lanes[self._selected_lane_idx]["expected_direction"] = direction_code
                self.is_dirty = True
                self._refresh_lanes_list()
                self._render_canvas()

    def _refresh_direction_buttons(self) -> None:
        for code, btn in self._dir_buttons.items():
            if code == self._selected_dir:
                btn.configure(fg_color=_ACCENT, hover_color=_ACCENT_HOVER, text_color="#FFFFFF")
            else:
                btn.configure(fg_color=_SURFACE, hover_color=_CARD_HOVER, text_color=_TEXT_SEC)

    def _refresh_lanes_list(self) -> None:
        for child in self._lanes_scroll.winfo_children():
            child.destroy()

        self._lanes_count_lbl.configure(text=f"SAVED LANES ({len(self._lanes)})")

        if not self._lanes:
            empty_lbl = ctk.CTkLabel(
                self._lanes_scroll,
                text="No lanes defined yet.\nClick on the video frame to draw a polygon.",
                font=ctk.CTkFont(size=12),
                text_color=_TEXT_MUTED,
                justify="center",
            )
            empty_lbl.grid(row=0, column=0, padx=10, pady=24)
            return

        for idx, lane in enumerate(self._lanes):
            is_selected = (idx == self._selected_lane_idx)
            color_bgr = LANE_COLORS_BGR[idx % len(LANE_COLORS_BGR)]
            color_hex = _bgr_to_hex(color_bgr)

            row_card = ctk.CTkFrame(
                self._lanes_scroll,
                fg_color=_CARD_HOVER if is_selected else _SURFACE,
                corner_radius=8,
                border_width=2 if is_selected else 1,
                border_color=_AMBER if is_selected else _BORDER,
            )
            row_card.grid(row=idx, column=0, sticky="ew", padx=6, pady=4)
            row_card.grid_columnconfigure(1, weight=1)

            dot = ctk.CTkFrame(row_card, width=12, height=12, corner_radius=6, fg_color=color_hex)
            dot.grid(row=0, column=0, padx=(10, 8), pady=10)

            dir_str = lane.get("expected_direction") or "ANY"
            pts_count = len(lane.get("points", []))
            prefix = "✎ " if is_selected else ""
            info_lbl = ctk.CTkLabel(
                row_card,
                text=f"{prefix}{lane['name']}  [{dir_str}]  ({pts_count} pts)",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=_AMBER if is_selected else _TEXT,
                anchor="w",
                cursor="hand2",
            )
            info_lbl.grid(row=0, column=1, sticky="ew", pady=8)

            # Bind click on row card and label to select lane for in-place editing
            row_card.bind("<Button-1>", lambda _e, i=idx: self._on_sidebar_lane_click(i))
            info_lbl.bind("<Button-1>", lambda _e, i=idx: self._on_sidebar_lane_click(i))
            dot.bind("<Button-1>", lambda _e, i=idx: self._on_sidebar_lane_click(i))

            del_btn = ctk.CTkButton(
                row_card,
                text="✕",
                width=28,
                height=26,
                corner_radius=6,
                fg_color="transparent",
                hover_color=_RED,
                text_color=_TEXT_SEC,
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda i=idx: self._delete_lane(i),
            )
            del_btn.grid(row=0, column=2, padx=(4, 8), pady=6)

    def _on_sidebar_lane_click(self, idx: int) -> None:
        if self._current_pts:
            self._cancel_current_polygon()
        if self._selected_lane_idx == idx:
            self._select_lane(None)
        else:
            self._select_lane(idx)

    # ── Mouse & Polygon Interaction ───────────────────────────────────────────

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        self._render_canvas()

    def _on_mouse_move(self, event: tk.Event) -> None:
        fx, fy = self._renderer.canvas_to_frame(event.x, event.y)
        self._coord_lbl.configure(
            text=f"Resolution: {self._frame_w}×{self._frame_h} px  |  Cursor: ({fx}, {fy})"
        )
        self._hover_pt = (fx, fy)
        self._hover_vertex_idx = self._renderer.find_nearest_vertex(event.x, event.y, self._current_pts)

        self._close_snap_active = False
        if len(self._current_pts) >= 3 and not self._polygon_closed:
            p0_cx, p0_cy = self._renderer.frame_to_canvas(self._current_pts[0][0], self._current_pts[0][1])
            if math.hypot(event.x - p0_cx, event.y - p0_cy) <= CLOSE_SNAP_RADIUS_PX:
                self._close_snap_active = True

        saved_vertex_hit = (
            self._renderer.find_lane_vertex(event.x, event.y, self._lanes, self._selected_lane_idx)
            if not self._current_pts
            else None
        )
        containing_lane = (
            self._renderer.find_containing_lane(event.x, event.y, self._lanes)
            if not self._current_pts and saved_vertex_hit is None
            else None
        )

        if self._close_snap_active or self._hover_vertex_idx is not None or saved_vertex_hit is not None:
            self._canvas.configure(cursor="hand2")
        elif containing_lane is not None:
            self._canvas.configure(cursor="hand2")
        else:
            self._canvas.configure(cursor="crosshair")

        self._render_canvas()

    def _on_left_press(self, event: tk.Event) -> None:
        self._drag_moved = False

        # 1. If a draft polygon is currently active, handle draft vertex snap/drag/add
        if self._current_pts:
            if len(self._current_pts) >= 3 and not self._polygon_closed:
                p0_cx, p0_cy = self._renderer.frame_to_canvas(self._current_pts[0][0], self._current_pts[0][1])
                if math.hypot(event.x - p0_cx, event.y - p0_cy) <= CLOSE_SNAP_RADIUS_PX:
                    self._polygon_closed = True
                    self._close_snap_active = False
                    self._commit_current_lane()
                    return

            hit_idx = self._renderer.find_nearest_vertex(event.x, event.y, self._current_pts)
            if hit_idx is not None:
                self._dragging_vertex_idx = hit_idx
                return

            if self._polygon_closed:
                return

            fx, fy = self._renderer.canvas_to_frame(event.x, event.y)
            self._current_pts.append([fx, fy])
            self._render_canvas()
            return

        # 2. No active draft polygon: check if user clicked a vertex of an existing saved lane
        lane_vertex = self._renderer.find_lane_vertex(event.x, event.y, self._lanes, self._selected_lane_idx)
        if lane_vertex is not None:
            l_idx, v_idx = lane_vertex
            if self._selected_lane_idx != l_idx:
                self._select_lane(l_idx)
            self._dragging_lane_vertex = (l_idx, v_idx)
            return

        # 3. Check if user clicked inside an existing saved lane polygon
        hit_lane_idx = self._renderer.find_containing_lane(event.x, event.y, self._lanes)
        if hit_lane_idx is not None:
            self._select_lane(hit_lane_idx)
            return

        # 4. Clicked outside all lanes: if a lane was selected, clear selection first
        if self._selected_lane_idx is not None:
            self._select_lane(None)
            return

        # 5. Otherwise start drawing a new lane polygon at the clicked point
        fx, fy = self._renderer.canvas_to_frame(event.x, event.y)
        self._current_pts.append([fx, fy])
        self._render_canvas()

    def _on_left_drag(self, event: tk.Event) -> None:
        if self._dragging_vertex_idx is not None:
            self._drag_moved = True
            fx, fy = self._renderer.canvas_to_frame(event.x, event.y)
            self._current_pts[self._dragging_vertex_idx] = [fx, fy]
            self._coord_lbl.configure(
                text=f"Resolution: {self._frame_w}×{self._frame_h} px  |  Dragging P{self._dragging_vertex_idx + 1}: ({fx}, {fy})"
            )
            self._render_canvas()
            return

        if self._dragging_lane_vertex is not None:
            self._drag_moved = True
            l_idx, v_idx = self._dragging_lane_vertex
            fx, fy = self._renderer.canvas_to_frame(event.x, event.y)
            if 0 <= l_idx < len(self._lanes):
                pts = self._lanes[l_idx].get("points", [])
                if 0 <= v_idx < len(pts):
                    pts[v_idx] = [fx, fy]
                    self.is_dirty = True
            self._coord_lbl.configure(
                text=f"Resolution: {self._frame_w}×{self._frame_h} px  |  Editing Lane #{l_idx + 1} P{v_idx + 1}: ({fx}, {fy})"
            )
            self._render_canvas()

    def _on_left_release(self, _event: tk.Event) -> None:
        self._dragging_vertex_idx = None
        self._dragging_lane_vertex = None

    def _on_double_click(self, _event: tk.Event) -> None:
        if len(self._current_pts) >= 3:
            self._polygon_closed = True
            self._commit_current_lane()

    def _on_right_click(self, _event: tk.Event) -> None:
        if self._current_pts:
            self._undo_last_point()
        elif self._selected_lane_idx is not None:
            self._select_lane(None)

    def _undo_last_point(self) -> None:
        if self._polygon_closed:
            self._polygon_closed = False
            self._render_canvas()
            return
        if self._current_pts:
            self._current_pts.pop()
            self._render_canvas()

    def _cancel_current_polygon(self) -> None:
        self._current_pts.clear()
        self._polygon_closed = False
        self._close_snap_active = False
        self._dragging_vertex_idx = None
        self._render_canvas()

    def _commit_current_lane(self) -> None:
        if self._selected_lane_idx is not None and not self._current_pts:
            self._select_lane(None)
            return

        if len(self._current_pts) < 3:
            return

        lane_name = self._name_var.get().strip() or f"Lane {len(self._lanes) + 1}"
        self._lanes.append({
            "name": lane_name,
            "points": [list(pt) for pt in self._current_pts],
            "expected_direction": self._selected_dir,
        })
        self.is_dirty = True

        self._current_pts.clear()
        self._polygon_closed = False
        self._close_snap_active = False
        self._dragging_vertex_idx = None

        self._updating_ui_from_selection = True
        try:
            self._name_var.set(f"Lane {len(self._lanes) + 1}")
        finally:
            self._updating_ui_from_selection = False

        self._refresh_lanes_list()
        self._render_canvas()

    def _delete_lane(self, index: int) -> None:
        if 0 <= index < len(self._lanes):
            self._lanes.pop(index)
            self.is_dirty = True
            if self._selected_lane_idx == index:
                self._select_lane(None)
            elif self._selected_lane_idx is not None and self._selected_lane_idx > index:
                self._select_lane(self._selected_lane_idx - 1)
            else:
                self._refresh_lanes_list()
                self._render_canvas()

    def _clear_all_lanes(self) -> None:
        if not self._lanes and not self._current_pts:
            return
        if self._lanes:
            self.is_dirty = True
        self._lanes.clear()
        self._current_pts.clear()
        self._polygon_closed = False
        self._select_lane(None)

    def _render_canvas(self) -> None:
        canvas_w = max(1, self._canvas.winfo_width())
        canvas_h = max(1, self._canvas.winfo_height())
        if canvas_w < 10 or canvas_h < 10:
            return

        self._photo = self._renderer.render_photo(
            base_frame=self._base_frame,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            lanes=self._lanes,
            current_pts=self._current_pts,
            hover_pt=self._hover_pt,
            hover_idx=self._hover_vertex_idx,
            drag_idx=self._dragging_vertex_idx,
            close_snap_active=self._close_snap_active,
            polygon_closed=self._polygon_closed,
            selected_lane_idx=self._selected_lane_idx,
        )
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

    # ── Save & Dirty State Confirmation ───────────────────────────────────────

    def _persist_lanes_to_disk(self) -> None:
        lanes_dict: Dict[str, dict] = {}
        for idx, lane in enumerate(self._lanes):
            name = str(lane.get("name") or f"Lane_{idx + 1}")
            pts = [[int(p[0]), int(p[1])] for p in lane.get("points", []) if len(p) >= 2]
            if len(pts) < 3:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            roi = [min(xs), min(ys), max(min(xs) + 1, max(xs)), max(min(ys) + 1, max(ys))]
            raw_dir = lane.get("expected_direction")
            exp_dir = _UI_TO_MODEL_DIR.get(str(raw_dir), "") if raw_dir else ""
            lanes_dict[name] = {
                "roi": roi,
                "points": pts,
                "label_pt": [pts[0][0], pts[0][1]],
                "expected_direction": exp_dir,
                "display_name": name,
                "visible": True,
                "locked": False,
                "order": idx,
            }

        try:
            saved_path = save_camera_lanes(
                camera_id=self._source_id,
                cameras_dir=self._cameras_dir,
                lanes_data=lanes_dict,
                resolution=[self._frame_w, self._frame_h],
                display_name=self._source_id,
            )
            logger.info("Saved %d ROI lane(s) to %s", len(self._lanes), saved_path)
        except Exception as exc:
            logger.error("Failed to save camera ROI file for %s: %s", self._source_id, exc)

        self._initial_lanes = copy.deepcopy(self._lanes)
        self.is_dirty = False
        self._saved = True

    def _save_and_close(self) -> None:
        if len(self._current_pts) >= 3:
            self._commit_current_lane()
        self._persist_lanes_to_disk()
        self._close_window()

    def _on_close_request(self) -> None:
        if not self.is_dirty:
            self._close_window()
            return

        choice = messagebox.askyesnocancel(
            "Kaydedilmemiş Değişiklikler",
            "Kaydedilmemiş değişiklikler var. Kaydetmek istiyor musunuz?",
            parent=self,
        )
        if choice is True:
            self._save_and_close()
        elif choice is False:
            self._lanes = copy.deepcopy(self._initial_lanes)
            self.is_dirty = False
            self._close_window()
        else:
            # Cancel (None) -> keep editor open
            return

    def _close_window(self) -> None:
        self.grab_release()
        self.destroy()
        if self._owns_root and self._hidden_root is not None:
            self._hidden_root.destroy()

    def get_lanes(self) -> List[dict]:
        return list(self._lanes)


class ROISelector:
    """Adapter façade used by VideoLoop and Launcher to open the ROI editor."""

    def __init__(
        self,
        config_path: str = "config.json",
        source_id: str = "default",
        parent: Optional[tk.Misc] = None,
        cameras_dir: Optional[os.PathLike] = None,
    ) -> None:
        self.config_path = config_path
        self.source_id = source_id
        self.parent = parent
        self.cameras_dir = cameras_dir

    def select_rois(self, frame: np.ndarray) -> List[dict]:
        editor = ROISelectorWindow(
            parent=self.parent,
            frame=frame,
            config_path=self.config_path,
            source_id=self.source_id,
            cameras_dir=self.cameras_dir,
        )
        if editor._owns_root and editor._hidden_root is not None:
            editor.wait_window()
        else:
            editor.wait_window()
        return editor.get_lanes()


def run(
    config_path: str = "config.json",
    camera_id: str = "default",
    stream_url: Optional[str] = None,
    cameras_dir: Optional[os.PathLike] = None,
    video_path: Optional[str] = None,
    parent: Optional[tk.Misc] = None,
) -> List[dict]:
    """Capture a single reference frame from *stream_url* or *video_path* and open the ROI editor."""
    frame: Optional[np.ndarray] = None
    source = stream_url or video_path
    if source:
        if "mjpeg" in source.lower():
            from traffic_analyzer.infrastructure.video_loop import _iter_mjpeg
            for f in _iter_mjpeg(source):
                frame = f
                break
        else:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                ret, f = cap.read()
                if ret and f is not None:
                    frame = f
            cap.release()

    if frame is None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    selector = ROISelector(
        config_path=str(config_path),
        source_id=camera_id,
        parent=parent,
        cameras_dir=cameras_dir,
    )
    return selector.select_rois(frame)
