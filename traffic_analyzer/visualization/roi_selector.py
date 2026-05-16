"""
traffic_analyzer/visualization/roi_selector.py

ROI Selector — CTkToplevel window for drawing lane polygons on a camera frame.

Layout:
  Left  — dark canvas field (resizable, canvas centred with place()).
  Right — fixed sidebar (330 px): header + 3-step stepper + scrollable bento
          cards + pinned Save / Exit buttons.

Separation of concerns:
  ROIRenderer     — all OpenCV / PIL drawing  (roi_renderer.py)
  LaneEditorState — all mutable editor state  (roi_state.py)
  ROISelectorWindow — UI layout, event wiring, mode transitions  (this file)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
import customtkinter as ctk

from traffic_analyzer.infrastructure.config_loader import (
    load_config, load_camera_lanes, save_camera_lanes,
)
from traffic_analyzer.visualization.roi_state    import LaneEditorState
from traffic_analyzer.visualization.roi_renderer import ROIRenderer, _lane_hex


def _load_user_fonts(family_stem: str) -> None:
    """Load user-installed TTF files into GDI so tkinter can use them.

    Fonts installed per-user land in AppData/Local/Microsoft/Windows/Fonts
    but are NOT visible to tkinter's font.families() until explicitly loaded
    via AddFontResourceExW.  Call this before the first CTk window is created.
    """
    try:
        import ctypes
        font_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts"
        for ttf in font_dir.glob(f"{family_stem}*.ttf"):
            ctypes.windll.gdi32.AddFontResourceExW(str(ttf), 0x10, 0)
    except Exception:
        pass


_load_user_fonts("PlusJakartaSans")

# ── Midnight Navy palette — matches launcher theme ────────────────────────────
_BG       = "#0c0e1a"   # Midnight navy  — sidebar header / app background
_SURFACE  = "#10132a"   # Deep navy      — sidebar surface
_CARD     = "#161930"   # Navy card      — bento card background
_CARD_HI  = "#1d2040"   # Elevated navy  — hover / raised card
_BORDER_S = "#1e2244"   # Barely-visible — subtle border
_BORDER   = "#252a50"   # Blue-slate     — visible border
_FG       = "#d6daf8"   # Blue-white     — primary text
_FG_MED   = "#7880b8"   # Mid blue-grey  — secondary text
_FG_DIM   = "#3a3f6a"   # Dim blue       — hint / disabled
_ACCENT   = "#6070e8"   # Soft indigo    — primary CTA
_ACCENT_H = "#5060d8"   # Dark indigo    — hover
_TEAL     = "#4db8cc"   # Soft teal      — secondary accent
_SEL_BG   = "#111535"   # Indigo-navy    — selected background
_GREEN    = "#3dd68c"   # Soft emerald   — success
_AMBER    = "#e8a020"   # Soft amber     — warning
_RED      = "#e05575"   # Soft rose-red  — danger / delete

# Badge background tokens (dark tint bg, vivid text)
_GREEN_L  = "#0a2218"
_AMBER_L  = "#251a05"
_VIOLET_L = "#12103a"
_ACCENT_L = "#111535"

# ── Typography ────────────────────────────────────────────────────────────────
_PF = "Plus Jakarta Sans"
_FT = (_PF, 16, "bold")   # window title
_FH = (_PF, 13, "bold")   # section heading
_FB = (_PF, 11)            # body
_FS = (_PF, 10)            # small / label
_FC = ("Consolas", 10)     # mono / keyboard hints

SIDEBAR_W = 360

_DIRECTION_MAP = {
    "bottom_to_top": "↑",
    "top_to_bottom": "↓",
    "right_to_left": "←",
    "left_to_right": "→",
    "":              "—",
}
_DIRECTIONS = [
    ("↑", "Bottom → Top",  "bottom_to_top"),
    ("↓", "Top → Bottom",  "top_to_bottom"),
    ("←", "Right → Left",  "right_to_left"),
    ("→", "Left → Right",  "left_to_right"),
    ("—", "Skip / None",   ""),
]


# ── Helper dialogs ────────────────────────────────────────────────────────────

def _make_dialog(parent: tk.Misc, title: str, w: int, h: int) -> ctk.CTkToplevel:
    top = ctk.CTkToplevel(parent)
    top.title(title)
    top.configure(fg_color=_BG)
    top.resizable(False, False)
    top.attributes("-topmost", True)
    top.update_idletasks()
    sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
    top.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
    top.grab_set()
    return top


def _confirm_dialog(parent: tk.Misc, title: str, msg: str) -> bool:
    result = [False]
    top    = _make_dialog(parent, title, 400, 154)

    ctk.CTkLabel(top, text=msg, font=_FB, text_color=_FG_MED,
                 wraplength=360, justify="left").pack(pady=(24, 14), padx=24)

    bf = ctk.CTkFrame(top, fg_color=_BG)
    bf.pack(fill="x", padx=24, pady=(0, 18), side="bottom")

    def _yes():
        result[0] = True
        top.destroy()

    # Danger action: indigo card bg, red border/text
    ctk.CTkButton(bf, text="Exit anyway", font=_FB, width=110, height=34,
                  corner_radius=8, fg_color=_CARD, hover_color="#2a0f1a",
                  border_color=_RED, border_width=1,
                  text_color=_RED, command=_yes).pack(side="right", padx=(8, 0))
    # Primary action: indigo bg, white text
    ctk.CTkButton(bf, text="Keep editing", font=_FB, width=120, height=34,
                  corner_radius=8, fg_color=_ACCENT, hover_color=_ACCENT_H,
                  text_color="#ffffff", command=top.destroy).pack(side="right")

    top.protocol("WM_DELETE_WINDOW", top.destroy)
    top.wait_window()
    return result[0]


def _error_dialog(parent: tk.Misc, msg: str) -> None:
    top = _make_dialog(parent, "Notice", 420, 134)
    ctk.CTkLabel(top, text=msg, font=_FB, text_color=_FG_MED,
                 wraplength=380, justify="left").pack(pady=(24, 12), padx=24)
    ctk.CTkButton(top, text="OK", font=_FB, width=90, height=32,
                  corner_radius=8, fg_color=_ACCENT, hover_color=_ACCENT_H,
                  text_color="#ffffff", command=top.destroy).pack()
    top.protocol("WM_DELETE_WINDOW", top.destroy)
    top.wait_window()


# ── Frame capture ─────────────────────────────────────────────────────────────

def _grab_frame(source: str) -> Optional[np.ndarray]:
    """Grab a single BGR frame from a video file, camera index, or MJPEG URL."""
    if source.startswith("http"):
        import requests, urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        try:
            with requests.get(source, stream=True, verify=False, timeout=10) as r:
                r.raise_for_status()
                buf = b""
                for chunk in r.iter_content(chunk_size=4096):
                    buf += chunk
                    s = buf.find(b"\xff\xd8")
                    e = buf.find(b"\xff\xd9")
                    if s != -1 and e != -1 and e > s:
                        frame = cv2.imdecode(
                            np.frombuffer(buf[s:e + 2], np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            return frame
        except Exception as ex:
            print(f"[ROI] stream error: {ex}")
        return None

    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ── ROI Selector Window ───────────────────────────────────────────────────────

class ROISelectorWindow(ctk.CTkToplevel):
    """
    ROI selector — draws lane polygons on a camera frame.

    Layout: Left = dark canvas field (fill+expand).
            Right = fixed sidebar (330 px).

    Rendering is fully delegated to ROIRenderer.
    Mutable editor state lives in LaneEditorState.
    This class owns UI layout, event wiring, and mode transitions only.
    """

    def __init__(
        self,
        parent:        tk.Misc,
        frame:         np.ndarray,
        config_path:   str,
        camera_id:     str,
        cameras_dir:   Path,
        cam_label:     str = "",
        initial_lanes: list[dict] | None = None,
    ):
        super().__init__(parent)

        sw = parent.winfo_screenwidth()
        sh = parent.winfo_screenheight()
        vh, vw = frame.shape[:2]

        # Scale to fill as much of the screen as possible (capped at 4×)
        max_cw = sw - SIDEBAR_W - 6
        max_ch = sh - 72
        scale  = min(max_cw / vw, max_ch / vh)
        scale  = min(max(scale, 0.30), 4.0)

        self._scale = scale
        self._cw    = int(vw * scale)
        self._ch    = int(vh * scale)
        self._base  = cv2.resize(frame, (self._cw, self._ch),
                                 interpolation=cv2.INTER_LINEAR)

        self._config_path = config_path
        self._camera_id   = camera_id
        self._cameras_dir = cameras_dir
        self._cam_label   = cam_label

        # State and renderer — the two extracted collaborators
        self._state    = LaneEditorState(
            polygons=list(initial_lanes) if initial_lanes else [],
        )
        self._renderer = ROIRenderer(scale)

        self._photo:    Optional = None
        self._after_id: Optional[str] = None

        win_w = max(self._cw + SIDEBAR_W, 890)
        win_h = max(self._ch, 580)
        win_x = max((sw - win_w) // 2, 0)
        win_y = max((sh - win_h) // 2, 0)

        self.title("ROI Selector")
        self.configure(fg_color=_BG)
        self.geometry(f"{win_w}x{win_h}+{win_x}+{win_y}")
        self.minsize(890, 580)
        self.resizable(True, True)
        self.attributes("-topmost", True)

        self._build()
        self._refresh_sidebar()
        self._loop()
        self.focus_force()
        self.bind("<Key>", self._on_key)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # CTkToplevel's WM may override the initial geometry — reapply after 60 ms
        self.after(60, lambda: self.geometry(f"{win_w}x{win_h}+{win_x}+{win_y}"))  # reapply after CTk WM override

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        # Canvas field — dark background, canvas centred with place()
        self._canvas_frame = tk.Frame(self, bg="#040408")
        self._canvas_frame.pack(side="left", fill="both", expand=True)

        self._canvas = tk.Canvas(
            self._canvas_frame,
            width=self._cw, height=self._ch,
            bg="#040408", highlightthickness=0, cursor="crosshair",
        )
        self._canvas.place(relx=0.5, rely=0.5, anchor="center")
        self._canvas_img = self._canvas.create_image(0, 0, anchor="nw")

        self._canvas.bind("<Button-1>",        self._on_lclick)
        self._canvas.bind("<B1-Motion>",       self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_lrelease)
        self._canvas.bind("<Button-3>",        self._on_rclick)
        self._canvas.bind("<Motion>",          self._on_move)

        # Hairline divider between canvas and sidebar
        tk.Frame(self, width=1, bg=_BORDER_S).pack(side="left", fill="y")

        # Sidebar — fixed width
        self._sidebar = tk.Frame(self, width=SIDEBAR_W - 1, bg=_SURFACE)
        self._sidebar.pack(side="left", fill="y")
        self._sidebar.pack_propagate(False)

        # Sidebar header
        hdr      = tk.Frame(self._sidebar, bg=_BG)
        hdr.pack(fill="x")
        name_row = tk.Frame(hdr, bg=_BG)
        name_row.pack(fill="x", padx=18, pady=(16, 0))

        tk.Label(name_row, text="ROI Selector", font=_FT,
                 fg=_ACCENT, bg=_BG, anchor="w").pack(side="left")

        self._mode_badge = tk.Label(
            name_row, text="● DRAWING",
            font=(_PF, 7, "bold"),
            fg=_ACCENT, bg=_ACCENT_L, padx=8, pady=3, relief="flat",
        )
        self._mode_badge.pack(side="right")

        if self._cam_label:
            tk.Label(hdr, text=self._cam_label,
                     font=_FS, fg=_FG_DIM, bg=_BG, anchor="w").pack(
                fill="x", padx=18, pady=(2, 10))
        else:
            tk.Frame(hdr, height=8, bg=_BG).pack()

        # 3-step progress stepper
        self._stepper_frame = tk.Frame(self._sidebar, bg=_BG)
        self._stepper_frame.pack(fill="x")
        self._build_stepper()

        tk.Frame(self._sidebar, height=1, bg=_BORDER_S).pack(fill="x")

        # Scrollable bento content area
        self._scroll = ctk.CTkScrollableFrame(
            self._sidebar,
            fg_color=_SURFACE, corner_radius=0,
            scrollbar_button_color=_BORDER,
            scrollbar_button_hover_color=_FG_DIM,
        )
        self._scroll.pack(fill="both", expand=True, pady=(4, 0))

        # Action buttons pinned to sidebar bottom
        tk.Frame(self._sidebar, height=1, bg=_BORDER_S).pack(fill="x", side="bottom")
        bot = tk.Frame(self._sidebar, bg=_BG, height=110)
        bot.pack(fill="x", side="bottom")
        bot.pack_propagate(False)

        # Primary CTA: Save & Close — indigo pill
        self._save_btn = ctk.CTkButton(
            bot,
            text="Save & Close",
            font=(_PF, 11, "bold"),
            height=44, corner_radius=22,
            fg_color=_ACCENT, hover_color=_ACCENT_H,
            text_color="#ffffff",
            command=self._save_and_close,
        )
        self._save_btn.pack(fill="x", padx=16, pady=(12, 6))

        # Secondary: Exit without saving — muted ghost pill
        ctk.CTkButton(
            bot,
            text="Exit without Saving",
            font=_FB, height=36, corner_radius=22,
            fg_color="transparent", hover_color=_CARD_HI,
            border_color=_BORDER, border_width=1,
            text_color=_FG_MED,
            command=self._exit_without_saving,
        ).pack(fill="x", padx=16)

    # ── Progress stepper ──────────────────────────────────────────────────────

    def _build_stepper(self):
        """Rebuild the 3-step horizontal stepper in the sidebar header."""
        for w in self._stepper_frame.winfo_children():
            w.destroy()

        steps = [
            ("drawing",    "1", "Draw"),
            ("naming",     "2", "Name"),
            ("direction",  "3", "Direction"),
            ("label_drag", "4", "Label"),
        ]
        container = tk.Frame(self._stepper_frame, bg=_BG)
        container.pack(fill="x", padx=18, pady=(8, 12))

        for i, (step_id, num, label) in enumerate(steps):
            is_done   = self._step_done(step_id)
            is_active = (self._state.mode == step_id)

            # Connector line between step circles
            if i > 0:
                prev_done  = self._step_done(steps[i - 1][0])
                line_color = _GREEN if prev_done else _BORDER
                tk.Frame(container, height=2, bg=line_color).pack(
                    side="left", fill="x", expand=True, pady=11)

            group = tk.Frame(container, bg=_BG)
            group.pack(side="left")

            circle_bg = _GREEN if is_done else (_ACCENT if is_active else _CARD_HI)
            circle_fg = "#0c0e1a" if is_done else (_FG if is_active else _FG_DIM)
            num_text  = "✓" if is_done else num

            circ = tk.Canvas(group, width=26, height=26, bg=_BG, highlightthickness=0)
            circ.pack()
            circ.create_oval(1, 1, 25, 25, fill=circle_bg, outline="")
            if is_active and not is_done:
                circ.create_oval(2, 2, 24, 24, fill=circle_bg, outline=_ACCENT, width=2)
            circ.create_text(13, 13, text=num_text,
                             font=(_PF, 9, "bold"), fill=circle_fg)

            lbl_color = _GREEN if is_done else (_FG if is_active else _FG_DIM)
            tk.Label(group, text=label, font=(_PF, 7, "bold"),
                     fg=lbl_color, bg=_BG).pack()

    def _step_done(self, step_id: str) -> bool:
        order = ["drawing", "naming", "direction", "label_drag"]
        try:
            return order.index(self._state.mode) > order.index(step_id)
        except ValueError:
            return False

    # ── Incremental drawing-status update ─────────────────────────────────────

    def _update_drawing_status(self):
        """Update only the status dot, text, and conditional buttons — no full rebuild."""
        n_pts = len(self._state.current_pts)
        ready = n_pts >= 3

        if n_pts == 0:
            dot_color    = _FG_DIM
            status_text  = "Click the frame to place points"
            status_color = _FG_DIM
        elif not ready:
            dot_color    = _AMBER
            status_text  = f"{n_pts} point{'s' if n_pts > 1 else ''} — need {3 - n_pts} more"
            status_color = _FG_MED
        else:
            dot_color    = _GREEN
            status_text  = f"{n_pts} points — ready to confirm"
            status_color = _FG_MED

        self._draw_dot.configure(text_color=dot_color)
        self._draw_status.configure(text=status_text, text_color=status_color)

        # Toggle button appearance without geometry changes (prevents flicker)
        if ready:
            self._draw_confirm_btn.configure(
                state="normal", fg_color=_SEL_BG,
                border_color=_ACCENT, text_color=_ACCENT,
            )
        else:
            self._draw_confirm_btn.configure(
                state="disabled", fg_color=_CARD,
                border_color=_BORDER_S, text_color=_FG_DIM,
            )

        if n_pts:
            self._draw_reset_btn.configure(
                state="normal", border_color=_BORDER, text_color=_FG_MED,
            )
        else:
            self._draw_reset_btn.configure(
                state="disabled", border_color=_BORDER_S, text_color=_FG_DIM,
            )

    # ── Bento card helper ─────────────────────────────────────────────────────

    def _bento(self, title: str | None = None) -> ctk.CTkFrame:
        card = ctk.CTkFrame(
            self._scroll,
            fg_color=_CARD, corner_radius=10,
            border_color=_BORDER_S, border_width=1,
        )
        card.pack(fill="x", padx=13, pady=5)
        if title:
            ctk.CTkLabel(card, text=title,
                         font=("Segoe UI", 8, "bold"),
                         text_color=_FG_DIM).pack(
                anchor="w", padx=16, pady=(12, 0))
            ctk.CTkFrame(card, height=1, fg_color=_BORDER_S,
                         corner_radius=0).pack(fill="x", padx=16, pady=(6, 8))
        return card

    # ── Sidebar panels ────────────────────────────────────────────────────────

    def _refresh_sidebar(self):
        badge_map = {
            "drawing":    ("● DRAWING",   _ACCENT, _ACCENT_L),
            "naming":     ("✎ NAMING",    _AMBER,  _AMBER_L),
            "direction":  ("◈ DIRECTION", _TEAL,   _VIOLET_L),
            "label_drag": ("⤢ LABEL",     _GREEN,  _GREEN_L),
        }
        text, color, bg = badge_map.get(self._state.mode, ("●", _FG_DIM, _BG))
        self._mode_badge.configure(text=text, fg=color, bg=bg)

        self._build_stepper()

        for w in self._scroll.winfo_children():
            w.destroy()

        if self._state.mode == "drawing":
            self._panel_drawing()
        elif self._state.mode == "naming":
            self._panel_naming()
        elif self._state.mode == "direction":
            self._panel_direction()
        elif self._state.mode == "label_drag":
            self._panel_label_drag()

    # ── Drawing panel ─────────────────────────────────────────────────────────

    def _panel_drawing(self):
        n_lanes = len(self._state.polygons)

        # Defined lanes card — fully static within drawing mode
        card = self._bento(f"DEFINED LANES  ({n_lanes})")
        if not n_lanes:
            ctk.CTkLabel(card,
                         text="No lanes yet.\nClick the video to start placing points.",
                         font=_FS, text_color=_FG_DIM,
                         wraplength=284, justify="left").pack(
                padx=16, pady=(0, 12), anchor="w")
        else:
            for i, poly in enumerate(self._state.polygons):
                self._lane_row(card, i, poly)
            tk.Frame(card, height=6, bg=_CARD).pack()

        # Current polygon card — status elements stored for incremental updates
        cur_card   = self._bento(f"DRAWING — LANE {n_lanes + 1}")
        status_row = ctk.CTkFrame(cur_card, fg_color="transparent", corner_radius=0)
        status_row.pack(fill="x", padx=16, pady=(0, 8))

        self._draw_dot = ctk.CTkLabel(status_row, text="●", font=(_PF, 10),
                                      text_color=_FG_DIM)
        self._draw_dot.pack(side="left", padx=(0, 6))

        self._draw_status = ctk.CTkLabel(status_row, text="", font=_FS,
                                         text_color=_FG_DIM,
                                         wraplength=240, justify="left")
        self._draw_status.pack(side="left", anchor="w")

        self._draw_confirm_btn = ctk.CTkButton(
            cur_card,
            text="✓  Confirm Lane",
            font=(_PF, 11, "bold"), height=38, corner_radius=10,
            fg_color=_SEL_BG, hover_color=_CARD_HI,
            border_color=_ACCENT, border_width=2,
            text_color=_ACCENT,
            command=self._start_naming,
        )
        self._draw_confirm_btn.pack(fill="x", padx=16, pady=(0, 4))

        self._draw_reset_btn = ctk.CTkButton(
            cur_card,
            text="Reset Current",
            font=_FS, height=32, corner_radius=8,
            fg_color="transparent", hover_color=_CARD_HI,
            border_color=_BORDER, border_width=1,
            text_color=_FG_MED,
            command=self._reset_current,
        )
        self._draw_reset_btn.pack(fill="x", padx=16, pady=(0, 4))

        tk.Frame(cur_card, height=4, bg=_CARD).pack()
        self._update_drawing_status()

        if n_lanes:
            danger_card = self._bento()
            ctk.CTkButton(
                danger_card,
                text="Reset All Lanes",
                font=_FS, height=30, corner_radius=7,
                fg_color="transparent", hover_color="#2a0f1a",
                border_color=_RED, border_width=1,
                text_color=_RED,
                command=self._reset_all,
            ).pack(fill="x", padx=16, pady=(10, 10))

        sh_card = self._bento("KEYBOARD SHORTCUTS")
        shortcuts = [
            ("L-Click",  "Add a point"),
            ("Drag",     "Move a point"),
            ("R-Click",  "Undo last point"),
            ("Enter",    "Confirm  (≥ 3 pts)"),
            ("Esc / r",  "Reset current"),
            ("R",        "Reset all lanes"),
            ("S",        "Save & close"),
        ]
        for key, desc in shortcuts:
            row = ctk.CTkFrame(sh_card, fg_color="transparent", corner_radius=0)
            row.pack(fill="x", padx=16, pady=2)
            ctk.CTkLabel(row, text=key, font=_FC,
                         text_color=_TEAL, width=64, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=desc, font=_FS,
                         text_color=_FG_DIM, anchor="w").pack(side="left")
        tk.Frame(sh_card, height=6, bg=_CARD).pack()

    def _lane_row(self, parent: ctk.CTkFrame, idx: int, poly: dict):
        color_hex = _lane_hex(idx)
        dir_arrow = _DIRECTION_MAP.get(poly.get("expected_direction", ""), "—")

        row = ctk.CTkFrame(parent, fg_color=_CARD_HI, corner_radius=8,
                           border_color=_BORDER_S, border_width=1, height=38)
        row.pack(fill="x", padx=16, pady=3)
        row.pack_propagate(False)

        dot = tk.Canvas(row, width=10, height=10, bg=_CARD_HI, highlightthickness=0)
        dot.pack(side="left", padx=(12, 6), pady=14)
        dot.create_oval(1, 1, 9, 9, fill=color_hex, outline="")

        ctk.CTkLabel(row, text=poly["name"], font=_FB,
                     text_color=_FG, anchor="w").pack(
            side="left", fill="x", expand=True)

        dir_badge = ctk.CTkFrame(row, fg_color=_SEL_BG, corner_radius=5,
                                 border_color=_BORDER_S, border_width=1,
                                 width=28, height=22)
        dir_badge.pack(side="left", padx=4)
        dir_badge.pack_propagate(False)
        ctk.CTkLabel(dir_badge, text=dir_arrow,
                     font=(_PF, 11), text_color=_TEAL).place(
            relx=0.5, rely=0.5, anchor="center")

        ctk.CTkButton(
            row, text="✕", font=(_PF, 9),
            width=26, height=26, corner_radius=6,
            fg_color="transparent", hover_color="#2a0f1a",
            text_color=_FG_DIM,
            command=lambda i=idx: self._delete_lane(i),
        ).pack(side="right", padx=(2, 8))

    # ── Naming panel ──────────────────────────────────────────────────────────

    def _panel_naming(self):
        card = self._bento(f"NAME — LANE {len(self._state.polygons) + 1}")

        ctk.CTkLabel(card, text="What should this lane be called?",
                     font=_FS, text_color=_FG_DIM).pack(
            padx=16, pady=(0, 8), anchor="w")

        self._name_var   = ctk.StringVar(value=self._state.temp_name)
        self._name_entry = ctk.CTkEntry(
            card,
            textvariable=self._name_var,
            placeholder_text="e.g.  Serit 1",
            font=("Segoe UI", 11),
            height=40, corner_radius=9,
            fg_color=_CARD_HI,
            border_color=_ACCENT,
            border_width=2,
            text_color=_FG,
        )
        self._name_entry.pack(fill="x", padx=16, pady=(0, 4))
        self._name_entry.after(80, self._name_entry.focus_set)
        self._name_entry.bind("<Return>", lambda _: self._confirm_name())
        self._name_entry.bind("<Escape>", lambda _: self._cancel_mode())

        self._name_err = ctk.CTkLabel(card, text="", font=_FS, text_color=_RED)
        self._name_err.pack(padx=16, anchor="w")

        ctk.CTkButton(
            card,
            text="Next  →",
            font=(_PF, 10, "bold"), height=38, corner_radius=19,
            fg_color=_ACCENT, hover_color=_ACCENT_H,
            text_color="#ffffff",
            command=self._confirm_name,
        ).pack(fill="x", padx=16, pady=(10, 4))

        ctk.CTkButton(
            card, text="Cancel",
            font=_FS, height=30, corner_radius=7,
            fg_color="transparent", hover_color=_CARD_HI,
            border_color=_BORDER, border_width=1,
            text_color=_FG_MED,
            command=self._cancel_mode,
        ).pack(fill="x", padx=16, pady=(0, 12))

    # ── Direction panel ───────────────────────────────────────────────────────

    def _panel_direction(self):
        card = self._bento("TRAFFIC DIRECTION")

        ctk.CTkLabel(card, text=f"Lane:  {self._state.temp_name}",
                     font=_FB, text_color=_FG_MED).pack(
            padx=16, pady=(0, 4), anchor="w")
        ctk.CTkLabel(card,
                     text="Which direction do vehicles travel through this lane?",
                     font=_FS, text_color=_FG_DIM,
                     wraplength=266, justify="left").pack(
            padx=16, pady=(0, 10), anchor="w")

        for arrow, label, value in _DIRECTIONS:
            sel = (self._state.temp_dir == value)
            btn_card = ctk.CTkFrame(
                card,
                fg_color=_SEL_BG  if sel else _CARD_HI,
                corner_radius=9,
                border_color=_ACCENT if sel else _BORDER_S,
                border_width=2 if sel else 1,
                height=42, cursor="hand2",
            )
            btn_card.pack(fill="x", padx=16, pady=3)
            btn_card.pack_propagate(False)

            inner = ctk.CTkFrame(btn_card, fg_color="transparent", corner_radius=0)
            inner.pack(fill="both", expand=True, padx=13)

            ctk.CTkLabel(inner, text=arrow,
                         font=("Segoe UI", 15, "bold"),
                         text_color=_ACCENT if sel else _FG_MED,
                         width=28).pack(side="left")
            ctk.CTkLabel(inner, text=label, font=_FB,
                         text_color=_FG if sel else _FG_DIM,
                         anchor="w").pack(side="left", padx=8)

            def _bind(w: tk.Widget, v: str = value):
                w.bind("<Button-1>", lambda _e: self._select_direction(v))
                for ch in w.winfo_children():
                    ch.bind("<Button-1>", lambda _e, val=v: self._select_direction(val))
                    for gc in ch.winfo_children():
                        gc.bind("<Button-1>", lambda _e, val=v: self._select_direction(val))
            _bind(btn_card)

        ctk.CTkButton(
            card,
            text="Add Lane",
            font=(_PF, 10, "bold"), height=38, corner_radius=19,
            fg_color=_ACCENT, hover_color=_ACCENT_H,
            text_color="#ffffff",
            command=self._confirm_direction,
        ).pack(fill="x", padx=16, pady=(14, 4))

        ctk.CTkButton(
            card, text="← Back",
            font=_FS, height=30, corner_radius=7,
            fg_color="transparent", hover_color=_CARD_HI,
            border_color=_BORDER, border_width=1,
            text_color=_FG_MED,
            command=self._back_to_naming,
        ).pack(fill="x", padx=16, pady=(0, 12))

    # ── Label drag panel ──────────────────────────────────────────────────────

    def _panel_label_drag(self):
        poly = self._state.polygons[-1] if self._state.polygons else {}
        name = poly.get("name", "")
        card = self._bento(f"POSITION LABEL — {name}")

        ctk.CTkLabel(
            card,
            text="Drag the label on the canvas to\nplace it wherever you want.",
            font=_FS, text_color=_FG_DIM,
            wraplength=284, justify="left",
        ).pack(padx=16, pady=(0, 12), anchor="w")

        ctk.CTkButton(
            card,
            text="✓  Done",
            font=(_PF, 10, "bold"), height=38, corner_radius=19,
            fg_color=_GREEN, hover_color="#2ab870",
            text_color="#0a2218",
            command=self._confirm_label,
        ).pack(fill="x", padx=16, pady=(0, 12))

        sh_card = self._bento("KEYBOARD SHORTCUTS")
        for key, desc in [("Drag", "Move label"), ("Enter / Esc", "Confirm position")]:
            row = ctk.CTkFrame(sh_card, fg_color="transparent", corner_radius=0)
            row.pack(fill="x", padx=16, pady=2)
            ctk.CTkLabel(row, text=key,  font=_FC, text_color=_TEAL,
                         width=80, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=desc, font=_FS, text_color=_FG_DIM,
                         anchor="w").pack(side="left")
        tk.Frame(sh_card, height=6, bg=_CARD).pack()

    # ── Mode transitions ──────────────────────────────────────────────────────

    def _start_naming(self):
        if len(self._state.current_pts) < 3:
            return
        self._state.mode      = "naming"
        self._state.temp_name = f"Serit {len(self._state.polygons) + 1}"
        self._state.temp_dir  = ""
        self._refresh_sidebar()

    def _confirm_name(self):
        name = self._name_var.get().strip()
        if not name:
            self._name_err.configure(text="Name cannot be empty.")
            return
        if name in {p["name"] for p in self._state.polygons}:
            self._name_err.configure(text=f"'{name}' is already used.")
            return
        self._state.temp_name = name
        self._state.mode      = "direction"
        self._refresh_sidebar()

    def _back_to_naming(self):
        self._state.mode = "naming"
        self._refresh_sidebar()

    def _cancel_mode(self):
        self._state.mode = "drawing"
        self._refresh_sidebar()
        self.focus_force()

    def _select_direction(self, value: str):
        self._state.temp_dir = value
        self._refresh_sidebar()

    def _confirm_direction(self):
        xs = [p[0] for p in self._state.current_pts]
        ys = [p[1] for p in self._state.current_pts]
        self._state.polygons.append({
            "name":               self._state.temp_name,
            "pts":                list(self._state.current_pts),
            "expected_direction": self._state.temp_dir,
            "label_pt":           [min(xs), min(ys)],
        })
        self._state.current_pts  = []
        self._state.drag_idx     = None
        self._state.hover_idx    = None
        self._state.label_dragging = False
        self._state.mode         = "label_drag"
        self._refresh_sidebar()
        self.focus_force()

    def _confirm_label(self):
        """Finalise label position and return to drawing mode."""
        self._state.label_dragging = False
        self._state.mode           = "drawing"
        self._canvas.configure(cursor="crosshair")
        self._refresh_sidebar()
        self.focus_force()

    # ── Lane management ───────────────────────────────────────────────────────

    def _delete_lane(self, idx: int):
        if 0 <= idx < len(self._state.polygons):
            self._state.polygons.pop(idx)
            self._refresh_sidebar()

    def _reset_current(self):
        self._state.current_pts = []
        self._state.drag_idx    = None
        self._state.hover_idx   = None
        self._update_drawing_status()

    def _reset_all(self):
        self._state.polygons    = []
        self._state.current_pts = []
        self._state.drag_idx    = None
        self._state.hover_idx   = None
        self._refresh_sidebar()

    # ── Mouse events ──────────────────────────────────────────────────────────

    def _on_lclick(self, e):
        if self._state.mode == "label_drag":
            self._state.label_dragging = True
            ox, oy = self._renderer.to_orig(e.x, e.y)
            self._state.polygons[-1]["label_pt"] = [ox, oy]
            return
        if self._state.mode != "drawing":
            return
        idx = self._renderer.nearest(e.x, e.y, self._state.current_pts)
        if idx is not None:
            self._state.drag_idx = idx
        else:
            self._state.current_pts.append(self._renderer.to_orig(e.x, e.y))
            self._update_drawing_status()

    def _on_drag(self, e):
        if self._state.mode == "label_drag" and self._state.label_dragging:
            ox, oy = self._renderer.to_orig(e.x, e.y)
            self._state.polygons[-1]["label_pt"] = [ox, oy]
            return
        if self._state.mode != "drawing" or self._state.drag_idx is None:
            return
        if self._state.drag_idx >= len(self._state.current_pts):
            self._state.drag_idx = None
            return
        self._state.current_pts[self._state.drag_idx] = self._renderer.to_orig(e.x, e.y)

    def _on_lrelease(self, _e):
        self._state.drag_idx     = None
        self._state.label_dragging = False

    def _on_rclick(self, _e):
        if self._state.mode != "drawing" or not self._state.current_pts:
            return
        self._state.current_pts.pop()
        if (self._state.drag_idx is not None and
                self._state.drag_idx >= len(self._state.current_pts)):
            self._state.drag_idx = None
        self._update_drawing_status()

    def _on_move(self, e):
        if self._state.mode == "label_drag":
            if self._canvas["cursor"] != "fleur":
                self._canvas.configure(cursor="fleur")
            return
        if self._state.mode != "drawing":
            return
        prev                  = self._state.hover_idx
        self._state.hover_idx = self._renderer.nearest(e.x, e.y, self._state.current_pts)
        cur = "fleur" if self._state.hover_idx is not None else "crosshair"
        if self._canvas["cursor"] != cur:
            self._canvas.configure(cursor=cur)
        if prev != self._state.hover_idx:
            self._render()

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def _on_key(self, e):
        k = e.keysym
        if k in ("Return", "space"):
            if   self._state.mode == "drawing":    self._start_naming()
            elif self._state.mode == "naming":     self._confirm_name()
            elif self._state.mode == "direction":  self._confirm_direction()
            elif self._state.mode == "label_drag": self._confirm_label()
        elif k == "Escape":
            if self._state.mode in ("naming", "direction"):
                self._cancel_mode()
            elif self._state.mode == "label_drag":
                self._confirm_label()
            elif self._state.mode == "drawing" and self._state.current_pts:
                self._reset_current()
        elif k == "r" and self._state.mode == "drawing":
            self._reset_current()
        elif k == "R" and self._state.mode == "drawing":
            self._reset_all()
        elif k.lower() == "s":
            self._save_and_close()

    # ── Render loop ───────────────────────────────────────────────────────────

    def _loop(self):
        self._render()
        self._after_id = self.after(33, self._loop)

    def _render(self):
        """Delegate frame rendering to ROIRenderer and update the canvas image."""
        self._photo = self._renderer.render(self._base, self._state)
        self._canvas.itemconfig(self._canvas_img, image=self._photo)

    # ── Save / close ──────────────────────────────────────────────────────────

    def _save_and_close(self):
        if not self._state.polygons:
            _error_dialog(self, "No lanes defined.\nDraw at least one lane before saving.")
            return
        lanes: dict = {}
        for poly in self._state.polygons:
            xs  = [p[0] for p in poly["pts"]]
            ys  = [p[1] for p in poly["pts"]]
            roi = [min(xs), min(ys), max(xs), max(ys)]
            entry: dict = {
                "roi":      roi,
                "points":   [list(p) for p in poly["pts"]],
                "label_pt": list(poly.get("label_pt", [roi[0], roi[1]])),
            }
            if poly.get("expected_direction"):
                entry["expected_direction"] = poly["expected_direction"]
            lanes[poly["name"]] = entry

        if self._camera_id and self._cameras_dir:
            save_camera_lanes(self._camera_id, self._cameras_dir, lanes)
        else:
            with open(self._config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["lanes"] = lanes
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)

        print(f"[ROI Selector] {len(self._state.polygons)} lane(s) saved.")
        self._cleanup()
        self.destroy()

    def _exit_without_saving(self):
        """Skip confirmation if no lanes exist; ask if there are unsaved lanes."""
        if self._state.polygons:
            if not _confirm_dialog(self, "Exit without saving?",
                                   f"You have {len(self._state.polygons)} unsaved lane(s).\n"
                                   "They will be lost. Exit anyway?"):
                return
        self._cleanup()
        self.destroy()

    def _on_close(self):
        """Window X-button handler — always confirm if there is work to lose."""
        if self._state.current_pts or self._state.polygons:
            if not _confirm_dialog(self, "Exit without saving?",
                                   "Unsaved changes will be lost.\nExit anyway?"):
                return
        self._cleanup()
        self.destroy()

    def _cleanup(self):
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    config_path: str | None  = None,
    camera_id:   str | None  = None,
    stream_url:  str | None  = None,
    cameras_dir: Path | None = None,
) -> None:
    if config_path is None:
        this_dir    = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.normpath(os.path.join(this_dir, "..", "config.json"))
    if cameras_dir is None:
        cameras_dir = Path(os.path.dirname(os.path.abspath(config_path))) / "cameras"

    if stream_url:
        source    = stream_url
        cam_label = camera_id or ""
    else:
        cfg        = load_config(config_path)
        video_path = cfg.camera.video_path
        if not os.path.isabs(video_path):
            video_path = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(config_path)), video_path))
        source    = video_path
        camera_id = camera_id or "local"
        cam_label = Path(video_path).name

    parent = tk._default_root
    if parent is None:
        parent = ctk.CTk()
        parent.withdraw()

    frame = _grab_frame(source)
    if frame is None:
        _error_dialog(parent, f"Cannot grab a frame from:\n{source}")
        return

    initial_lanes: list[dict] = []
    try:
        for lane in load_camera_lanes(camera_id, cameras_dir):
            initial_lanes.append({
                "name":               lane.name,
                "pts":                [tuple(p) for p in lane.points],
                "expected_direction": lane.expected_direction,
                "label_pt":           lane.label_pt if lane.label_pt
                                      else (lane.points[0] if lane.points else [0, 0]),
            })
    except Exception:
        pass

    ROISelectorWindow(
        parent=parent,
        frame=frame,
        config_path=config_path,
        camera_id=camera_id,
        cameras_dir=cameras_dir,
        cam_label=cam_label,
        initial_lanes=initial_lanes,
    ).wait_window()


if __name__ == "__main__":
    run()
