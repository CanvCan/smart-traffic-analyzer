"""
traffic_analyzer/visualization/roi_selector.py

Premium ROI Selector — canvas-first layout, 3-step progress stepper, no cv2 windows.
Design: Unified deep-space dark theme, vertically centered canvas on dark field,
        resizable window with enforced minimum dimensions.
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
from PIL import Image, ImageDraw, ImageFont, ImageTk

from traffic_analyzer.infrastructure.config_loader import (
    load_config, load_camera_lanes, save_camera_lanes,
)
from traffic_analyzer.visualization.colors import LANE_PALETTE

# ── Unified deep-space palette (matches launcher theme) ───────────────────────
_BG       = "#0b0b14"   # app background
_SURFACE  = "#111120"   # sidebar / panel surface
_CARD     = "#181828"   # bento card
_CARD_HI  = "#1f1f35"   # elevated / hover card
_BORDER_S = "#1e1e38"   # barely-visible border
_BORDER   = "#2d2d50"   # visible border
_FG       = "#e8e8f4"   # primary text
_FG_MED   = "#8a8ab0"   # secondary text
_FG_DIM   = "#40405c"   # hint / disabled text
_ACCENT   = "#6c8efa"   # CTA blue (Linear-style)
_ACCENT_H = "#5577e8"   # accent hover
_TEAL     = "#5bbdcc"   # teal accent
_SEL_BG   = "#131828"   # selected item background
_GREEN    = "#4ade96"   # success
_AMBER    = "#f5a832"   # warning
_RED      = "#f06080"   # danger / delete

# ── Typography ─────────────────────────────────────────────────────────────────
_FT = ("Segoe UI", 15, "bold")   # window title
_FH = ("Segoe UI", 11, "bold")   # section heading
_FB = ("Segoe UI", 10)           # body
_FS = ("Segoe UI",  9)           # small / label
_FC = ("Consolas",  9)           # mono / keyboard hints

SNAP_RADIUS = 12
SIDEBAR_W   = 300

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

# ── PIL font cache ─────────────────────────────────────────────────────────────
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _pil_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        for path in [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/segoeuil.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]:
            try:
                _FONT_CACHE[size] = ImageFont.truetype(path, size)
                return _FONT_CACHE[size]
            except OSError:
                pass
        _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _text_size(text: str, size: int) -> tuple[int, int]:
    d  = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bb = d.textbbox((0, 0), text, font=_pil_font(size))
    return bb[2] - bb[0], bb[3] - bb[1]


def _np_to_photo(bgr: np.ndarray) -> ImageTk.PhotoImage:
    return ImageTk.PhotoImage(Image.fromarray(bgr[:, :, ::-1]))


def _lane_hex(idx: int) -> str:
    b, g, r = LANE_PALETTE[idx % len(LANE_PALETTE)]
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Helper dialogs ─────────────────────────────────────────────────────────────

def _make_dialog(parent: tk.Misc, title: str, w: int, h: int) -> ctk.CTkToplevel:
    top = ctk.CTkToplevel(parent)
    top.title(title)
    top.configure(fg_color=_BG)
    top.resizable(False, False)
    top.attributes("-topmost", True)
    top.update_idletasks()
    sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
    top.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
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

    ctk.CTkButton(bf, text="Exit anyway", font=_FB, width=110, height=34,
                  corner_radius=8, fg_color=_CARD, hover_color="#2a0f1a",
                  border_color=_RED, border_width=1,
                  text_color=_RED, command=_yes).pack(side="right", padx=(8, 0))
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


# ── Frame capture ──────────────────────────────────────────────────────────────

def _grab_frame(source: str) -> Optional[np.ndarray]:
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
                            np.frombuffer(buf[s:e+2], np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            return frame
        except Exception as ex:
            print(f"[ROI] stream error: {ex}")
        return None
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ── ROI Selector Window ────────────────────────────────────────────────────────

class ROISelectorWindow(ctk.CTkToplevel):
    """
    Premium ROI selector.

    Layout: Left = dark field with canvas vertically centered (fill+expand).
            Right = fixed-width sidebar: header + 3-step stepper + bento scroll + save button.
    Window minimum: 860 × 560. Resizable vertically so sidebar content is always accessible.
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

        # Fill as much of the screen as possible — cap at 4.0x
        max_cw = sw - SIDEBAR_W - 6
        max_ch = sh - 72      # leave room for taskbar + title bar
        scale  = min(max_cw / vw, max_ch / vh)
        scale  = min(max(scale, 0.30), 4.0)

        self._scale = scale
        self._cw    = int(vw * scale)
        self._ch    = int(vh * scale)
        self._base  = cv2.resize(frame, (self._cw, self._ch), interpolation=cv2.INTER_LINEAR)

        self._config_path = config_path
        self._camera_id   = camera_id
        self._cameras_dir = cameras_dir
        self._cam_label   = cam_label

        self._polygons:    list[dict]            = list(initial_lanes) if initial_lanes else []
        self._current_pts: list[tuple[int, int]] = []
        self._drag_idx:    Optional[int]         = None
        self._hover_idx:   Optional[int]         = None
        self._mode      = "drawing"
        self._temp_name = ""
        self._temp_dir  = ""
        self._photo:    Optional[ImageTk.PhotoImage] = None
        self._after_id: Optional[str]               = None

        # Window geometry: canvas width + sidebar, minimum 860 × 580
        win_w = max(self._cw + SIDEBAR_W, 860)
        win_h = max(self._ch, 580)
        win_x = max((sw - win_w) // 2, 0)
        win_y = max((sh - win_h) // 2, 0)

        self.title("ROI Selector")
        self.configure(fg_color=_BG)
        self.geometry(f"{win_w}x{win_h}+{win_x}+{win_y}")
        self.minsize(860, 580)
        self.resizable(True, True)
        self.attributes("-topmost", True)

        self._build()
        self._refresh_sidebar()
        self._loop()
        self.focus_force()
        self.bind("<Key>", self._on_key)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # CTkToplevel's WM may override the initial geometry — reapply position after 60 ms
        self.after(60, lambda: self.geometry(f"{win_w}x{win_h}+{win_x}+{win_y}"))

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        # Left: dark canvas field — canvas is placed at center, field fills remaining space
        self._canvas_frame = tk.Frame(self, bg="#040408")
        self._canvas_frame.pack(side="left", fill="both", expand=True)

        self._canvas = tk.Canvas(
            self._canvas_frame,
            width=self._cw, height=self._ch,
            bg="#040408", highlightthickness=0, cursor="crosshair",
        )
        # place() keeps the canvas centered as the window resizes
        self._canvas.place(relx=0.5, rely=0.5, anchor="center")
        self._canvas_img = self._canvas.create_image(0, 0, anchor="nw")

        self._canvas.bind("<Button-1>",        self._on_lclick)
        self._canvas.bind("<B1-Motion>",       self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_lrelease)
        self._canvas.bind("<Button-3>",        self._on_rclick)
        self._canvas.bind("<Motion>",          self._on_move)

        # 1px divider between canvas field and sidebar
        tk.Frame(self, width=1, bg=_BORDER_S).pack(side="left", fill="y")

        # Right: sidebar — fixed width, fills full height
        self._sidebar = tk.Frame(self, width=SIDEBAR_W - 1, bg=_SURFACE)
        self._sidebar.pack(side="left", fill="y")
        self._sidebar.pack_propagate(False)

        # ── Sidebar header ────────────────────────────────────────────────────
        hdr = tk.Frame(self._sidebar, bg=_BG)
        hdr.pack(fill="x")

        name_row = tk.Frame(hdr, bg=_BG)
        name_row.pack(fill="x", padx=16, pady=(16, 0))

        tk.Label(name_row, text="ROI Selector", font=_FT,
                 fg=_FG, bg=_BG, anchor="w").pack(side="left")

        self._mode_badge = tk.Label(
            name_row, text="● DRAWING",
            font=("Segoe UI", 7, "bold"),
            fg=_ACCENT, bg=_CARD, padx=7, pady=3, relief="flat",
        )
        self._mode_badge.pack(side="right")

        if self._cam_label:
            tk.Label(hdr, text=self._cam_label,
                     font=_FS, fg=_FG_DIM, bg=_BG, anchor="w").pack(
                fill="x", padx=16, pady=(2, 10))
        else:
            tk.Frame(hdr, height=8, bg=_BG).pack()

        # ── 3-step progress stepper ───────────────────────────────────────────
        self._stepper_frame = tk.Frame(self._sidebar, bg=_BG)
        self._stepper_frame.pack(fill="x")
        self._build_stepper()

        tk.Frame(self._sidebar, height=1, bg=_BORDER_S).pack(fill="x")

        # ── Scrollable bento content ──────────────────────────────────────────
        self._scroll = ctk.CTkScrollableFrame(
            self._sidebar,
            fg_color=_SURFACE, corner_radius=0,
            scrollbar_button_color=_BORDER,
            scrollbar_button_hover_color=_FG_DIM,
        )
        self._scroll.pack(fill="both", expand=True, pady=(4, 0))

        # ── Save & Close button (pinned to sidebar bottom) ────────────────────
        tk.Frame(self._sidebar, height=1, bg=_BORDER_S).pack(fill="x", side="bottom")
        bot = tk.Frame(self._sidebar, bg=_BG, height=68)
        bot.pack(fill="x", side="bottom")
        bot.pack_propagate(False)
        self._save_btn = ctk.CTkButton(
            bot,
            text="Save & Close",
            font=("Segoe UI", 11, "bold"),
            height=42, corner_radius=10,
            fg_color=_ACCENT, hover_color=_ACCENT_H,
            text_color="#ffffff",
            command=self._save_and_close,
        )
        self._save_btn.pack(fill="x", padx=14, pady=13)

    # ── Progress stepper ──────────────────────────────────────────────────────

    def _build_stepper(self):
        """Render the 3-step horizontal stepper in the sidebar header."""
        for w in self._stepper_frame.winfo_children():
            w.destroy()

        steps = [
            ("drawing",   "1", "Draw"),
            ("naming",    "2", "Name"),
            ("direction", "3", "Dir"),
        ]
        container = tk.Frame(self._stepper_frame, bg=_BG)
        container.pack(fill="x", padx=16, pady=(8, 12))

        for i, (step_id, num, label) in enumerate(steps):
            is_done   = self._step_done(step_id)
            is_active = (self._mode == step_id)

            # Connector line between circles
            if i > 0:
                prev_done  = self._step_done(steps[i - 1][0])
                line_color = _GREEN if prev_done else _BORDER
                tk.Frame(container, height=2, bg=line_color).pack(
                    side="left", fill="x", expand=True, pady=11)

            group = tk.Frame(container, bg=_BG)
            group.pack(side="left")

            circle_bg = _GREEN  if is_done   else (_ACCENT if is_active else _CARD)
            circle_fg = "#0b0b14" if is_done else (_FG     if is_active else _FG_DIM)
            num_text  = "✓"     if is_done   else num

            circ = tk.Canvas(group, width=26, height=26, bg=_BG, highlightthickness=0)
            circ.pack()
            circ.create_oval(1, 1, 25, 25, fill=circle_bg, outline="")
            if is_active and not is_done:
                circ.create_oval(1, 1, 25, 25, fill=circle_bg, outline=_ACCENT, width=2)
            circ.create_text(13, 13, text=num_text,
                             font=("Segoe UI", 9, "bold"), fill=circle_fg)

            lbl_color = _GREEN if is_done else (_FG if is_active else _FG_DIM)
            tk.Label(group, text=label, font=("Segoe UI", 7, "bold"),
                     fg=lbl_color, bg=_BG).pack()

    def _step_done(self, step_id: str) -> bool:
        if step_id == "drawing":
            return self._mode in ("naming", "direction")
        if step_id == "naming":
            return self._mode == "direction"
        return False

    # ── Bento card helper ─────────────────────────────────────────────────────

    def _bento(self, title: str | None = None) -> ctk.CTkFrame:
        card = ctk.CTkFrame(
            self._scroll,
            fg_color=_CARD,
            corner_radius=10,
            border_color=_BORDER_S,
            border_width=1,
        )
        card.pack(fill="x", padx=12, pady=5)
        if title:
            ctk.CTkLabel(card, text=title,
                         font=("Segoe UI", 8, "bold"),
                         text_color=_FG_DIM).pack(
                anchor="w", padx=14, pady=(12, 0))
            ctk.CTkFrame(card, height=1, fg_color=_BORDER_S,
                         corner_radius=0).pack(fill="x", padx=14, pady=(6, 8))
        return card

    # ── Sidebar panels ────────────────────────────────────────────────────────

    def _refresh_sidebar(self):
        badge_map = {
            "drawing":   ("● DRAWING",   _ACCENT),
            "naming":    ("✎ NAMING",    _AMBER),
            "direction": ("◈ DIRECTION", _TEAL),
        }
        text, color = badge_map.get(self._mode, ("●", _FG_DIM))
        self._mode_badge.configure(text=text, fg=color)

        self._build_stepper()

        for w in self._scroll.winfo_children():
            w.destroy()

        if self._mode == "drawing":
            self._panel_drawing()
        elif self._mode == "naming":
            self._panel_naming()
        elif self._mode == "direction":
            self._panel_direction()

    # ── Drawing panel ─────────────────────────────────────────────────────────

    def _panel_drawing(self):
        n_lanes = len(self._polygons)
        n_pts   = len(self._current_pts)

        # Defined lanes card
        card = self._bento(f"DEFINED LANES  ({n_lanes})")
        if not n_lanes:
            ctk.CTkLabel(card,
                         text="No lanes yet.\nClick the video to start placing points.",
                         font=_FS, text_color=_FG_DIM,
                         wraplength=240, justify="left").pack(
                padx=14, pady=(0, 12), anchor="w")
        else:
            for i, poly in enumerate(self._polygons):
                self._lane_row(card, i, poly)
            tk.Frame(card, height=6, bg=_CARD).pack()

        # Current polygon card
        ready    = n_pts >= 3
        cur_card = self._bento(f"DRAWING — LANE {n_lanes + 1}")

        if n_pts == 0:
            dot_color, status = _FG_DIM, "Click the frame to place points"
        elif not ready:
            dot_color = _AMBER
            status    = f"{n_pts} point{'s' if n_pts > 1 else ''} — need {3 - n_pts} more"
        else:
            dot_color = _GREEN
            status    = f"{n_pts} points — ready to confirm"

        status_row = ctk.CTkFrame(cur_card, fg_color="transparent", corner_radius=0)
        status_row.pack(fill="x", padx=14, pady=(0, 8))
        ctk.CTkLabel(status_row, text="●", font=("Segoe UI", 9),
                     text_color=dot_color).pack(side="left", padx=(0, 6))
        ctk.CTkLabel(status_row, text=status, font=_FS,
                     text_color=_FG_MED if n_pts else _FG_DIM,
                     wraplength=196, justify="left").pack(side="left", anchor="w")

        if ready:
            ctk.CTkButton(
                cur_card,
                text="✓  Confirm Lane",
                font=("Segoe UI", 10, "bold"), height=36, corner_radius=8,
                fg_color=_SEL_BG, hover_color=_CARD_HI,
                border_color=_ACCENT, border_width=2,
                text_color=_ACCENT,
                command=self._start_naming,
            ).pack(fill="x", padx=14, pady=(0, 4))

        if n_pts:
            ctk.CTkButton(
                cur_card,
                text="Reset Current",
                font=_FS, height=30, corner_radius=7,
                fg_color="transparent", hover_color=_CARD_HI,
                border_color=_BORDER, border_width=1,
                text_color=_FG_MED,
                command=self._reset_current,
            ).pack(fill="x", padx=14, pady=(0, 4))

        tk.Frame(cur_card, height=4, bg=_CARD).pack()

        if n_lanes:
            danger_card = self._bento()
            ctk.CTkButton(
                danger_card,
                text="Reset All Lanes",
                font=_FS, height=30, corner_radius=7,
                fg_color="transparent", hover_color="#220a12",
                border_color=_RED, border_width=1,
                text_color=_RED,
                command=self._reset_all,
            ).pack(fill="x", padx=14, pady=(10, 10))

        # Shortcuts card
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
            row.pack(fill="x", padx=14, pady=2)
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
        row.pack(fill="x", padx=14, pady=3)
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
                     font=("Segoe UI", 11), text_color=_TEAL).place(
            relx=0.5, rely=0.5, anchor="center")

        ctk.CTkButton(
            row, text="✕", font=("Segoe UI", 9),
            width=26, height=26, corner_radius=6,
            fg_color="transparent", hover_color="#220a12",
            text_color=_FG_DIM,
            command=lambda i=idx: self._delete_lane(i),
        ).pack(side="right", padx=(2, 8))

    # ── Naming panel ──────────────────────────────────────────────────────────

    def _panel_naming(self):
        card = self._bento(f"NAME — LANE {len(self._polygons) + 1}")

        ctk.CTkLabel(card, text="What should this lane be called?",
                     font=_FS, text_color=_FG_DIM).pack(
            padx=14, pady=(0, 8), anchor="w")

        self._name_var   = ctk.StringVar(value=self._temp_name)
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
        self._name_entry.pack(fill="x", padx=14, pady=(0, 4))
        self._name_entry.after(80, self._name_entry.focus_set)
        self._name_entry.bind("<Return>", lambda _: self._confirm_name())
        self._name_entry.bind("<Escape>", lambda _: self._cancel_mode())

        self._name_err = ctk.CTkLabel(card, text="", font=_FS, text_color=_RED)
        self._name_err.pack(padx=14, anchor="w")

        ctk.CTkButton(
            card,
            text="Next  →",
            font=("Segoe UI", 10, "bold"), height=38, corner_radius=9,
            fg_color=_ACCENT, hover_color=_ACCENT_H,
            text_color="#ffffff",
            command=self._confirm_name,
        ).pack(fill="x", padx=14, pady=(10, 4))

        ctk.CTkButton(
            card, text="Cancel",
            font=_FS, height=30, corner_radius=7,
            fg_color="transparent", hover_color=_CARD_HI,
            border_color=_BORDER, border_width=1,
            text_color=_FG_MED,
            command=self._cancel_mode,
        ).pack(fill="x", padx=14, pady=(0, 12))

    # ── Direction panel ───────────────────────────────────────────────────────

    def _panel_direction(self):
        card = self._bento("TRAFFIC DIRECTION")

        ctk.CTkLabel(card, text=f"Lane:  {self._temp_name}",
                     font=_FB, text_color=_FG_MED).pack(
            padx=14, pady=(0, 4), anchor="w")
        ctk.CTkLabel(card,
                     text="Which direction do vehicles travel through this lane?",
                     font=_FS, text_color=_FG_DIM,
                     wraplength=244, justify="left").pack(
            padx=14, pady=(0, 10), anchor="w")

        for arrow, label, value in _DIRECTIONS:
            sel = (self._temp_dir == value)
            btn_card = ctk.CTkFrame(
                card,
                fg_color=_SEL_BG  if sel else _CARD_HI,
                corner_radius=9,
                border_color=_ACCENT if sel else _BORDER_S,
                border_width=2 if sel else 1,
                height=42, cursor="hand2",
            )
            btn_card.pack(fill="x", padx=14, pady=3)
            btn_card.pack_propagate(False)

            inner = ctk.CTkFrame(btn_card, fg_color="transparent", corner_radius=0)
            inner.pack(fill="both", expand=True, padx=12)

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
            font=("Segoe UI", 10, "bold"), height=38, corner_radius=9,
            fg_color=_ACCENT, hover_color=_ACCENT_H,
            text_color="#ffffff",
            command=self._confirm_direction,
        ).pack(fill="x", padx=14, pady=(14, 4))

        ctk.CTkButton(
            card, text="← Back",
            font=_FS, height=30, corner_radius=7,
            fg_color="transparent", hover_color=_CARD_HI,
            border_color=_BORDER, border_width=1,
            text_color=_FG_MED,
            command=self._back_to_naming,
        ).pack(fill="x", padx=14, pady=(0, 12))

    # ── Mode transitions ──────────────────────────────────────────────────────

    def _start_naming(self):
        if len(self._current_pts) < 3:
            return
        self._mode      = "naming"
        self._temp_name = f"Serit {len(self._polygons) + 1}"
        self._temp_dir  = ""
        self._refresh_sidebar()

    def _confirm_name(self):
        name = self._name_var.get().strip()
        if not name:
            self._name_err.configure(text="Name cannot be empty.")
            return
        if name in {p["name"] for p in self._polygons}:
            self._name_err.configure(text=f"'{name}' is already used.")
            return
        self._temp_name = name
        self._mode      = "direction"
        self._refresh_sidebar()

    def _back_to_naming(self):
        self._mode = "naming"
        self._refresh_sidebar()

    def _cancel_mode(self):
        self._mode = "drawing"
        self._refresh_sidebar()
        self.focus_force()

    def _select_direction(self, value: str):
        self._temp_dir = value
        self._refresh_sidebar()

    def _confirm_direction(self):
        xs = [p[0] for p in self._current_pts]
        ys = [p[1] for p in self._current_pts]
        self._polygons.append({
            "name":               self._temp_name,
            "pts":                list(self._current_pts),
            "expected_direction": self._temp_dir,
            "label_pt":           [min(xs), min(ys)],
        })
        self._current_pts = []
        self._drag_idx    = None
        self._hover_idx   = None
        self._mode        = "drawing"
        self._refresh_sidebar()
        self.focus_force()

    # ── Lane management ───────────────────────────────────────────────────────

    def _delete_lane(self, idx: int):
        if 0 <= idx < len(self._polygons):
            self._polygons.pop(idx)
            self._refresh_sidebar()

    def _reset_current(self):
        self._current_pts = []
        self._drag_idx    = None
        self._hover_idx   = None
        self._refresh_sidebar()

    def _reset_all(self):
        self._polygons    = []
        self._current_pts = []
        self._drag_idx    = None
        self._hover_idx   = None
        self._refresh_sidebar()

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _to_orig(self, cx: int, cy: int) -> tuple[int, int]:
        return int(cx / self._scale), int(cy / self._scale)

    def _to_canvas(self, ox: int, oy: int) -> tuple[int, int]:
        return int(ox * self._scale), int(oy * self._scale)

    def _nearest(self, cx: int, cy: int) -> Optional[int]:
        for i, (ox, oy) in enumerate(self._current_pts):
            px, py = self._to_canvas(ox, oy)
            if (px - cx) ** 2 + (py - cy) ** 2 <= SNAP_RADIUS ** 2:
                return i
        return None

    # ── Mouse events ──────────────────────────────────────────────────────────

    def _on_lclick(self, e):
        if self._mode != "drawing":
            return
        idx = self._nearest(e.x, e.y)
        if idx is not None:
            self._drag_idx = idx
        else:
            self._current_pts.append(self._to_orig(e.x, e.y))
            self._refresh_sidebar()

    def _on_drag(self, e):
        if self._mode != "drawing" or self._drag_idx is None:
            return
        if self._drag_idx >= len(self._current_pts):
            self._drag_idx = None
            return
        self._current_pts[self._drag_idx] = self._to_orig(e.x, e.y)

    def _on_lrelease(self, _e):
        self._drag_idx = None

    def _on_rclick(self, _e):
        if self._mode != "drawing" or not self._current_pts:
            return
        self._current_pts.pop()
        if self._drag_idx is not None and self._drag_idx >= len(self._current_pts):
            self._drag_idx = None
        self._refresh_sidebar()

    def _on_move(self, e):
        if self._mode != "drawing":
            return
        prev            = self._hover_idx
        self._hover_idx = self._nearest(e.x, e.y)
        cur             = "fleur" if self._hover_idx is not None else "crosshair"
        if self._canvas["cursor"] != cur:
            self._canvas.configure(cursor=cur)
        if prev != self._hover_idx:
            self._render()

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def _on_key(self, e):
        k = e.keysym
        if k in ("Return", "space"):
            if   self._mode == "drawing":   self._start_naming()
            elif self._mode == "naming":    self._confirm_name()
            elif self._mode == "direction": self._confirm_direction()
        elif k == "Escape":
            if self._mode in ("naming", "direction"):
                self._cancel_mode()
            elif self._mode == "drawing" and self._current_pts:
                self._reset_current()
        elif k == "r" and self._mode == "drawing":
            self._reset_current()
        elif k == "R" and self._mode == "drawing":
            self._reset_all()
        elif k.lower() == "s":
            self._save_and_close()

    # ── Render loop ───────────────────────────────────────────────────────────

    def _loop(self):
        self._render()
        self._after_id = self.after(33, self._loop)

    def _render(self):
        display  = self._base.copy()
        deferred: list[tuple[str, int, int, tuple]] = []

        # Completed polygons
        for i, poly in enumerate(self._polygons):
            bgr  = LANE_PALETTE[i % len(LANE_PALETTE)]
            spts = np.array([self._to_canvas(*p) for p in poly["pts"]], np.int32)

            ov = display.copy()
            cv2.fillPoly(ov, [spts], bgr)
            cv2.addWeighted(ov, 0.22, display, 0.78, 0, display)
            cv2.polylines(display, [spts], True, bgr, 2, cv2.LINE_AA)

            for sp in spts:
                sp = tuple(sp)
                cv2.circle(display, sp, 5, bgr, -1, cv2.LINE_AA)
                cv2.circle(display, sp, 5, (255, 255, 255), 1, cv2.LINE_AA)

            if "label_pt" in poly:
                ox, oy = int(poly["label_pt"][0]), int(poly["label_pt"][1])
                sx, sy = self._to_canvas(ox, oy)
                tw, th = _text_size(poly["name"], 13)
                pad    = 7
                bx1, by1 = max(sx - pad, 0), max(sy - pad, 0)
                bx2      = min(sx + tw + pad, display.shape[1] - 1)
                by2      = min(sy + th + pad, display.shape[0] - 1)
                ov2 = display.copy()
                cv2.rectangle(ov2, (bx1, by1), (bx2, by2), (10, 8, 18), -1)
                cv2.addWeighted(ov2, 0.88, display, 0.12, 0, display)
                cv2.rectangle(display, (bx1, by1), (bx2, by2), bgr, 1, cv2.LINE_AA)
                deferred.append((poly["name"], sx, sy, bgr))

        # Active (in-progress) polygon
        if self._current_pts:
            bgr  = LANE_PALETTE[len(self._polygons) % len(LANE_PALETTE)]
            spts = [self._to_canvas(*p) for p in self._current_pts]

            for j in range(len(spts) - 1):
                cv2.line(display, spts[j], spts[j + 1], bgr, 2, cv2.LINE_AA)

            if len(spts) >= 3:
                dim = tuple(max(0, int(c * 0.35)) for c in bgr)
                cv2.line(display, spts[-1], spts[0], dim, 1, cv2.LINE_AA)

            for j, sp in enumerate(spts):
                hover    = (j == self._hover_idx)
                r        = SNAP_RADIUS + 4 if hover else SNAP_RADIUS
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

        # Glassmorphism HUD bar at canvas bottom
        self._draw_hud(display)

        # PIL text pass for lane labels
        if deferred:
            pil  = Image.fromarray(display[:, :, ::-1])
            draw = ImageDraw.Draw(pil)
            font = _pil_font(13)
            for text, sx, sy, bgr in deferred:
                rgb = (bgr[2], bgr[1], bgr[0])
                draw.text((sx + 1, sy + 1), text, font=font, fill=(0, 0, 0, 180))
                draw.text((sx, sy),         text, font=font, fill=rgb)
            display = np.array(pil)[:, :, ::-1]

        self._photo = _np_to_photo(display)
        self._canvas.itemconfig(self._canvas_img, image=self._photo)

    def _draw_hud(self, display: np.ndarray):
        h, w = display.shape[:2]
        n    = len(self._current_pts)

        if self._mode == "drawing":
            if n == 0:
                msg = "Left-click to place points  ·  Enter: confirm  ·  S: save"
            elif n < 3:
                msg = f"{n} pt{'s' if n > 1 else ''} placed  ·  Need {3-n} more  ·  R-Click: undo"
            else:
                msg = f"{n} pts ready  ·  Enter: confirm  ·  Drag to adjust  ·  R-Click: undo"
        elif self._mode == "naming":
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

        n_lanes = len(self._polygons)
        if n_lanes:
            bw, bh = _text_size(f"{n_lanes} lane{'s' if n_lanes > 1 else ''}", 11)
            bw += 16
            overlay = display.copy()
            cv2.rectangle(overlay, (10, 10), (10 + bw, 10 + bh + 8), (10, 8, 20), -1)
            cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
            cv2.rectangle(display, (10, 10), (10 + bw, 10 + bh + 8), (40, 80, 100), 1, cv2.LINE_AA)
            cv2.putText(display, f"{n_lanes} lane{'s' if n_lanes > 1 else ''}",
                        (18, 10 + bh + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (91, 189, 204), 1, cv2.LINE_AA)

    # ── Save / close ──────────────────────────────────────────────────────────

    def _save_and_close(self):
        if not self._polygons:
            _error_dialog(self, "No lanes defined.\nDraw at least one lane before saving.")
            return
        lanes: dict = {}
        for poly in self._polygons:
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

        print(f"[ROI Selector] {len(self._polygons)} lane(s) saved.")
        self._cleanup()
        self.destroy()

    def _on_close(self):
        if self._current_pts or self._polygons:
            if not _confirm_dialog(self, "Exit without saving?",
                                   "Unsaved changes will be lost.\nExit anyway?"):
                return
        self._cleanup()
        self.destroy()

    def _cleanup(self):
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None


# ── Entry point ────────────────────────────────────────────────────────────────

def run(
    config_path: str | None = None,
    camera_id:   str | None = None,
    stream_url:  str | None = None,
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
