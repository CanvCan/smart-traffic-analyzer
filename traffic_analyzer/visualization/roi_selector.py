import tkinter as tk

import cv2
import json
import os
from pathlib import Path

import numpy as np
import customtkinter as ctk

from traffic_analyzer.infrastructure.config_loader import load_config, save_camera_lanes
from traffic_analyzer.visualization.colors import LANE_PALETTE

SNAP_RADIUS = 10
SCALING_FACTOR = 0.5

# ── Launcher colour palette (kept in sync with launcher/app.py) ───────────────
_BG      = "#1e1e2e"
_SURFACE = "#252535"
_CARD    = "#2a2a3e"
_BORDER  = "#383858"
_FG      = "#cdd6f4"
_FG_MED  = "#a6adc8"
_FG_DIM  = "#585b70"
_TEAL    = "#7eb8c9"
_SEL_BG  = "#1e2d3d"
_SEL_BOR = "#7eb8c9"
_RED     = "#f38ba8"

_F_TITLE = ("Segoe UI", 13, "bold")
_F_BODY  = ("Segoe UI", 10)
_F_SMALL = ("Segoe UI", 9)

# ── Global state ──────────────────────────────────────────────────────────────
polygons    = []
current_pts = []
drag_idx    = None
placing_label = False
label_preview = None


# ── Coordinate helpers ────────────────────────────────────────────────────────
def _to_orig(sx, sy):
    return int(sx / SCALING_FACTOR), int(sy / SCALING_FACTOR)

def _to_screen(ox, oy):
    return int(ox * SCALING_FACTOR), int(oy * SCALING_FACTOR)

def _nearest(pts, sx, sy):
    for i, (ox, oy) in enumerate(pts):
        spx, spy = _to_screen(ox, oy)
        if abs(spx - sx) <= SNAP_RADIUS and abs(spy - sy) <= SNAP_RADIUS:
            return i
    return None


# ── Mouse callbacks ───────────────────────────────────────────────────────────
def mouse_polygon(event, x, y, flags, param):
    global current_pts, drag_idx
    ox, oy = _to_orig(x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = _nearest(current_pts, x, y)
        if idx is not None:
            drag_idx = idx
        else:
            current_pts.append((ox, oy))
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_idx is not None:
            current_pts[drag_idx] = (ox, oy)
    elif event == cv2.EVENT_LBUTTONUP:
        drag_idx = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_pts:
            current_pts.pop()

def mouse_label(event, x, y, flags, param):
    global label_preview, polygons, placing_label
    ox, oy = _to_orig(x, y)
    if event == cv2.EVENT_MOUSEMOVE:
        label_preview = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        polygons[-1]["label_pt"] = (ox, oy)
        placing_label = False
        label_preview = None
        cv2.setMouseCallback("ROI Selector", mouse_polygon)


# ── PIL text (Unicode / Turkish support) ─────────────────────────────────────
_PIL_FONTS: dict = {}

def _pil_font(size: int):
    from PIL import ImageFont
    if size not in _PIL_FONTS:
        for path in ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]:
            try:
                _PIL_FONTS[size] = ImageFont.truetype(path, size)
                return _PIL_FONTS[size]
            except OSError:
                pass
        _PIL_FONTS[size] = ImageFont.load_default()
    return _PIL_FONTS[size]

def _pil_text(img: np.ndarray, text: str, x: int, y: int,
              font_size: int, color_bgr: tuple, shadow: bool = True) -> None:
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img[:, :, ::-1])
    draw = ImageDraw.Draw(pil)
    font = _pil_font(font_size)
    cr, cg, cb = color_bgr[2], color_bgr[1], color_bgr[0]
    if shadow:
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(cr, cg, cb))
    img[:] = np.array(pil)[:, :, ::-1]

def _measure_text(text: str, font_size: int) -> tuple[int, int]:
    from PIL import Image, ImageDraw
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bb = draw.textbbox((0, 0), text, font=_pil_font(font_size))
    return bb[2] - bb[0], bb[3] - bb[1]


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _draw_polygon(display, pts, color, closed=False):
    if not pts:
        return
    spts = [_to_screen(ox, oy) for ox, oy in pts]
    for i in range(len(spts) - 1):
        cv2.line(display, spts[i], spts[i + 1], color, 2, cv2.LINE_AA)
    if closed and len(spts) > 2:
        cv2.line(display, spts[-1], spts[0], color, 2, cv2.LINE_AA)
        overlay = display.copy()
        cv2.fillPoly(overlay, [np.array(spts)], color)
        cv2.addWeighted(overlay, 0.14, display, 0.86, 0, display)
    for i, sp in enumerate(spts):
        cv2.circle(display, sp, SNAP_RADIUS, color, -1, cv2.LINE_AA)
        cv2.circle(display, sp, SNAP_RADIUS, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, str(i + 1), (sp[0] + 10, sp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_label(display, text: str, ox: int, oy: int, color: tuple) -> None:
    """Draw a styled pill-label for a lane name (PIL for Turkish chars)."""
    sx, sy = _to_screen(ox, oy)
    fs = 15
    tw, th = _measure_text(text, fs)
    pad = 6
    bx1, by1 = sx - pad, sy - pad
    bx2, by2 = sx + tw + pad, sy + th + pad
    # clip to frame
    bx1 = max(bx1, 0); by1 = max(by1, 0)
    bx2 = min(bx2, display.shape[1] - 1); by2 = min(by2, display.shape[0] - 1)
    # semi-transparent dark background
    overlay = display.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (15, 12, 22), -1)
    cv2.addWeighted(overlay, 0.78, display, 0.22, 0, display)
    # coloured border
    cv2.rectangle(display, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)
    # text
    _pil_text(display, text, sx, sy, fs, (230, 230, 230), shadow=False)


def _draw_status(display, lines, h):
    bar_h = 22 * len(lines) + 8
    cv2.rectangle(display, (0, h - bar_h), (display.shape[1], h), (18, 16, 28), -1)
    for i, text in enumerate(lines):
        cv2.putText(display, text, (10, h - bar_h + 18 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.41, (190, 190, 210), 1, cv2.LINE_AA)


# ── GUI dialogs (dark CTk theme) ─────────────────────────────────────────────
def _base_dialog(title: str, w: int, h: int) -> ctk.CTkToplevel:
    top = ctk.CTkToplevel()
    top.title(title)
    top.resizable(False, False)
    top.configure(fg_color=_BG)
    top.attributes("-topmost", True)
    top.update_idletasks()
    sw = top.winfo_screenwidth()
    sh = top.winfo_screenheight()
    top.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
    top.update()
    top.grab_set()
    top.focus_force()
    return top


def _ask_name_gui(existing_polygons: list) -> "str | None":
    existing = {p["name"] for p in existing_polygons}
    result   = [None]

    top = _base_dialog("Lane Name", 380, 200)

    ctk.CTkLabel(top, text=f"Lane {len(existing_polygons) + 1}  —  enter a name",
                 font=_F_TITLE, text_color=_TEAL).pack(pady=(20, 4), padx=20, anchor="w")
    ctk.CTkLabel(top, text="Used as the label on the ROI canvas.",
                 font=_F_SMALL, text_color=_FG_DIM).pack(padx=20, anchor="w")

    var   = ctk.StringVar()
    entry = ctk.CTkEntry(top, textvariable=var, font=_F_BODY,
                         height=36, fg_color=_CARD, border_color=_BORDER,
                         text_color=_FG, placeholder_text="e.g. Şerit 1")
    entry.pack(fill="x", padx=20, pady=(10, 4))
    entry.focus_set()

    err = ctk.CTkLabel(top, text="", font=_F_SMALL, text_color=_RED)
    err.pack(padx=20, anchor="w")

    bf = ctk.CTkFrame(top, fg_color=_BG)
    bf.pack(fill="x", padx=20, pady=(6, 16), side="bottom")

    def confirm(event=None):
        name = var.get().strip()
        if not name:
            err.configure(text="Name cannot be empty.")
            return
        if name in existing:
            err.configure(text=f"'{name}' is already used.")
            return
        result[0] = name
        top.destroy()

    def cancel(event=None):
        top.destroy()

    ctk.CTkButton(bf, text="OK", font=_F_BODY, width=110, height=34,
                  fg_color=_TEAL, hover_color=_SEL_BOR, text_color=_BG,
                  command=confirm).pack(side="right", padx=(6, 0))
    ctk.CTkButton(bf, text="Cancel", font=_F_BODY, width=110, height=34,
                  fg_color=_CARD, hover_color=_SEL_BG,
                  border_color=_BORDER, border_width=1, text_color=_FG_MED,
                  command=cancel).pack(side="right")

    entry.bind("<Return>", confirm)
    entry.bind("<Escape>", cancel)
    top.protocol("WM_DELETE_WINDOW", cancel)
    top.wait_window()
    return result[0]


_DIRECTIONS = [
    # (label, subtitle, value, grid_row, grid_col)
    ("↑",  "Bottom → Top",  "bottom_to_top",  0, 1),
    ("←",  "Right → Left",  "right_to_left",  1, 0),
    ("→",  "Left → Right",  "left_to_right",  1, 2),
    ("↓",  "Top → Bottom",  "top_to_bottom",  2, 1),
]


def _ask_direction_gui(lane_name: str) -> str:
    result   = [None]
    selected = [None]   # StringVar placeholder
    btn_refs = {}

    top = _base_dialog("Vehicle Direction", 420, 370)

    # ── Header ────────────────────────────────────────────────────────────────
    ctk.CTkLabel(top, text="Expected traffic direction",
                 font=_F_TITLE, text_color=_TEAL).pack(pady=(18, 2), padx=20, anchor="w")
    ctk.CTkLabel(top, text=f"Lane: {lane_name}",
                 font=_F_SMALL, text_color=_FG_DIM).pack(padx=20, anchor="w")

    # ── Compass grid ─────────────────────────────────────────────────────────
    compass = ctk.CTkFrame(top, fg_color=_BG)
    compass.pack(pady=(14, 6))

    def _select(value):
        selected[0] = value
        for v, (btn, _) in btn_refs.items():
            if v == value:
                btn.configure(fg_color=_SEL_BG, border_color=_TEAL, border_width=2,
                              text_color=_TEAL)
            else:
                btn.configure(fg_color=_CARD, border_color=_BORDER, border_width=1,
                              text_color=_FG_MED)

    for arrow, subtitle, value, row, col in _DIRECTIONS:
        f = ctk.CTkFrame(compass, fg_color=_CARD, corner_radius=10,
                         border_color=_BORDER, border_width=1,
                         width=108, height=78, cursor="hand2")
        f.grid(row=row, column=col, padx=6, pady=6)
        f.grid_propagate(False)

        ctk.CTkLabel(f, text=arrow, font=("Segoe UI", 26, "bold"),
                     text_color=_FG_MED).place(relx=0.5, rely=0.35, anchor="center")
        ctk.CTkLabel(f, text=subtitle, font=("Segoe UI", 8),
                     text_color=_FG_DIM).place(relx=0.5, rely=0.78, anchor="center")

        btn_refs[value] = (f, subtitle)
        f.bind("<Button-1>", lambda e, v=value: _select(v))
        for w in f.winfo_children():
            w.bind("<Button-1>", lambda e, v=value: _select(v))

    # ── Skip button ───────────────────────────────────────────────────────────
    skip_f = ctk.CTkFrame(top, fg_color=_CARD, corner_radius=8,
                          border_color=_BORDER, border_width=1,
                          height=36, cursor="hand2")
    skip_f.pack(fill="x", padx=24, pady=(0, 6))
    skip_f.pack_propagate(False)
    ctk.CTkLabel(skip_f, text="Skip  —  no wrong-way detection for this lane",
                 font=_F_SMALL, text_color=_FG_DIM).place(relx=0.5, rely=0.5, anchor="center")
    skip_f.bind("<Button-1>", lambda e: _select(""))
    for w in skip_f.winfo_children():
        w.bind("<Button-1>", lambda e: _select(""))
    btn_refs[""] = (skip_f, "skip")

    # ── Buttons ───────────────────────────────────────────────────────────────
    bf = ctk.CTkFrame(top, fg_color=_BG)
    bf.pack(fill="x", padx=20, pady=(4, 16), side="bottom")

    def confirm():
        if selected[0] is None:
            # nothing picked → treat as skip
            result[0] = ""
        else:
            result[0] = selected[0]
        top.destroy()

    def cancel():
        result[0] = ""
        top.destroy()

    ctk.CTkButton(bf, text="Confirm", font=_F_BODY, width=120, height=34,
                  fg_color=_TEAL, hover_color=_SEL_BOR, text_color=_BG,
                  command=confirm).pack(side="right", padx=(6, 0))
    ctk.CTkButton(bf, text="Cancel", font=_F_BODY, width=110, height=34,
                  fg_color=_CARD, hover_color=_SEL_BG,
                  border_color=_BORDER, border_width=1, text_color=_FG_MED,
                  command=cancel).pack(side="right")

    top.protocol("WM_DELETE_WINDOW", cancel)
    top.wait_window()
    return result[0] if result[0] is not None else ""


def _confirm_gui(title: str, message: str) -> bool:
    result = [False]
    top = _base_dialog(title, 380, 140)

    ctk.CTkLabel(top, text=message, font=_F_BODY, text_color=_FG_MED,
                 wraplength=340, justify="left").pack(pady=(22, 12), padx=20)

    bf = ctk.CTkFrame(top, fg_color=_BG)
    bf.pack(fill="x", padx=20, pady=(0, 16), side="bottom")

    def yes():
        result[0] = True
        top.destroy()

    ctk.CTkButton(bf, text="Yes, exit", font=_F_BODY, width=110, height=32,
                  fg_color=_CARD, hover_color="#3a1e1e",
                  border_color=_RED, border_width=1, text_color=_RED,
                  command=yes).pack(side="right", padx=(6, 0))
    ctk.CTkButton(bf, text="Keep editing", font=_F_BODY, width=120, height=32,
                  fg_color=_TEAL, hover_color=_SEL_BOR, text_color=_BG,
                  command=top.destroy).pack(side="right")

    top.protocol("WM_DELETE_WINDOW", top.destroy)
    top.wait_window()
    return result[0]


def _info_gui(title: str, message: str) -> None:
    top = _base_dialog(title, 360, 120)
    ctk.CTkLabel(top, text=message, font=_F_BODY, text_color=_FG_MED,
                 wraplength=320, justify="left").pack(pady=(22, 10), padx=20)
    ctk.CTkButton(top, text="OK", font=_F_BODY, width=90, height=30,
                  fg_color=_TEAL, hover_color=_SEL_BOR, text_color=_BG,
                  command=top.destroy).pack()
    top.protocol("WM_DELETE_WINDOW", top.destroy)
    top.wait_window()


# ── Frame acquisition ─────────────────────────────────────────────────────────
def _grab_frame(source: str) -> "np.ndarray | None":
    if source.startswith("http"):
        import requests, urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        try:
            with requests.get(source, stream=True, verify=False, timeout=10) as r:
                r.raise_for_status()
                buf = b""
                for chunk in r.iter_content(chunk_size=4096):
                    buf += chunk
                    start = buf.find(b"\xff\xd8")
                    end   = buf.find(b"\xff\xd9")
                    if start != -1 and end != -1 and end > start:
                        jpg   = buf[start:end + 2]
                        frame = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            return frame
        except Exception as e:
            print(f"[ROI Selector] Stream error: {e}")
        return None
    else:
        cap = cv2.VideoCapture(source)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None


# ── Save config ───────────────────────────────────────────────────────────────
def _build_lanes_dict(polys: list) -> dict:
    lanes = {}
    for poly in polys:
        xs  = [p[0] for p in poly["pts"]]
        ys  = [p[1] for p in poly["pts"]]
        roi = [min(xs), min(ys), max(xs), max(ys)]
        entry = {
            "roi":      roi,
            "points":   [list(p) for p in poly["pts"]],
            "label_pt": list(poly.get("label_pt", [roi[0], roi[3]])),
        }
        if poly.get("expected_direction"):
            entry["expected_direction"] = poly["expected_direction"]
        lanes[poly["name"]] = entry
    return lanes


def _save_config(polys, config_path, camera_id=None, cameras_dir=None):
    lanes = _build_lanes_dict(polys)
    if camera_id and cameras_dir:
        save_camera_lanes(camera_id, cameras_dir, lanes)
    else:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        cfg["lanes"] = lanes
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=4)
    print(f"[ROI Selector] {len(polys)} lane(s) saved.")


# ── Main ──────────────────────────────────────────────────────────────────────
def run(config_path=None, camera_id: str | None = None,
        stream_url: str | None = None, cameras_dir: Path | None = None):
    global polygons, current_pts, drag_idx, placing_label, label_preview
    global SCALING_FACTOR

    polygons      = []
    current_pts   = []
    drag_idx      = None
    placing_label = False
    label_preview = None

    if config_path is None:
        this_dir    = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.normpath(os.path.join(this_dir, '..', 'config.json'))
    if cameras_dir is None:
        cameras_dir = Path(os.path.dirname(os.path.abspath(config_path))) / 'cameras'

    # ── Frame source ──────────────────────────────────────────────────────────
    if stream_url:
        source = stream_url
    else:
        cfg        = load_config(config_path)
        video_path = cfg.camera.video_path
        if not os.path.isabs(video_path):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            video_path = os.path.normpath(os.path.join(config_dir, video_path))
        source    = video_path
        camera_id = camera_id or "local"

    frame = _grab_frame(source)
    if frame is None:
        _info_gui("Error", f"Cannot grab frame from:\n{source}")
        return

    # ── Screen size (reuse existing Tk/CTk root) ──────────────────────────────
    try:
        root = tk._default_root
        if root is not None:
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
        else:
            _tmp = tk.Tk(); _tmp.withdraw()
            screen_w = _tmp.winfo_screenwidth()
            screen_h = _tmp.winfo_screenheight()
            _tmp.destroy()
    except Exception:
        screen_w, screen_h = 1920, 1080

    vid_h, vid_w = frame.shape[:2]
    SCALING_FACTOR = min((screen_w - 40) / vid_w, (screen_h - 140) / vid_h)
    dw = int(vid_w * SCALING_FACTOR)
    dh = int(vid_h * SCALING_FACTOR)
    base = cv2.resize(frame, (dw, dh))

    cv2.namedWindow("ROI Selector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI Selector", dw, dh)
    cv2.setMouseCallback("ROI Selector", mouse_polygon)

    while True:
        display = base.copy()

        # Completed polygons
        for i, poly in enumerate(polygons):
            color = LANE_PALETTE[i % len(LANE_PALETTE)]
            _draw_polygon(display, poly["pts"], color, closed=True)
            if "label_pt" in poly:
                _draw_label(display, poly["name"],
                            poly["label_pt"][0], poly["label_pt"][1], color)

        # ── Label placement mode ──────────────────────────────────────────────
        if placing_label:
            color = LANE_PALETTE[(len(polygons) - 1) % len(LANE_PALETTE)]
            name  = polygons[-1]["name"]
            if label_preview:
                sx, sy = label_preview
                tw, th = _measure_text(name, 15)
                pad    = 6
                overlay = display.copy()
                cv2.rectangle(overlay, (sx - pad, sy - pad),
                              (sx + tw + pad, sy + th + pad), (15, 12, 22), -1)
                cv2.addWeighted(overlay, 0.72, display, 0.28, 0, display)
                cv2.rectangle(display, (sx - pad, sy - pad),
                              (sx + tw + pad, sy + th + pad), color, 1)
                _pil_text(display, name, sx, sy, 15, (230, 230, 230), shadow=False)
            _draw_status(display,
                         [f"Click where the '{name}' label should appear on the canvas"], dh)

        # ── Polygon drawing mode ──────────────────────────────────────────────
        else:
            active_color = LANE_PALETTE[len(polygons) % len(LANE_PALETTE)]
            _draw_polygon(display, current_pts, active_color, closed=False)
            last_has_label = polygons and "label_pt" in polygons[-1]
            line1 = (f"Lane {len(polygons) + 1} | Points: {len(current_pts)} | "
                     f"Enter=confirm  r=reset  R=reset all  q=save & quit")
            line2 = "'l' to reposition last label" if last_has_label else ""
            _draw_status(display, [line1, line2] if line2 else [line1], dh)

        cv2.imshow("ROI Selector", display)
        key = cv2.waitKey(20) & 0xFF

        if placing_label:
            continue

        if key in (13, 32):   # Enter / Space — confirm polygon
            if len(current_pts) < 3:
                _info_gui("Warning", "At least 3 points are required to define a lane.")
            else:
                cv2.setMouseCallback("ROI Selector", lambda *a: None)
                name = _ask_name_gui(polygons)
                if name is None:
                    cv2.setMouseCallback("ROI Selector", mouse_polygon)
                else:
                    direction = _ask_direction_gui(name)
                    polygons.append({
                        "name":               name,
                        "pts":                list(current_pts),
                        "expected_direction": direction,
                    })
                    current_pts   = []
                    placing_label = True
                    label_preview = None
                    cv2.setMouseCallback("ROI Selector", mouse_label)

        elif key == ord('l'):
            if polygons and "label_pt" in polygons[-1]:
                placing_label = True
                label_preview = None
                cv2.setMouseCallback("ROI Selector", mouse_label)

        elif key == ord('r'):
            current_pts = []

        elif key == ord('R'):
            polygons    = []
            current_pts = []

        elif key == ord('q'):
            if current_pts:
                if not _confirm_gui("Unsaved points",
                                    f"{len(current_pts)} unconfirmed point(s) will be discarded.\n"
                                    "Exit anyway?"):
                    continue
            if not polygons:
                _info_gui("Nothing to save", "No lanes defined. Exiting without changes.")
            else:
                _save_config(polygons, config_path,
                             camera_id=camera_id, cameras_dir=cameras_dir)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
