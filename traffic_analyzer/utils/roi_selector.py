import cv2
import json
import os
import numpy as np
from traffic_analyzer.utils.config_loader import load_config
from traffic_analyzer.visualization.colors import LANE_PALETTE

SNAP_RADIUS = 10
SCALING_FACTOR = 0.5  # updated at runtime by run()

# ── Global state ──────────────────────────────────────────────────────────────
polygons = []  # [{"name": str, "pts": [(ox,oy),...], "label_pt": (ox,oy)}, ...]
current_pts = []
drag_idx = None
placing_label = False  # True while waiting for label click
label_preview = None  # (sx, sy) screen coords for live preview


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
        print(f"  Label position set. Press 'l' to reposition, or start next polygon.")
        cv2.setMouseCallback("ROI Selector", mouse_polygon)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _draw_polygon(display, pts, color, closed=False):
    if not pts:
        return
    spts = [_to_screen(ox, oy) for ox, oy in pts]
    for i in range(len(spts) - 1):
        cv2.line(display, spts[i], spts[i + 1], color, 2)
    if closed and len(spts) > 2:
        cv2.line(display, spts[-1], spts[0], color, 2)
        overlay = display.copy()
        cv2.fillPoly(overlay, [np.array(spts)], color)
        cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
    for i, sp in enumerate(spts):
        cv2.circle(display, sp, SNAP_RADIUS, color, -1)
        cv2.circle(display, sp, SNAP_RADIUS, (255, 255, 255), 1)
        cv2.putText(display, str(i + 1), (sp[0] + 8, sp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)


def _draw_label(display, text, ox, oy, color):
    sx, sy = _to_screen(ox, oy)
    cv2.putText(display, text, (sx, sy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(display, text, (sx, sy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
    cv2.drawMarker(display, (sx, sy), color, cv2.MARKER_CROSS, 10, 1)


def _draw_status(display, lines, h):
    """Draw one or two lines at the bottom."""
    bar_h = 20 * len(lines) + 8
    cv2.rectangle(display, (0, h - bar_h), (display.shape[1], h), (20, 20, 20), -1)
    for i, text in enumerate(lines):
        cv2.putText(display, text, (8, h - bar_h + 16 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)


# ── Terminal helpers ──────────────────────────────────────────────────────────
def _ask_name():
    existing = [p["name"] for p in polygons]
    while True:
        name = input(f"  Name for polygon {len(polygons) + 1}: ").strip()
        if not name:
            print("  Name cannot be empty.")
        elif name in existing:
            print(f"  '{name}' already used.")
        else:
            return name


# ── Save config ───────────────────────────────────────────────────────────────
def _save_config(polygons, config_path):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    lanes = {}
    for poly in polygons:
        xs = [p[0] for p in poly["pts"]]
        ys = [p[1] for p in poly["pts"]]
        roi = [min(xs), min(ys), max(xs), max(ys)]
        lanes[poly["name"]] = {
            "roi": roi,
            "points": [list(p) for p in poly["pts"]],
            "label_pt": list(poly.get("label_pt", [roi[0], roi[3]])),
        }
    cfg["lanes"] = lanes
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=4)
    print(f"\n  config.json updated — {len(polygons)} lane(s) saved.")
    for name, data in lanes.items():
        print(f"    {name}: label_pt={data['label_pt']}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run(config_path=None):
    global polygons, current_pts, drag_idx, placing_label, label_preview

    global SCALING_FACTOR
    polygons = []
    current_pts = []
    drag_idx = None
    placing_label = False
    label_preview = None

    if config_path is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.normpath(os.path.join(this_dir, '..', 'config.json'))

    cfg = load_config(config_path)
    video_path = cfg.camera.video_path
    if not os.path.isabs(video_path):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        video_path = os.path.normpath(os.path.join(config_dir, video_path))

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    # Auto-fit: detect screen resolution and scale video to fit
    try:
        import tkinter as _tk
        _root = _tk.Tk()
        _root.withdraw()
        screen_w = _root.winfo_screenwidth()
        screen_h = _root.winfo_screenheight()
        _root.destroy()
    except Exception:
        screen_w, screen_h = 1920, 1080  # safe fallback

    vid_h, vid_w = frame.shape[:2]
    max_w = screen_w - 20
    max_h = screen_h - 120  # leave room for taskbar + window title
    scale_w = max_w / vid_w
    scale_h = max_h / vid_h
    SCALING_FACTOR = min(scale_w, scale_h, 1.0)  # never upscale

    dw = int(vid_w * SCALING_FACTOR)
    dh = int(vid_h * SCALING_FACTOR)
    base = cv2.resize(frame, (dw, dh))

    cv2.namedWindow("ROI Selector")
    cv2.setMouseCallback("ROI Selector", mouse_polygon)

    print(f"[ROI Selector] {video_path} | Scale {int(SCALING_FACTOR * 100)}%")
    print("Left click    → add point  |  Drag → reposition  |  Right click → remove last")
    print("Enter/Space   → confirm polygon, type name, click label position")
    print("'l'           → reposition last label")
    print("'r'           → reset active polygon  |  'R' → reset all  |  'q' → save & exit\n")

    while True:
        display = base.copy()

        # Completed polygons
        for i, poly in enumerate(polygons):
            color = LANE_PALETTE[i % len(LANE_PALETTE)]
            _draw_polygon(display, poly["pts"], color, closed=True)
            if "label_pt" in poly:
                _draw_label(display, poly["name"],
                            poly["label_pt"][0], poly["label_pt"][1], color)

        # ── Label placement mode ──────────────────────────────────────────
        if placing_label:
            color = LANE_PALETTE[(len(polygons) - 1) % len(LANE_PALETTE)]
            name = polygons[-1]["name"]
            if label_preview:
                sx, sy = label_preview
                cv2.putText(display, name, (sx, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(display, name, (sx, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
            _draw_status(display, [f"Click where '{name}' label should appear"], dh)

        # ── Polygon drawing mode ──────────────────────────────────────────
        else:
            active_color = LANE_PALETTE[len(polygons) % len(LANE_PALETTE)]
            _draw_polygon(display, current_pts, active_color, closed=False)

            last_has_label = polygons and "label_pt" in polygons[-1]
            line1 = (f"Polygon {len(polygons) + 1}  |  Points: {len(current_pts)}  |  "
                     f"Enter=confirm  r=reset  R=reset all  q=save & exit")
            line2 = "'l' → reposition last label" if last_has_label else ""
            _draw_status(display, [line1, line2] if line2 else [line1], dh)

        cv2.imshow("ROI Selector", display)
        key = cv2.waitKey(20) & 0xFF

        if placing_label:
            continue

        # ── Confirm polygon ───────────────────────────────────────────────
        if key in (13, 32):
            if len(current_pts) < 3:
                print("  Need at least 3 points to confirm.")
            else:
                cv2.setMouseCallback("ROI Selector", lambda *a: None)
                name = _ask_name()
                polygons.append({"name": name, "pts": list(current_pts)})
                current_pts = []
                placing_label = True
                label_preview = None
                print(f"  '{name}' confirmed. Click where the label should appear.")
                cv2.setMouseCallback("ROI Selector", mouse_label)

        # ── Reposition last label ─────────────────────────────────────────
        elif key == ord('l'):
            if polygons and "label_pt" in polygons[-1]:
                placing_label = True
                label_preview = None
                print(f"  Repositioning label for '{polygons[-1]['name']}'. Click new position.")
                cv2.setMouseCallback("ROI Selector", mouse_label)
            else:
                print("  No label to reposition yet.")

        elif key == ord('r'):
            current_pts = []
            print("  Active polygon reset.")

        elif key == ord('R'):
            polygons = []
            current_pts = []
            print("  Full reset.")

        elif key == ord('q'):
            if current_pts:
                print(f"\n  WARNING: {len(current_pts)} unconfirmed point(s).")
                answer = input("  Discard and exit? (y/n): ").strip().lower()
                if answer != 'y':
                    continue
            if not polygons:
                print("  No polygons to save. Exiting without changes.")
            else:
                _save_config(polygons, config_path)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
