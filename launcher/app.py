# -*- coding: utf-8 -*-
"""launcher/app.py — Smart Traffic Analyzer launcher / composition root"""
from __future__ import annotations
import threading
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

_ROOT        = Path(__file__).parent.parent


def _load_user_fonts(family_stem: str) -> None:
    """Load user-installed TTF files into GDI so tkinter can use them.

    Fonts installed per-user land in AppData/Local/Microsoft/Windows/Fonts
    but are NOT visible to tkinter's font.families() until explicitly loaded
    via AddFontResourceExW.  Call this before the first CTk window is created.
    """
    try:
        import ctypes
        import os
        font_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts"
        for ttf in font_dir.glob(f"{family_stem}*.ttf"):
            ctypes.windll.gdi32.AddFontResourceExW(str(ttf), 0x10, 0)
    except Exception:
        pass


_load_user_fonts("PlusJakartaSans")
_CFG_PATH    = _ROOT / "traffic_analyzer" / "config.json"
_CAMERAS_DIR = _ROOT / "traffic_analyzer" / "cameras"

def _cam_file_id(cam) -> str:
    """Camera key used as a filename — camera name with spaces replaced by underscores."""
    return cam.name.replace(" ", "_")


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Midnight Navy palette ──────────────────────────────────────────────────────
# Dark mode: navy-based, not pure black — easy on the eyes, edgy and deep.
BG       = "#0c0e1a"   # Midnight navy  — window background
SURFACE  = "#10132a"   # Deep navy      — title bar / panel surface
CARD     = "#161930"   # Navy card      — card background
BORDER   = "#252a50"   # Blue-slate     — border
SEL_BG   = "#111535"   # Indigo-navy    — selected item background
SEL_BOR  = "#6070e8"   # Soft indigo    — selected border

FG       = "#d6daf8"   # Blue-white     — primary text
FG_MED   = "#7880b8"   # Mid blue-grey  — secondary text
FG_DIM   = "#3a3f6a"   # Dim blue       — hint / disabled text

BLUE     = "#6070e8"   # Soft indigo    — primary
TEAL     = "#6070e8"   # Soft indigo    — primary accent
TEAL_H   = "#5060d8"   # Dark indigo    — hover
GREEN    = "#3dd68c"   # Soft emerald   — success
GREEN_D  = "#0a2218"   # Dark emerald   — success background
YELLOW   = "#e8a020"   # Soft amber     — warning
YELLOW_D = "#251a05"   # Dark amber     — warning background
RED      = "#e05575"   # Soft rose-red  — error / danger

# ── Typography ─────────────────────────────────────────────────────────────────
_PF     = "Plus Jakarta Sans"
F_TITLE = (_PF, 17, "bold")
F_HEAD  = (_PF, 12, "bold")
F_BODY  = (_PF, 11)
F_SMALL = (_PF, 10)
F_MONO  = ("Consolas", 10)


# ── Dialog helpers ────────────────────────────────────────────────────────────
# Custom CTk dialogs matching the Midnight Navy design system.
# All text is in English; all windows are centred on screen.

def _mk_dialog(parent, title: str, w: int, h: int) -> ctk.CTkToplevel:
    top = ctk.CTkToplevel(parent)
    top.title(title)
    top.configure(fg_color=BG)
    top.resizable(False, False)
    top.attributes("-topmost", True)
    top.update_idletasks()
    sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
    top.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
    top.grab_set()
    return top


def _dlg_info(parent, msg: str, title: str = "Notice") -> None:
    """Single-button informational / warning dialog."""
    top = _mk_dialog(parent, title, 420, 148)
    ctk.CTkLabel(top, text=msg, font=F_BODY, text_color=FG_MED,
                 wraplength=376, justify="left").pack(pady=(24, 14), padx=22)
    ctk.CTkButton(top, text="OK", font=F_BODY, width=90, height=34,
                  corner_radius=8, fg_color=TEAL, hover_color=TEAL_H,
                  text_color="#ffffff", command=top.destroy).pack()
    top.protocol("WM_DELETE_WINDOW", top.destroy)
    top.wait_window()


def _dlg_confirm(parent, title: str, msg: str,
                 ok_text: str = "Confirm",
                 cancel_text: str = "Cancel") -> bool:
    """Two-button confirmation dialog — returns True when the primary action is chosen."""
    result = [False]
    top    = _mk_dialog(parent, title, 440, 168)

    ctk.CTkLabel(top, text=msg, font=F_BODY, text_color=FG_MED,
                 wraplength=396, justify="left").pack(pady=(22, 10), padx=22)

    bf = ctk.CTkFrame(top, fg_color=BG)
    bf.pack(fill="x", padx=22, pady=(0, 18), side="bottom")

    def _yes():
        result[0] = True
        top.destroy()

    ctk.CTkButton(bf, text=ok_text, font=F_BODY, width=110, height=34,
                  corner_radius=8, fg_color=TEAL, hover_color=TEAL_H,
                  text_color="#ffffff", command=_yes).pack(side="right", padx=(8, 0))
    ctk.CTkButton(bf, text=cancel_text, font=F_BODY, width=90, height=34,
                  corner_radius=8, fg_color=CARD, hover_color=SEL_BG,
                  border_color=BORDER, border_width=1,
                  text_color=FG_MED, command=top.destroy).pack(side="right")

    top.protocol("WM_DELETE_WINDOW", top.destroy)
    top.wait_window()
    return result[0]


# ─────────────────────────────────────────────────────────────────────────────
class CameraRow(ctk.CTkFrame):
    """Single-row camera card — compact list item."""

    H = 40   # row height in pixels

    def __init__(self, master, camera, has_roi: bool, on_select, **kw):
        super().__init__(master, height=self.H, corner_radius=8,
                         fg_color=CARD, border_color=BORDER,
                         border_width=1, cursor="hand2", **kw)
        self.pack_propagate(False)
        self._cam       = camera
        self._has_roi   = has_roi
        self._on_select = on_select
        self._selected  = False
        self._build()
        for w in [self] + list(self.winfo_children()):
            w.bind("<Button-1>", self._click)

    def _build(self):
        # ROI status dot (left)
        self._dot = ctk.CTkLabel(self, text="●", width=18,
                                  font=(_PF, 10),
                                  text_color=GREEN if self._has_roi else FG_DIM)
        self._dot.pack(side="left", padx=(10, 2))

        # Camera name — double space compensates for Plus Jakarta Sans rendering
        # single spaces as near-invisible in tkinter's GDI context.
        display_name = self._cam.name.replace(" ", "  ")
        self._name = ctk.CTkLabel(self, text=display_name,
                                   font=F_BODY, text_color=FG, anchor="w")
        self._name.pack(side="left", fill="x", expand=True)

        # Camera ID (right-aligned, small)
        self._id = ctk.CTkLabel(self, text=self._cam.camera_id,
                                 font=F_MONO, text_color=FG_DIM, width=130, anchor="e")
        self._id.pack(side="right", padx=(0, 10))

    def set_selected(self, v: bool):
        if self._selected == v:
            return
        self._selected = v
        if v:
            self.configure(fg_color=SEL_BG, border_color=SEL_BOR, border_width=2)
            self._name.configure(text_color=TEAL)
        else:
            self.configure(fg_color=CARD, border_color=BORDER, border_width=1)
            self._name.configure(text_color=FG)

    def refresh_roi(self):
        self._has_roi = (_CAMERAS_DIR / f"{_cam_file_id(self._cam)}.json").exists()
        self._dot.configure(text_color=GREEN if self._has_roi else FG_DIM)

    def _click(self, _e=None):
        self._on_select(self)


# ─────────────────────────────────────────────────────────────────────────────
class LauncherApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Smart Traffic Analyzer")
        self.minsize(860, 620)
        self.configure(fg_color=BG)

        self._cameras: list              = []
        self._all_rows: list[CameraRow] = []
        self._sel_row:  CameraRow | None = None
        self._source    = ctk.StringVar(value="live")
        self._search    = ctk.StringVar()
        self._filepath  = ctk.StringVar()
        self._job       = None

        self._build()
        self._center()
        self._prefill()
        self._load_async()

    # ──────────────────────────────────────────────────────────────────────────
    def _build(self):
        # Title bar — white surface with bottom separator
        top = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0, height=58)
        top.pack(fill="x")
        top.pack_propagate(False)

        # App name in Indigo 600 (Corporate Trust primary)
        ctk.CTkLabel(top, text="Smart Traffic Analyzer",
                     font=F_TITLE, text_color=TEAL).pack(
            side="left", padx=22, pady=14)

        self._status_lbl = ctk.CTkLabel(
            top, text="", font=F_SMALL, text_color=FG_DIM)
        self._status_lbl.pack(side="right", padx=22)

        # Slate 200 hairline divider
        ctk.CTkFrame(self, height=1, fg_color=BORDER,
                     corner_radius=0).pack(fill="x")

        # Source selector
        src = ctk.CTkFrame(self, fg_color=BG, corner_radius=0, height=46)
        src.pack(fill="x", padx=20, pady=(12, 0))
        src.pack_propagate(False)

        ctk.CTkLabel(src, text="Kaynak:", font=F_SMALL,
                     text_color=FG_DIM).pack(side="left", padx=(0, 12))

        for txt, val in [("Canlı Kamera", "live"), ("Video Dosyası", "file")]:
            ctk.CTkRadioButton(
                src, text=txt, variable=self._source, value=val,
                command=self._toggle_source,
                font=F_BODY, text_color=FG,
                fg_color=TEAL, hover_color=TEAL_H,
            ).pack(side="left", padx=(0, 24))

        # Content panels
        self._live = self._make_live_panel()
        self._file = self._make_file_panel()

        # Bottom action bar
        self._make_bottom()
        self._toggle_source()

    # ── Live camera panel ─────────────────────────────────────────────────────
    def _make_live_panel(self):
        p = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)

        # Search row
        row = ctk.CTkFrame(p, fg_color=BG, corner_radius=0)
        row.pack(fill="x", padx=20, pady=(8, 6))

        self._search_entry = ctk.CTkEntry(
            row, textvariable=self._search,
            placeholder_text="Kamera ara...",
            font=F_BODY, height=36, corner_radius=8,
            fg_color=SURFACE, border_color=BORDER, text_color=FG,
            placeholder_text_color=FG_DIM,
        )
        self._search_entry.pack(side="left", fill="x", expand=True)

        self._count = ctk.CTkLabel(row, text="", font=F_SMALL,
                                    text_color=FG_DIM, width=88, anchor="e")
        self._count.pack(side="left", padx=(10, 0))

        self._search.trace_add("write", self._debounce)

        # Scrollable camera list — Slate 50 bg so white cards "pop"
        self._scroll = ctk.CTkScrollableFrame(
            p, fg_color=BG,
            border_color=BORDER, border_width=1, corner_radius=10,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=FG_DIM,
        )
        self._scroll.pack(fill="both", expand=True, padx=20, pady=(0, 6))

        self._load_lbl = ctk.CTkLabel(
            self._scroll, text="Kameralar yükleniyor...",
            font=F_BODY, text_color=FG_DIM)
        self._load_lbl.pack(pady=36)

        # Selected camera info bar
        info = ctk.CTkFrame(p, fg_color=SURFACE, corner_radius=10,
                             border_color=BORDER, border_width=1, height=38)
        info.pack(fill="x", padx=20, pady=(0, 4))
        info.pack_propagate(False)

        self._info_name = ctk.CTkLabel(info, text="Kamera seçilmedi",
                                        font=F_BODY, text_color=FG_DIM, anchor="w")
        self._info_name.pack(side="left", padx=14)

        # Badge: transparent until a camera is selected
        self._roi_badge = ctk.CTkLabel(info, text="",
                                        font=F_SMALL, text_color=FG_DIM,
                                        fg_color="transparent", corner_radius=6,
                                        width=98)
        self._roi_badge.pack(side="right", padx=10)

        return p

    # ── File panel ────────────────────────────────────────────────────────────
    def _make_file_panel(self):
        p = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)

        inner = ctk.CTkFrame(p, fg_color=SURFACE, corner_radius=12,
                              border_color=BORDER, border_width=1)
        inner.pack(padx=20, pady=20, fill="x")

        ctk.CTkLabel(inner, text="Video dosyası seçin",
                     font=F_HEAD, text_color=FG_MED).pack(
            anchor="w", padx=18, pady=(16, 8))

        row = ctk.CTkFrame(inner, fg_color=SURFACE, corner_radius=0)
        row.pack(fill="x", padx=18, pady=(0, 16))

        ctk.CTkEntry(row, textvariable=self._filepath,
                     placeholder_text="Dosya yolu...",
                     font=F_BODY, height=38, corner_radius=8,
                     fg_color=CARD, border_color=BORDER,
                     text_color=FG).pack(side="left", fill="x", expand=True)

        # Secondary button: white bg, slate border, indigo text
        ctk.CTkButton(row, text="Gözat", width=84, height=38,
                      corner_radius=8,
                      fg_color=SURFACE, hover_color=SEL_BG,
                      border_color=BORDER, border_width=1,
                      text_color=TEAL, font=F_BODY,
                      command=self._browse).pack(side="left", padx=(8, 0))

        return p

    # ── Bottom action bar ─────────────────────────────────────────────────────
    def _make_bottom(self):
        # Slate 200 hairline divider
        ctk.CTkFrame(self, height=1, fg_color=BORDER,
                     corner_radius=0).pack(fill="x")

        bar = ctk.CTkFrame(self, fg_color=SURFACE,
                           corner_radius=0, height=62)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        bf = ctk.CTkFrame(bar, fg_color=SURFACE, corner_radius=0)
        bf.pack(side="right", padx=20, pady=11)

        # Secondary action: white bg, slate border, indigo text
        self._roi_btn = ctk.CTkButton(
            bf, text="ROI Düzenle", font=F_BODY,
            height=40, width=160, corner_radius=8,
            fg_color=SURFACE, hover_color=SEL_BG,
            border_color=BORDER, border_width=1,
            text_color=TEAL,
            command=self._roi,
        )
        self._roi_btn.pack(side="left", padx=(0, 10))

        # Primary CTA: Indigo 600, pill shape, white text
        self._start_btn = ctk.CTkButton(
            bf, text="▶  Start",
            font=(_PF, 10, "bold"),
            height=40, width=148, corner_radius=20,
            fg_color=TEAL, hover_color=TEAL_H,
            text_color="#FFFFFF",
            command=self._start,
        )
        self._start_btn.pack(side="left")

    # ──────────────────────────────────────────────────────────────────────────
    def _toggle_source(self):
        if self._source.get() == "live":
            self._file.pack_forget()
            self._live.pack(fill="both", expand=True)
        else:
            self._live.pack_forget()
            self._file.pack(fill="both", expand=True)

    # ──────────────────────────────────────────────────────────────────────────
    def _load_async(self):
        threading.Thread(target=self._fetch, daemon=True).start()

    def _fetch(self):
        try:
            from traffic_analyzer.adapters.camera.camera_fetcher import fetch_cameras
            self._cameras = [c for c in fetch_cameras() if c.stream_url]
            self.after(0, self._build_rows)
        except Exception as e:
            self.after(0, lambda: self._set_status(f"Error: {e}", RED))

    def _build_rows(self):
        if self._load_lbl.winfo_exists():
            self._load_lbl.destroy()
        for cam in self._cameras:
            has_roi = (_CAMERAS_DIR / f"{_cam_file_id(cam)}.json").exists()
            r = CameraRow(self._scroll, cam, has_roi, self._on_select)
            r.pack(fill="x", padx=4, pady=2)
            self._all_rows.append(r)
        n = len(self._cameras)
        self._count.configure(text=f"{n} kamera")
        self._set_status(f"{n} kamera yüklendi.", GREEN)

    # ──────────────────────────────────────────────────────────────────────────
    def _debounce(self, *_):
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(200, self._filter)

    def _filter(self):
        q = self._search.get().lower()
        visible = 0
        for r in self._all_rows:
            show = not q or q in r._cam.name.lower() or q in r._cam.camera_id.lower()
            if show:
                r.pack(fill="x", padx=4, pady=2)
                visible += 1
            else:
                r.pack_forget()
        n = len(self._cameras)
        self._count.configure(text=f"{visible}/{n}" if q else f"{n} kamera")

    # ──────────────────────────────────────────────────────────────────────────
    def _on_select(self, row: CameraRow):
        if self._sel_row and self._sel_row is not row:
            self._sel_row.set_selected(False)
        row.set_selected(True)
        self._sel_row = row
        cam = row._cam
        self._info_name.configure(text=cam.name, text_color=FG)
        has_roi = (_CAMERAS_DIR / f"{_cam_file_id(cam)}.json").exists()
        if has_roi:
            # Koyu emerald bg, parlak emerald metin — dark mode badge
            self._roi_badge.configure(text="✓ ROI Mevcut",
                                       fg_color=GREEN_D, text_color=GREEN)
        else:
            # Koyu amber bg, parlak amber metin — dark mode badge
            self._roi_badge.configure(text="ROI Yok",
                                       fg_color=YELLOW_D, text_color=YELLOW)

    # ──────────────────────────────────────────────────────────────────────────
    def _browse(self):
        p = filedialog.askopenfilename(
            title="Video seç",
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov"), ("Tümü", "*.*")])
        if p:
            self._filepath.set(p)

    def _prefill(self):
        try:
            from traffic_analyzer.infrastructure.config_loader import load_config
            self._filepath.set(load_config(str(_CFG_PATH)).camera.video_path)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    def _roi(self):
        if self._source.get() == "live":
            if not self._sel_row:
                _dlg_info(self, "Select a camera first.")
                return
            cam = self._sel_row._cam
            try:
                from traffic_analyzer.visualization.roi_selector import run as roi_run
                roi_run(config_path=str(_CFG_PATH), camera_id=_cam_file_id(cam),
                        stream_url=cam.stream_url, cameras_dir=_CAMERAS_DIR)
            finally:
                self._sel_row.refresh_roi()
                self._on_select(self._sel_row)
                self._set_status("ROI saved.", GREEN)
                self._bring_to_front()
        else:
            if not self._filepath.get().strip():
                _dlg_info(self, "Select a video file first.")
                return
            try:
                from traffic_analyzer.visualization.roi_selector import run as roi_run
                roi_run(config_path=str(_CFG_PATH), camera_id="local",
                        cameras_dir=_CAMERAS_DIR)
            finally:
                self._set_status("ROI saved.", GREEN)
                self._bring_to_front()

    # ──────────────────────────────────────────────────────────────────────────
    def _start(self):
        if self._source.get() == "live":
            if not self._sel_row:
                _dlg_info(self, "Select a camera first.")
                return
            cam = self._sel_row._cam
            src, lbl = cam.stream_url, cam.name
            cid = cam.name.replace(" ", "_")
        else:
            vp = self._filepath.get().strip()
            if not vp:
                _dlg_info(self, "Select a video file first.")
                return
            vp_path = Path(vp)
            if not vp_path.is_absolute():
                vp_path = (_CFG_PATH.parent / vp_path).resolve()
            src, cid, lbl = str(vp_path), "local", vp_path.name

        # ROI check
        try:
            from traffic_analyzer.infrastructure.config_loader import load_camera_lanes
            has_lanes = bool(load_camera_lanes(cid, _CAMERAS_DIR))
        except Exception as e:
            _dlg_info(self, f"Could not load ROI:\n{e}", title="Error")
            return

        if not has_lanes:
            if not _dlg_confirm(
                self,
                "No ROI Defined",
                f"{lbl}\n\n"
                "No ROI has been defined for this camera.\n"
                "Only visual analysis will run — data will not be written\n"
                "to Kafka, Spark, or InfluxDB.\n\n"
                "Continue anyway?",
                ok_text="Continue",
            ):
                return
        else:
            if not _dlg_confirm(self, "Start Analysis", f"{lbl}\n\nStart the analysis?",
                                ok_text="Start"):
                return

        self._start_btn.configure(state="disabled", text="Running...")
        self._set_status(f"Connecting: {lbl}", YELLOW)
        self.withdraw()

        def _on_done():
            self.deiconify()
            self.update()          # force full repaint before raising window
            self._bring_to_front()
            self._start_btn.configure(state="normal", text="▶  Start")
            self._set_status("Analysis complete.", GREEN)

        def _run():
            try:
                self._pipeline(src, cid, visual_only=not has_lanes)
            except Exception as e:
                print(f"[Launcher] Pipeline error: {e}")
            finally:
                # destroyAllWindows is already called inside video_loop.py's finally.
                # DO NOT call waitKey(1) here — on Windows it blocks indefinitely when
                # invoked from a background thread after all HighGUI windows are gone,
                # preventing self.after(0, _on_done) from ever running.
                import cv2 as _cv2
                _cv2.destroyAllWindows()
                self.after(0, _on_done)

        threading.Thread(target=_run, daemon=True).start()

    def _pipeline(self, video_source: str, camera_id: str, visual_only: bool = False):
        import torch
        from traffic_analyzer.infrastructure.config_loader import load_config, load_camera_lanes
        from traffic_analyzer.adapters.detector import RFDETRDetector
        from traffic_analyzer.adapters.tracker import ByteTracker
        from traffic_analyzer.application.event_builder import EventBuilder
        from traffic_analyzer.application.frame_processor import FrameProcessor
        from traffic_analyzer.infrastructure.video_loop import VideoLoop
        from traffic_analyzer.application.analyzer import Analyzer
        from traffic_analyzer.visualization.ghost_track_manager import GhostTrackManager, DEFAULT_GHOST_FRAMES
        from traffic_analyzer.visualization.frame_renderer      import FrameRenderer
        from traffic_analyzer.domain.ports import CompositePublisher, NullPublisher

        cfg    = load_config(str(_CFG_PATH))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.camera.source_id = camera_id
        cfg.lanes = load_camera_lanes(camera_id, _CAMERAS_DIR)

        model_path = cfg.model.model_path
        if not Path(model_path).is_absolute():
            model_path = str((_ROOT / model_path).resolve())

        detector = RFDETRDetector(
            model_path=model_path, device=device,
            threshold=cfg.model.threshold, resolution=cfg.model.resolution)
        tracker  = ByteTracker(lost_track_buffer=150,
                               min_matching_threshold=0.8, min_consecutive_frames=3)

        if visual_only:
            print("[Launcher] ROI tanimlanmamis — gorsel test modu, veri yazilmiyor.")
            publisher = NullPublisher()
        else:
            from traffic_analyzer.adapters.influx_publisher import InfluxPublisher
            influx = InfluxPublisher(camera_id=camera_id, url=cfg.influx.url,
                                     token=cfg.influx.token, org=cfg.influx.org,
                                     bucket=cfg.influx.bucket)
            try:
                from traffic_analyzer.adapters.kafka_producer import TrafficProducer
                publisher = CompositePublisher(TrafficProducer(), influx)
            except ImportError:
                print("[Launcher] kafka_layer bulunamadi — sadece InfluxDB kullaniliyor.")
                publisher = influx

        processor = FrameProcessor(
            detector=detector, tracker=tracker,
            event_builder=EventBuilder(cfg), renderer=FrameRenderer(),
            ghost_manager=GhostTrackManager(ghost_frames=DEFAULT_GHOST_FRAMES), config=cfg)
        Analyzer(
            video_loop=VideoLoop(video_path=video_source,
                                 display_width=cfg.camera.display_width),
            frame_processor=processor, publisher=publisher).run()

    # ──────────────────────────────────────────────────────────────────────────
    def _bring_to_front(self):
        self.update()
        self.lift()
        self.wm_attributes("-topmost", True)
        self.after(50, lambda: self.wm_attributes("-topmost", False))
        self.focus_force()

    def _set_status(self, msg: str, color: str = FG_DIM):
        self._status_lbl.configure(text=msg, text_color=color)

    def _center(self):
        w, h = 1060, 720
        self.geometry(f"{w}x{h}")
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")


def main():
    LauncherApp().mainloop()

if __name__ == "__main__":
    main()
