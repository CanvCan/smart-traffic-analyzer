# -*- coding: utf-8 -*-
"""launcher/app.py — Smart Traffic Analyzer Başlatıcı"""
from __future__ import annotations
import threading
from pathlib import Path
from tkinter import messagebox, filedialog

import customtkinter as ctk

_ROOT        = Path(__file__).parent.parent
_CFG_PATH    = _ROOT / "traffic_analyzer" / "config.json"
_CAMERAS_DIR = _ROOT / "traffic_analyzer" / "cameras"

def _cam_file_id(cam) -> str:
    """Dosya adı için kullanılan kamera anahtarı — kamera adı, boşluklar _ ile."""
    return cam.name.replace(" ", "_")


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Pastel koyu palet ──────────────────────────────────────────────────────
BG      = "#1e1e2e"
SURFACE = "#252535"
CARD    = "#2a2a3e"
BORDER  = "#383858"
SEL_BG  = "#1e2d3d"
SEL_BOR = "#7eb8c9"

FG      = "#cdd6f4"
FG_MED  = "#a6adc8"
FG_DIM  = "#585b70"

BLUE    = "#89b4fa"
TEAL    = "#7eb8c9"
GREEN   = "#a6e3a1"
GREEN_D = "#1e3a28"
YELLOW  = "#f9e2af"
YELLOW_D= "#2e2510"
RED     = "#f38ba8"

F_TITLE = ("Segoe UI", 17, "bold")
F_HEAD  = ("Segoe UI", 11, "bold")
F_BODY  = ("Segoe UI", 10)
F_SMALL = ("Segoe UI", 9)
F_MONO  = ("Consolas", 9)


# ─────────────────────────────────────────────────────────────────────────────
class CameraRow(ctk.CTkFrame):
    """Tek satır kamera kartı — kompakt."""

    H = 38   # piksel yükseklik

    def __init__(self, master, camera, has_roi: bool, on_select, **kw):
        super().__init__(master, height=self.H, corner_radius=6,
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
        # ROI noktası (sol)
        self._dot = ctk.CTkLabel(self, text="●", width=18,
                                  font=("Segoe UI", 10),
                                  text_color=GREEN if self._has_roi else FG_DIM)
        self._dot.pack(side="left", padx=(8, 2))

        # Kamera adı
        self._name = ctk.CTkLabel(self, text=self._cam.name,
                                   font=F_BODY, text_color=FG, anchor="w")
        self._name.pack(side="left", fill="x", expand=True)

        # ID (sağda küçük)
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
        self.geometry("780x580")
        self.minsize(700, 500)
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
        # Başlık
        top = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0, height=56)
        top.pack(fill="x")
        top.pack_propagate(False)

        ctk.CTkLabel(top, text="Smart Traffic Analyzer",
                     font=F_TITLE, text_color=TEAL).pack(
            side="left", padx=20, pady=14)

        self._status_lbl = ctk.CTkLabel(
            top, text="", font=F_SMALL, text_color=FG_DIM)
        self._status_lbl.pack(side="right", padx=20)

        ctk.CTkFrame(self, height=1, fg_color=BORDER,
                     corner_radius=0).pack(fill="x")

        # Kaynak seçici
        src = ctk.CTkFrame(self, fg_color=BG, corner_radius=0, height=44)
        src.pack(fill="x", padx=18, pady=(10, 0))
        src.pack_propagate(False)

        ctk.CTkLabel(src, text="Kaynak:", font=F_SMALL,
                     text_color=FG_DIM).pack(side="left", padx=(0, 10))

        for txt, val in [("Canlı Kamera", "live"), ("Video Dosyası", "file")]:
            ctk.CTkRadioButton(
                src, text=txt, variable=self._source, value=val,
                command=self._toggle_source,
                font=F_BODY, text_color=FG,
                fg_color=TEAL, hover_color=SEL_BOR,
            ).pack(side="left", padx=(0, 22))

        # İçerik
        self._live = self._make_live_panel()
        self._file = self._make_file_panel()

        # Alt butonlar
        self._make_bottom()
        self._toggle_source()

    # ── Canlı panel ───────────────────────────────────────────────────────────
    def _make_live_panel(self):
        p = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)

        # Arama
        row = ctk.CTkFrame(p, fg_color=BG, corner_radius=0)
        row.pack(fill="x", padx=18, pady=(8, 4))

        self._search_entry = ctk.CTkEntry(
            row, textvariable=self._search,
            placeholder_text="Kamera ara...",
            font=F_BODY, height=34,
            fg_color=CARD, border_color=BORDER, text_color=FG,
            placeholder_text_color=FG_DIM,
        )
        self._search_entry.pack(side="left", fill="x", expand=True)

        self._count = ctk.CTkLabel(row, text="", font=F_SMALL,
                                    text_color=FG_DIM, width=80, anchor="e")
        self._count.pack(side="left", padx=(8, 0))

        self._search.trace_add("write", self._debounce)

        # Liste
        self._scroll = ctk.CTkScrollableFrame(
            p, fg_color=SURFACE,
            border_color=BORDER, border_width=1, corner_radius=8,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=FG_DIM,
        )
        self._scroll.pack(fill="both", expand=True, padx=18, pady=(0, 4))

        self._load_lbl = ctk.CTkLabel(
            self._scroll, text="Kameralar yükleniyor...",
            font=F_BODY, text_color=FG_DIM)
        self._load_lbl.pack(pady=30)

        # Seçili kamera bilgisi — tek satır
        info = ctk.CTkFrame(p, fg_color=CARD, corner_radius=8,
                             border_color=BORDER, border_width=1, height=36)
        info.pack(fill="x", padx=18, pady=(0, 2))
        info.pack_propagate(False)

        self._info_name = ctk.CTkLabel(info, text="Kamera seçilmedi",
                                        font=F_BODY, text_color=FG_MED, anchor="w")
        self._info_name.pack(side="left", padx=12)

        self._roi_badge = ctk.CTkLabel(info, text="",
                                        font=F_SMALL, text_color=BG,
                                        fg_color=FG_DIM, corner_radius=4,
                                        width=90)
        self._roi_badge.pack(side="right", padx=10)

        return p

    # ── Dosya paneli ──────────────────────────────────────────────────────────
    def _make_file_panel(self):
        p = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)

        inner = ctk.CTkFrame(p, fg_color=SURFACE, corner_radius=10,
                              border_color=BORDER, border_width=1)
        inner.pack(padx=18, pady=18, fill="x")

        ctk.CTkLabel(inner, text="Video dosyası seçin",
                     font=F_HEAD, text_color=FG_MED).pack(
            anchor="w", padx=16, pady=(14, 8))

        row = ctk.CTkFrame(inner, fg_color=SURFACE, corner_radius=0)
        row.pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkEntry(row, textvariable=self._filepath,
                     placeholder_text="Dosya yolu...",
                     font=F_BODY, height=36,
                     fg_color=CARD, border_color=BORDER,
                     text_color=FG).pack(side="left", fill="x", expand=True)

        ctk.CTkButton(row, text="Gözat", width=80, height=36,
                      fg_color=CARD, hover_color=SEL_BG,
                      border_color=BORDER, border_width=1,
                      text_color=TEAL, font=F_BODY,
                      command=self._browse).pack(side="left", padx=(6, 0))

        return p

    # ── Alt butonlar ──────────────────────────────────────────────────────────
    def _make_bottom(self):
        ctk.CTkFrame(self, height=1, fg_color=BORDER,
                     corner_radius=0).pack(fill="x")

        bar = ctk.CTkFrame(self, fg_color=SURFACE,
                           corner_radius=0, height=60)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        bf = ctk.CTkFrame(bar, fg_color=SURFACE, corner_radius=0)
        bf.pack(side="right", padx=18, pady=10)

        self._roi_btn = ctk.CTkButton(
            bf, text="ROI Düzenle", font=F_BODY,
            height=38, width=160,
            fg_color=CARD, hover_color=SEL_BG,
            border_color=BORDER, border_width=1,
            text_color=TEAL,
            command=self._roi,
        )
        self._roi_btn.pack(side="left", padx=(0, 8))

        self._start_btn = ctk.CTkButton(
            bf, text="▶  Başlat",
            font=("Segoe UI", 10, "bold"),
            height=38, width=140,
            fg_color=TEAL, hover_color=SEL_BOR,
            text_color=BG,
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
            self.after(0, lambda: self._set_status(f"Hata: {e}", RED))

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
            self._roi_badge.configure(text="✓ ROI Mevcut",
                                       fg_color=GREEN, text_color=GREEN_D)
        else:
            self._roi_badge.configure(text="ROI Yok",
                                       fg_color=YELLOW, text_color=YELLOW_D)

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
                messagebox.showwarning("Uyarı", "Önce bir kamera seçin.")
                return
            cam = self._sel_row._cam
            self.withdraw()
            try:
                from traffic_analyzer.visualization.roi_selector import run as roi_run
                roi_run(config_path=str(_CFG_PATH), camera_id=_cam_file_id(cam),
                        stream_url=cam.stream_url, cameras_dir=_CAMERAS_DIR)
            finally:
                self.deiconify()
                self._sel_row.refresh_roi()
                self._on_select(self._sel_row)
                self._set_status("ROI kaydedildi.", GREEN)
        else:
            if not self._filepath.get().strip():
                messagebox.showwarning("Uyarı", "Önce bir video dosyası seçin.")
                return
            self.withdraw()
            try:
                from traffic_analyzer.visualization.roi_selector import run as roi_run
                roi_run(config_path=str(_CFG_PATH), camera_id="local",
                        cameras_dir=_CAMERAS_DIR)
            finally:
                self.deiconify()
                self._set_status("ROI kaydedildi.", GREEN)

    # ──────────────────────────────────────────────────────────────────────────
    def _start(self):
        if self._source.get() == "live":
            if not self._sel_row:
                messagebox.showwarning("Uyarı", "Önce bir kamera seçin.")
                return
            cam = self._sel_row._cam
            src, lbl = cam.stream_url, cam.name
            cid = cam.name.replace(" ", "_")
        else:
            vp = self._filepath.get().strip()
            if not vp:
                messagebox.showwarning("Uyarı", "Önce bir video dosyası seçin.")
                return
            # Göreli yolları traffic_analyzer/ klasörüne göre çöz
            vp_path = Path(vp)
            if not vp_path.is_absolute():
                vp_path = (_CFG_PATH.parent / vp_path).resolve()
            src, cid, lbl = str(vp_path), "local", vp_path.name

        # ROI kontrolü
        try:
            from traffic_analyzer.infrastructure.config_loader import load_camera_lanes
            has_lanes = bool(load_camera_lanes(cid, _CAMERAS_DIR))
        except Exception as e:
            messagebox.showerror("Hata", f"ROI yuklenemedi: {e}")
            return

        if not has_lanes:
            ans = messagebox.askyesno(
                "ROI Tanımlanmamış",
                f"{lbl}\n\n"
                "Bu kamera için ROI tanımlanmamış.\n"
                "Sadece gorsel test analizi yapılacak — veriler Kafka, Spark\n"
                "veya InfluxDB'ye yazılmayacak.\n\n"
                "Devam edilsin mi?"
            )
            if not ans:
                return
        else:
            if not messagebox.askyesno("Analizi Başlat", f"{lbl}\n\nBaşlatılsın mı?"):
                return

        self._start_btn.configure(state="disabled", text="Çalışıyor...")
        self._set_status(f"Bağlanıyor: {lbl}", YELLOW)
        self.withdraw()
        try:
            self._pipeline(src, cid, visual_only=not has_lanes)
        finally:
            self.deiconify()
            self._start_btn.configure(state="normal", text="▶  Başlat")
            self._set_status("Analiz tamamlandı.", GREEN)

    def _pipeline(self, video_source: str, camera_id: str, visual_only: bool = False):
        import torch
        from traffic_analyzer.infrastructure.config_loader import load_config, load_camera_lanes
        from traffic_analyzer.adapters.detector import RFDETRDetector
        from traffic_analyzer.adapters.tracker import ByteTracker
        from traffic_analyzer.application.event_builder import EventBuilder
        from traffic_analyzer.application.frame_processor import FrameProcessor
        from traffic_analyzer.infrastructure.video_loop import VideoLoop
        from traffic_analyzer.application.analyzer import Analyzer
        from traffic_analyzer.visualization.ghost_track_manager import GhostTrackManager
        from traffic_analyzer.visualization.renderer import Renderer, GHOST_FRAMES
        from traffic_analyzer.domain.ports import CompositePublisher, NullPublisher

        cfg    = load_config(str(_CFG_PATH))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.camera.source_id = camera_id
        cfg.lanes = load_camera_lanes(camera_id, _CAMERAS_DIR)

        # Resolve model_path relative to project root so the app works
        # regardless of the working directory when launched.
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
            event_builder=EventBuilder(cfg), renderer=Renderer(),
            ghost_manager=GhostTrackManager(ghost_frames=GHOST_FRAMES), config=cfg)
        Analyzer(
            video_loop=VideoLoop(video_path=video_source,
                                 display_width=cfg.camera.display_width),
            frame_processor=processor, publisher=publisher).run()

    # ──────────────────────────────────────────────────────────────────────────
    def _set_status(self, msg: str, color: str = FG_DIM):
        self._status_lbl.configure(text=msg, text_color=color)

    def _center(self):
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w,  h  = self.winfo_width(),       self.winfo_height()
        self.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")


def main():
    LauncherApp().mainloop()

if __name__ == "__main__":
    main()
