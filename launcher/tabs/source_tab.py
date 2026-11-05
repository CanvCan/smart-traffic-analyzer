"""
launcher/tabs/source_tab.py

Camera cache, local video file selector, and ROI/Analysis triggers.
"""

from __future__ import annotations

import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog
from typing import Any, Callable, Dict, List, Optional, Tuple

import customtkinter as ctk
import cv2

from launcher.ipc_runner import AnalysisProcessManager
from launcher.theme import (
    ACCENT,
    ACCENT_H,
    BG,
    BORDER,
    CARD,
    CARD_HI,
    F_BODY,
    F_HEAD,
    F_MONO,
    F_SMALL,
    FG,
    FG_DIM,
    FG_MED,
    FONT_FAMILY,
    GREEN,
    GREEN_D,
    RED,
    SEL_BG,
    SEL_BOR,
    SURFACE,
    YELLOW,
    YELLOW_D,
)
from traffic_analyzer.domain.text import slugify_ascii
from traffic_analyzer.infrastructure.config_loader import (
    load_camera_lanes,
    load_config,
    resolve_local_video_source_id,
    save_config,
)
from traffic_analyzer.visualization.roi_selector import run as roi_run


@dataclass
class SourceItem:
    kind: str  # "live" or "file"
    source_id: str
    display_name: str
    path_or_url: str
    meta_text: str


class SelectableRow(ctk.CTkFrame):
    """Selectable source card row."""

    H = 42

    def __init__(
        self,
        master: tk.Misc,
        item: SourceItem,
        has_roi: bool,
        on_select: Callable[["SelectableRow"], None],
    ) -> None:
        super().__init__(
            master,
            height=self.H,
            corner_radius=8,
            fg_color=CARD,
            border_color=BORDER,
            border_width=1,
            cursor="hand2",
        )
        self.pack_propagate(False)
        self.item = item
        self.has_roi = has_roi
        self._on_select = on_select
        self._selected = False

        self._dot = ctk.CTkLabel(
            self,
            text="●",
            width=18,
            font=(FONT_FAMILY, 11),
            text_color=GREEN if has_roi else FG_DIM,
        )
        self._dot.pack(side="left", padx=(12, 4))

        self._name_lbl = ctk.CTkLabel(
            self,
            text=item.display_name,
            font=F_BODY,
            text_color=FG,
            anchor="w",
        )
        self._name_lbl.pack(side="left", fill="x", expand=True)

        self._roi_tag = ctk.CTkLabel(
            self,
            text="✓ ROI Configured" if has_roi else "○ No ROI",
            font=F_SMALL,
            text_color=GREEN if has_roi else FG_MED,
            width=115,
            anchor="e",
        )
        self._roi_tag.pack(side="right", padx=(6, 12))

        self._meta_lbl = ctk.CTkLabel(
            self,
            text=item.meta_text,
            font=F_MONO,
            text_color=FG_DIM,
            anchor="e",
        )
        self._meta_lbl.pack(side="right", padx=(6, 8))

        for w in (self, self._dot, self._name_lbl, self._roi_tag, self._meta_lbl):
            w.bind("<Button-1>", self._handle_click)

    def _handle_click(self, _event: Any = None) -> None:
        self._on_select(self)

    def set_selected(self, selected: bool) -> None:
        if self._selected == selected:
            return
        self._selected = selected
        if selected:
            self.configure(fg_color=SEL_BG, border_color=SEL_BOR, border_width=2)
            self._name_lbl.configure(text_color="#ffffff")
        else:
            self.configure(fg_color=CARD, border_color=BORDER, border_width=1)
            self._name_lbl.configure(text_color=FG)

    def update_roi_state(self, has_roi: bool) -> None:
        self.has_roi = has_roi
        self._dot.configure(text_color=GREEN if has_roi else FG_DIM)
        self._roi_tag.configure(
            text="✓ ROI Configured" if has_roi else "○ No ROI",
            text_color=GREEN if has_roi else FG_MED,
        )


class SourceTab(ctk.CTkFrame):
    """Tab managing live IZUM camera feeds, local video files, ROI editing, and analysis start/stop."""

    def __init__(
        self,
        parent: tk.Misc,
        project_root: Path,
        config_path: Path,
        cameras_dir: Path,
        proc_mgr: AnalysisProcessManager,
        on_before_start: Callable[[], None],
        on_status_message: Callable[[str, str], None],
    ) -> None:
        super().__init__(parent, fg_color=BG, corner_radius=0)
        self._project_root = project_root
        self._config_path = config_path
        self._cameras_dir = cameras_dir
        self._proc_mgr = proc_mgr
        self._on_before_start = on_before_start
        self._on_status_message = on_status_message

        self._cameras_cache: List[SourceItem] = []
        self._live_rows: List[SelectableRow] = []
        self._file_rows: List[SelectableRow] = []
        self._selected_row: Optional[SelectableRow] = None
        self._source_mode = tk.StringVar(value="live")
        self._search_var = tk.StringVar(value="")
        self._search_after_id: Optional[str] = None
        self._video_res_cache: Dict[str, str] = {}

        self._build_ui()
        self._populate_local_video_rows()
        self._fetch_live_cameras_once_async()

    def _build_ui(self) -> None:
        top_row = ctk.CTkFrame(self, fg_color="transparent")
        top_row.pack(fill="x", padx=28, pady=(16, 10))

        ctk.CTkLabel(
            top_row,
            text="Select Video Source:",
            font=F_HEAD,
            text_color=FG,
        ).pack(side="left", padx=(0, 14))

        for label, val in [("Live Cameras (IZUM)", "live"), ("Local Video Files", "file")]:
            ctk.CTkRadioButton(
                top_row,
                text=label,
                variable=self._source_mode,
                value=val,
                command=self._on_source_mode_changed,
                font=F_BODY,
                text_color=FG,
                fg_color=ACCENT,
                hover_color=ACCENT_H,
            ).pack(side="left", padx=(0, 22))

        self._source_stack = ctk.CTkFrame(self, fg_color="transparent")
        self._source_stack.pack(fill="both", expand=True, padx=28, pady=(0, 10))

        self._live_panel = ctk.CTkFrame(self._source_stack, fg_color="transparent")

        search_row = ctk.CTkFrame(self._live_panel, fg_color="transparent")
        search_row.pack(fill="x", pady=(0, 8))

        self._search_entry = ctk.CTkEntry(
            search_row,
            textvariable=self._search_var,
            placeholder_text="Search cameras by name or ID...",
            font=F_BODY,
            height=34,
            corner_radius=8,
            fg_color=SURFACE,
            border_color=BORDER,
            text_color=FG,
        )
        self._search_entry.pack(side="left", fill="x", expand=True)
        self._search_var.trace_add("write", self._on_search_changed)

        self._cam_count_lbl = ctk.CTkLabel(
            search_row,
            text="Loading...",
            font=F_SMALL,
            text_color=FG_MED,
            width=150,
            anchor="e",
        )
        self._cam_count_lbl.pack(side="right", padx=(10, 0))

        self._live_scroll = ctk.CTkScrollableFrame(
            self._live_panel,
            fg_color=SURFACE,
            corner_radius=8,
            border_width=1,
            border_color=BORDER,
        )
        self._live_scroll.pack(fill="both", expand=True)

        self._live_loading_lbl = ctk.CTkLabel(
            self._live_scroll,
            text="Fetching live IZUM cameras in background...",
            font=F_BODY,
            text_color=FG_MED,
        )
        self._live_loading_lbl.pack(pady=36)

        self._file_panel = ctk.CTkFrame(self._source_stack, fg_color="transparent")

        file_hdr = ctk.CTkFrame(self._file_panel, fg_color="transparent")
        file_hdr.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            file_hdr,
            text="Built-in Test Videos & Local Files (isolated per-video ROI in cameras/<source_id>.json)",
            font=F_SMALL,
            text_color=FG_MED,
            anchor="w",
        ).pack(side="left")

        ctk.CTkButton(
            file_hdr,
            text="Browse Video File...",
            width=148,
            height=30,
            corner_radius=6,
            fg_color=SURFACE,
            hover_color=CARD_HI,
            border_width=1,
            border_color=BORDER,
            text_color=FG,
            font=F_SMALL,
            command=self._browse_custom_video,
        ).pack(side="right")

        self._file_scroll = ctk.CTkScrollableFrame(
            self._file_panel,
            fg_color=SURFACE,
            corner_radius=8,
            border_width=1,
            border_color=BORDER,
        )
        self._file_scroll.pack(fill="both", expand=True)

        self._selection_card = ctk.CTkFrame(
            self,
            fg_color=SURFACE,
            corner_radius=10,
            border_width=1,
            border_color=BORDER,
            height=74,
        )
        self._selection_card.pack(fill="x", side="bottom", padx=28, pady=(0, 18))
        self._selection_card.pack_propagate(False)

        sel_left = ctk.CTkFrame(self._selection_card, fg_color="transparent")
        sel_left.pack(side="left", fill="both", expand=True, padx=16, pady=10)

        title_line = ctk.CTkFrame(sel_left, fg_color="transparent")
        title_line.pack(fill="x")

        self._sel_title_lbl = ctk.CTkLabel(
            title_line,
            text="No source selected",
            font=F_HEAD,
            text_color=FG,
            anchor="w",
        )
        self._sel_title_lbl.pack(side="left")

        self._sel_roi_badge = ctk.CTkLabel(
            title_line,
            text="○ No ROI",
            font=F_SMALL,
            text_color=FG_MED,
            fg_color=CARD,
            corner_radius=6,
            padx=10,
            pady=2,
        )
        self._sel_roi_badge.pack(side="left", padx=(12, 0))

        self._sel_meta_lbl = ctk.CTkLabel(
            sel_left,
            text="Select a live camera or local video file above to configure ROI or start analysis.",
            font=F_SMALL,
            text_color=FG_MED,
            anchor="w",
        )
        self._sel_meta_lbl.pack(fill="x", pady=(3, 0))

        sel_right = ctk.CTkFrame(self._selection_card, fg_color="transparent")
        sel_right.pack(side="right", padx=16)

        self._edit_roi_btn = ctk.CTkButton(
            sel_right,
            text="Edit ROI",
            width=126,
            height=38,
            corner_radius=8,
            fg_color=CARD,
            hover_color=CARD_HI,
            border_width=1,
            border_color=BORDER,
            text_color=FG,
            font=F_BODY,
            command=self._handle_edit_roi,
        )
        self._edit_roi_btn.pack(side="left", padx=(0, 10))

        self._start_analysis_btn = ctk.CTkButton(
            sel_right,
            text="▶  Start Analysis",
            width=156,
            height=38,
            corner_radius=8,
            fg_color=ACCENT,
            hover_color=ACCENT_H,
            text_color="#ffffff",
            font=F_HEAD,
            command=self._handle_start_or_stop_analysis,
        )
        self._start_analysis_btn.pack(side="left")

        self._live_panel.pack(fill="both", expand=True)

    def _on_source_mode_changed(self) -> None:
        if self._source_mode.get() == "live":
            self._file_panel.pack_forget()
            self._live_panel.pack(fill="both", expand=True)
            if self._live_rows and (self._selected_row is None or self._selected_row.item.kind != "live"):
                self._select_row(self._live_rows[0])
        else:
            self._live_panel.pack_forget()
            self._file_panel.pack(fill="both", expand=True)
            if self._file_rows and (self._selected_row is None or self._selected_row.item.kind != "file"):
                self._select_row(self._file_rows[0])

    def _has_roi_for_item(self, item: SourceItem) -> Tuple[bool, int]:
        lanes = load_camera_lanes(item.source_id, self._cameras_dir)
        return (len(lanes) > 0), len(lanes)

    def _probe_video_resolution(self, path_str: str) -> str:
        if path_str in self._video_res_cache:
            return self._video_res_cache[path_str]
        res_txt = "Video File"
        try:
            p = Path(path_str)
            if p.exists():
                cap = cv2.VideoCapture(str(p))
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    if w > 0 and h > 0:
                        res_txt = f"{w}×{h} @ {fps:.0f}fps"
                cap.release()
        except Exception:
            pass
        self._video_res_cache[path_str] = res_txt
        return res_txt

    def _populate_local_video_rows(self) -> None:
        video_dir = self._project_root / "traffic_analyzer"
        candidates: List[Path] = []
        seen = set()

        for name in ("test_video.mp4", "test_video1.mp4", "test_video2.mp4", "test_video3.mp4", "test_video4.mp4"):
            p = (video_dir / name).resolve()
            if p.exists() and str(p) not in seen:
                seen.add(str(p))
                candidates.append(p)

        for p in sorted(video_dir.glob("*.mp4")):
            rp = p.resolve()
            if str(rp) not in seen:
                seen.add(str(rp))
                candidates.append(rp)

        for p in candidates:
            self._add_file_row(p)

    def _add_file_row(self, video_path: Path, select: bool = False) -> SelectableRow:
        for existing in self._file_rows:
            if Path(existing.item.path_or_url).resolve() == video_path.resolve():
                if select:
                    self._select_row(existing)
                return existing

        source_id = resolve_local_video_source_id(video_path, self._cameras_dir)
        res_txt = self._probe_video_resolution(str(video_path))
        item = SourceItem(
            kind="file",
            source_id=source_id,
            display_name=video_path.name,
            path_or_url=str(video_path),
            meta_text=res_txt,
        )
        has_roi, _ = self._has_roi_for_item(item)
        row = SelectableRow(self._file_scroll, item, has_roi, self._select_row)
        row.pack(fill="x", padx=4, pady=3)
        self._file_rows.append(row)
        if select:
            self._select_row(row)
        return row

    def _browse_custom_video(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Select Video File",
            initialdir=str(self._project_root / "traffic_analyzer"),
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")],
        )
        if chosen:
            self._add_file_row(Path(chosen), select=True)

    def _fetch_live_cameras_once_async(self) -> None:
        def _worker() -> None:
            items: List[SourceItem] = []
            unreachable = False
            try:
                from traffic_analyzer.adapters.camera.camera_fetcher import fetch_cameras
                cams = [c for c in fetch_cameras() if c.stream_url]
                for c in cams:
                    items.append(
                        SourceItem(
                            kind="live",
                            source_id=slugify_ascii(c.name),
                            display_name=c.name,
                            path_or_url=c.stream_url,
                            meta_text=f"ID: {c.camera_id} • Live MJPEG",
                        )
                    )
            except Exception:
                unreachable = True

            self.after(0, lambda: self._build_cached_live_rows(items, unreachable=unreachable))

        threading.Thread(target=_worker, daemon=True, name="CameraCacheLoader").start()

    def _build_cached_live_rows(self, items: List[SourceItem], unreachable: bool = False) -> None:
        self._cameras_cache = list(items)
        if not items:
            if self._live_loading_lbl.winfo_exists():
                self._live_loading_lbl.configure(
                    text="IZUM API is unreachable",
                    text_color=RED if unreachable else FG_MED,
                )
            self._cam_count_lbl.configure(text="IZUM API is unreachable", text_color=RED)
            return

        if self._live_loading_lbl.winfo_exists():
            self._live_loading_lbl.destroy()

        for item in items:
            has_roi, _ = self._has_roi_for_item(item)
            row = SelectableRow(self._live_scroll, item, has_roi, self._select_row)
            row.pack(fill="x", padx=4, pady=2)
            self._live_rows.append(row)

        self._cam_count_lbl.configure(text=f"{len(self._live_rows)} cameras", text_color=FG_MED)
        if self._live_rows and self._selected_row is None:
            self._select_row(self._live_rows[0])

    def _on_search_changed(self, *_args: Any) -> None:
        if self._search_after_id:
            self.after_cancel(self._search_after_id)
        self._search_after_id = self.after(140, self._filter_live_rows)

    def _filter_live_rows(self) -> None:
        if not self._live_rows:
            return
        q = self._search_var.get().strip().lower()
        visible = 0
        for row in self._live_rows:
            haystack = f"{row.item.display_name} {row.item.source_id} {row.item.meta_text}".lower()
            if not q or q in haystack:
                row.pack(fill="x", padx=4, pady=2)
                visible += 1
            else:
                row.pack_forget()
        total = len(self._live_rows)
        self._cam_count_lbl.configure(text=f"{visible}/{total} cameras" if q else f"{total} cameras")

    def _select_row(self, row: SelectableRow) -> None:
        if self._selected_row is not None and self._selected_row is not row:
            self._selected_row.set_selected(False)
        row.set_selected(True)
        self._selected_row = row

        has_roi, lane_count = self._has_roi_for_item(row.item)
        row.update_roi_state(has_roi)

        self._sel_title_lbl.configure(text=row.item.display_name)
        if has_roi:
            self._sel_roi_badge.configure(
                text=f"✓ ROI Configured ({lane_count} lanes)",
                text_color=GREEN,
                fg_color=GREEN_D,
            )
        else:
            self._sel_roi_badge.configure(
                text="○ No ROI",
                text_color=YELLOW,
                fg_color=YELLOW_D,
            )

        self._sel_meta_lbl.configure(
            text=f"{row.item.meta_text}   •   Config ID: {row.item.source_id}"
        )

    def _handle_edit_roi(self) -> None:
        if self._selected_row is None:
            return
        item = self._selected_row.item
        effective_source_id = (
            resolve_local_video_source_id(item.path_or_url, self._cameras_dir)
            if item.kind == "file"
            else item.source_id
        )
        item.source_id = effective_source_id
        self._on_status_message(f"Opening ROI Editor: {item.display_name} [{effective_source_id}]", YELLOW)
        try:
            if item.kind == "live":
                roi_run(
                    config_path=str(self._config_path),
                    camera_id=effective_source_id,
                    stream_url=item.path_or_url,
                    cameras_dir=self._cameras_dir,
                    master=self,
                )
            else:
                roi_run(
                    config_path=str(self._config_path),
                    camera_id=effective_source_id,
                    cameras_dir=self._cameras_dir,
                    video_path=item.path_or_url,
                    master=self,
                )
        finally:
            self._select_row(self._selected_row)
            for r in self._file_rows:
                has_r, _ = self._has_roi_for_item(r.item)
                r.update_roi_state(has_r)
            self._on_status_message("ROI updated", GREEN)

    def _handle_start_or_stop_analysis(self) -> None:
        if self._proc_mgr.is_running:
            self._proc_mgr.request_stop()
            return

        if self._selected_row is None:
            return

        self._on_before_start()
        item = self._selected_row.item
        effective_source_id = (
            resolve_local_video_source_id(item.path_or_url, self._cameras_dir)
            if item.kind == "file"
            else item.source_id
        )
        item.source_id = effective_source_id
        has_roi, _ = self._has_roi_for_item(item)

        cfg = load_config(str(self._config_path))
        cfg.camera.source_id = effective_source_id
        cfg.camera.video_path = item.path_or_url
        save_config(cfg, self._config_path)

        started = self._proc_mgr.start(
            config_path=self._config_path,
            cameras_dir=self._cameras_dir,
            visual_only=not has_roi,
            source_id=effective_source_id,
            video_path=item.path_or_url,
        )
        if started:
            self._start_analysis_btn.configure(
                text="■  Stop Analysis",
                fg_color=RED,
                hover_color="#c44160",
            )
            self._on_status_message(f"Running: {item.display_name}", GREEN)

    def apply_worker_state(self, state: str, detail: str) -> None:
        if state in ("STARTING", "RUNNING"):
            self._start_analysis_btn.configure(
                text="■  Stop Analysis",
                fg_color=RED,
                hover_color="#c44160",
            )
            self._on_status_message(detail or "Analysis running", GREEN)
        elif state in ("STOPPED", "ERROR"):
            self._start_analysis_btn.configure(
                text="▶  Start Analysis",
                fg_color=ACCENT,
                hover_color=ACCENT_H,
            )
            color = RED if state == "ERROR" else FG_MED
            self._on_status_message(detail or "Ready", color)
