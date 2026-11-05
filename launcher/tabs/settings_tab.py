"""
launcher/tabs/settings_tab.py

Dynamic metadata-driven settings form with dataclass validation, factory defaults reset,
and atomic persistence via config_loader.save_config().
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import customtkinter as ctk

from launcher.theme import (
    ACCENT,
    ACCENT_H,
    BG,
    BORDER,
    CARD,
    CARD_HI,
    F_BODY,
    F_HEAD,
    F_HELPER,
    F_MONO,
    F_SMALL,
    FG,
    FG_DIM,
    GREEN,
    RED,
    SURFACE,
    YELLOW,
)
from launcher.validation import discover_model_files, snap_resolution_to_block40
from traffic_analyzer.infrastructure.config_loader import (
    AnomalyConfig,
    AppConfig,
    CameraConfig,
    KinematicsConfig,
    ModelConfig,
    SceneThresholdsConfig,
    TrackingConfig,
    load_config,
    save_config,
)

RESOLUTION_OPTIONS = [
    "640",
    "800",
    "880 (Recommended)",
    "1040",
    "1280",
    "Custom...",
]

# Metadata schema: (section.key, label, description, type, min_val, max_val)
SETTINGS_METADATA: List[Tuple[str, str, str, type, float, float]] = [
    # Model & Camera Display
    ("model.batch_size", "Batch Size", "Number of frames processed per forward pass in RF-DETR inference (default: 1).", int, 1, 64),
    ("camera.display_width", "Live Display Window Width (px)", "Width of the OpenCV real-time preview window (480 - 3840 px).", int, 480, 3840),
    # Multi-Object Tracking
    ("tracking.lost_track_buffer", "Lost Track Buffer (frames)", "Frames to retain occluded/lost tracks before dropping their ID (ByteTrack lost_track_buffer).", int, 1, 5000),
    ("tracking.min_matching_threshold", "Matching IoU Threshold (0.05 - 0.99)", "Minimum IoU similarity required to associate detections across consecutive frames.", float, 0.05, 0.99),
    ("tracking.min_consecutive_frames", "Min Consecutive Frames", "Minimum consecutive detections required before confirming a new vehicle track.", int, 1, 100),
    ("tracking.ghost_frames", "Ghost Rendering Frames", "Frames to keep rendering a dashed bounding box after a tracked vehicle temporarily disappears.", int, 0, 1000),
    # Kinematics
    ("kinematics.ema_tau", "Speed EMA Time Constant (ema_tau, sec)", "Continuous-time exponential smoothing time constant tau in seconds (alpha = 1 - exp(-dt/tau)).", float, 0.05, 5.0),
    ("kinematics.position_history", "Position History Length (frames)", "Maximum trajectory centroids stored per vehicle for velocity and stop analysis.", int, 5, 2000),
    ("kinematics.direction_window", "Direction Regression Window (frames)", "Number of recent observations used in linear regression to determine travel direction.", int, 3, 500),
    ("kinematics.class_vote_window", "Class Majority Vote Window (frames)", "Sliding window size used to stabilize vehicle class labels against single-frame flicker.", int, 1, 500),
    ("kinematics.min_frames_before_event", "Min Frames Before Event Emission", "Minimum frames a vehicle must reside inside an ROI lane before publishing Kafka/Influx events.", int, 1, 500),
    # Anomaly Detection
    ("anomaly.stop_std_threshold", "Stopped Vehicle Spatial Std Threshold (px)", "Maximum positional standard deviation (in pixels) below which a vehicle is flagged as stopped.", float, 0.1, 100.0),
    ("anomaly.stop_min_frames", "Stopped Detection Window (stop_min_frames)", "Number of recent trajectory samples evaluated when testing if a vehicle has come to a stop.", int, 3, 2000),
    ("anomaly.stop_min_elapsed_sec", "Stopped Minimum Elapsed Time (sec)", "Minimum physical duration in seconds a vehicle must remain stationary before triggering a stop event.", float, 0.1, 60.0),
    ("anomaly.slowdown_ratio", "Sudden Slowdown Drop Ratio (0.05 - 0.95)", "Fractional drop in recent speed relative to prior speed that triggers a sudden slowdown anomaly.", float, 0.05, 0.95),
    ("anomaly.slowdown_min_speed", "Slowdown Minimum Prior Speed (px/s)", "Minimum baseline speed required before a vehicle can trigger a sudden slowdown alert.", float, 1.0, 1000.0),
    # Scene Thresholds
    ("scene_thresholds.speed_free", "Free-Flow Speed Threshold (px/s)", "Average scene speed at or above which traffic flow is scored as FREE.", float, 1.0, 500.0),
    ("scene_thresholds.speed_flow", "Normal Flow Speed Threshold (px/s)", "Average scene speed threshold separating normal FLOW from heavier traffic.", float, 1.0, 500.0),
    ("scene_thresholds.speed_heavy", "Heavy Traffic Speed Threshold (px/s)", "Average scene speed threshold below which traffic degrades toward HEAVY congestion.", float, 1.0, 500.0),
    ("scene_thresholds.speed_jammed", "Jammed Traffic Speed Threshold (px/s)", "Critical low-speed threshold indicating JAMMED stop-and-go conditions.", float, 0.5, 500.0),
    ("scene_thresholds.occ_free", "Free Occupancy Ratio (0.0 - 1.0)", "ROI area occupancy ratio below which lane density is considered uncongested.", float, 0.01, 0.99),
    ("scene_thresholds.occ_flow", "Normal Flow Occupancy Ratio (0.0 - 1.0)", "ROI area occupancy ratio threshold separating FREE and normal FLOW conditions.", float, 0.01, 0.99),
    ("scene_thresholds.occ_heavy", "Heavy Occupancy Ratio (0.0 - 1.0)", "ROI area occupancy ratio threshold above which traffic transitions to HEAVY.", float, 0.02, 0.99),
    ("scene_thresholds.occ_jammed", "Jammed Occupancy Ratio (0.0 - 1.0)", "ROI area occupancy ratio above which the roadway is classified as heavily saturated.", float, 0.05, 1.00),
    ("scene_thresholds.snapshot_every", "Scene Snapshot Interval (frames)", "Number of frames between periodic scene-level traffic_snapshot events sent to Kafka/InfluxDB.", int, 1, 5000),
]

SECTION_TITLES: Dict[str, str] = {
    "model": "Model & Inference Settings (model_settings)",
    "tracking": "Multi-Object Tracking Settings (tracking_settings)",
    "kinematics": "Kinematics & Anomaly Detection (kinematics_settings & anomaly_settings)",
    "scene_thresholds": "Scene Density & Flow Thresholds (scene_thresholds)",
}


class SettingsTab(ctk.CTkFrame):
    """Metadata-driven configuration tab backed by AppConfig dataclasses and save_config()."""

    def __init__(
        self,
        parent: tk.Misc,
        project_root: Path,
        config_path: Path,
        on_status_message: Callable[[str, str], None],
    ) -> None:
        super().__init__(parent, fg_color=BG, corner_radius=0)
        self._project_root = project_root
        self._config_path = config_path
        self._on_status_message = on_status_message
        self._field_vars: Dict[str, tk.StringVar] = {}
        self._msg_clear_after_id: Optional[str] = None

        self._cfg: AppConfig = load_config(str(self._config_path))
        self._build_ui()

    def _build_ui(self) -> None:
        scroll = ctk.CTkScrollableFrame(self, fg_color=BG, corner_radius=0)
        scroll.pack(fill="both", expand=True, padx=28, pady=(14, 8))

        section_cards: Dict[str, ctk.CTkFrame] = {}
        for sec_key, sec_title in SECTION_TITLES.items():
            card = ctk.CTkFrame(scroll, fg_color=CARD, corner_radius=10, border_width=1, border_color=BORDER)
            card.pack(fill="x", pady=(0, 10))
            ctk.CTkLabel(
                card,
                text=sec_title,
                font=F_HEAD,
                text_color=ACCENT,
                anchor="w",
            ).pack(fill="x", padx=18, pady=(12, 6))
            section_cards[sec_key] = card

        self._build_model_header_controls(section_cards["model"])

        for key, label, desc, _val_type, _min_v, _max_v in SETTINGS_METADATA:
            section_name, attr_name = key.split(".", 1)
            target_card_key = "model" if section_name in ("model", "camera") else (
                "kinematics" if section_name in ("kinematics", "anomaly") else section_name
            )
            parent_card = section_cards[target_card_key]
            section_obj = getattr(self._cfg, section_name)
            cur_val = getattr(section_obj, attr_name)
            var = tk.StringVar(value=str(cur_val))
            self._field_vars[key] = var
            self._add_param_row(parent_card, label, desc, var)

        bot_bar = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0, height=56)
        bot_bar.pack(fill="x", side="bottom")
        bot_bar.pack_propagate(False)

        self._save_msg_lbl = ctk.CTkLabel(bot_bar, text="", font=F_SMALL, text_color=GREEN)
        self._save_msg_lbl.pack(side="left", padx=28)

        btn_box = ctk.CTkFrame(bot_bar, fg_color="transparent")
        btn_box.pack(side="right", padx=28, pady=10)

        ctk.CTkButton(
            btn_box,
            text="↺ Reset to Defaults",
            width=154,
            height=36,
            corner_radius=8,
            fg_color=CARD,
            hover_color=CARD_HI,
            border_width=1,
            border_color=BORDER,
            text_color=FG,
            font=F_HEAD,
            command=self.reset_to_defaults,
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_box,
            text="Save Settings",
            width=154,
            height=36,
            corner_radius=8,
            fg_color=ACCENT,
            hover_color=ACCENT_H,
            text_color="#ffffff",
            font=F_HEAD,
            command=lambda: self.save_settings_to_disk(quiet=False),
        ).pack(side="left")

    def _build_model_header_controls(self, m_card: ctk.CTkFrame) -> None:
        ms = self._cfg.model
        top_grid = ctk.CTkFrame(m_card, fg_color="transparent")
        top_grid.pack(fill="x", padx=18, pady=4)

        col_m = ctk.CTkFrame(top_grid, fg_color="transparent")
        col_m.pack(side="left", fill="x", expand=True, padx=(0, 14))
        ctk.CTkLabel(col_m, text="Model Checkpoint (.pt / .pth)", font=F_BODY, text_color=FG, anchor="w").pack(fill="x")
        ctk.CTkLabel(
            col_m,
            text="Trained RF-DETR weights file used for 6-class vehicle detection.",
            font=F_HELPER,
            text_color=FG_DIM,
            anchor="w",
        ).pack(fill="x", pady=(0, 4))

        models = discover_model_files(self._project_root)
        if ms.model_path not in models:
            models.insert(0, ms.model_path)
        self._model_var = tk.StringVar(value=ms.model_path)
        ctk.CTkOptionMenu(
            col_m,
            variable=self._model_var,
            values=models,
            height=30,
            corner_radius=6,
            fg_color=SURFACE,
            button_color=ACCENT,
            button_hover_color=ACCENT_H,
            text_color=FG,
            font=F_BODY,
        ).pack(fill="x")

        col_res = ctk.CTkFrame(top_grid, fg_color="transparent")
        col_res.pack(side="left", fill="x", expand=True, padx=(0, 14))
        ctk.CTkLabel(col_res, text="Inference Resolution", font=F_BODY, text_color=FG, anchor="w").pack(fill="x")
        ctk.CTkLabel(
            col_res,
            text="Input tensor size (multiple of 40); higher improves small-object recall.",
            font=F_HELPER,
            text_color=FG_DIM,
            anchor="w",
        ).pack(fill="x", pady=(0, 4))

        preset_map = {640: "640", 800: "800", 880: "880 (Recommended)", 1040: "1040", 1280: "1280"}
        init_res_choice = preset_map.get(ms.resolution, "Custom...")
        self._res_choice_var = tk.StringVar(value=init_res_choice)
        self._custom_res_var = tk.StringVar(value=str(ms.resolution))

        ctk.CTkOptionMenu(
            col_res,
            variable=self._res_choice_var,
            values=RESOLUTION_OPTIONS,
            height=30,
            corner_radius=6,
            fg_color=SURFACE,
            button_color=ACCENT,
            button_hover_color=ACCENT_H,
            text_color=FG,
            font=F_BODY,
            command=self._on_resolution_dropdown_changed,
        ).pack(fill="x")

        col_dev = ctk.CTkFrame(top_grid, fg_color="transparent")
        col_dev.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(col_dev, text="Compute Device", font=F_BODY, text_color=FG, anchor="w").pack(fill="x")
        ctk.CTkLabel(
            col_dev,
            text="Execution hardware target ('cuda' enables FP16 AMP acceleration).",
            font=F_HELPER,
            text_color=FG_DIM,
            anchor="w",
        ).pack(fill="x", pady=(0, 4))

        self._device_var = tk.StringVar(value=ms.device)
        ctk.CTkOptionMenu(
            col_dev,
            variable=self._device_var,
            values=["cuda", "cpu", "auto"],
            height=30,
            corner_radius=6,
            fg_color=SURFACE,
            button_color=ACCENT,
            button_hover_color=ACCENT_H,
            text_color=FG,
            font=F_BODY,
        ).pack(fill="x")

        self._custom_res_frame = ctk.CTkFrame(m_card, fg_color="transparent")
        c_row = ctk.CTkFrame(self._custom_res_frame, fg_color="transparent")
        c_row.pack(fill="x", padx=18, pady=(4, 2))
        ctk.CTkLabel(c_row, text="Custom Resolution (px):", font=F_BODY, text_color=FG).pack(side="left", padx=(0, 10))
        ctk.CTkEntry(
            c_row,
            textvariable=self._custom_res_var,
            width=100,
            height=28,
            corner_radius=6,
            fg_color=SURFACE,
            border_color=BORDER,
            text_color=FG,
            font=F_MONO,
        ).pack(side="left")
        ctk.CTkLabel(
            self._custom_res_frame,
            text="Resolution must be a multiple of 40 (e.g. 720, 960).",
            font=F_HELPER,
            text_color=FG_DIM,
            anchor="w",
        ).pack(fill="x", padx=18, pady=(0, 6))
        if init_res_choice == "Custom...":
            self._custom_res_frame.pack(fill="x")

        th_top = ctk.CTkFrame(m_card, fg_color="transparent")
        th_top.pack(fill="x", padx=18, pady=(8, 0))
        ctk.CTkLabel(th_top, text="Confidence Threshold", font=F_BODY, text_color=FG).pack(side="left")
        self._thr_val_lbl = ctk.CTkLabel(th_top, text=f"{ms.threshold:.2f}", font=F_MONO, text_color=ACCENT)
        self._thr_val_lbl.pack(side="right")

        ctk.CTkLabel(
            m_card,
            text="Minimum detection probability (0.1 - 0.9). Lower values detect more distant vehicles.",
            font=F_HELPER,
            text_color=FG_DIM,
            anchor="w",
        ).pack(fill="x", padx=18, pady=(0, 4))

        self._thr_slider = ctk.CTkSlider(
            m_card,
            from_=0.10,
            to=0.90,
            number_of_steps=80,
            fg_color=SURFACE,
            progress_color=ACCENT,
            button_color=FG,
            button_hover_color=ACCENT,
            command=lambda v: self._thr_val_lbl.configure(text=f"{float(v):.2f}"),
        )
        self._thr_slider.set(max(0.10, min(0.90, ms.threshold)))
        self._thr_slider.pack(fill="x", padx=18, pady=(0, 8))

    @staticmethod
    def _add_param_row(
        parent: tk.Misc,
        title: str,
        helper_text: str,
        var: tk.StringVar,
        width: int = 100,
    ) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=5)

        left = ctk.CTkFrame(row, fg_color="transparent")
        left.pack(side="left", fill="x", expand=True, padx=(0, 12))
        ctk.CTkLabel(left, text=title, font=F_BODY, text_color=FG, anchor="w").pack(fill="x")
        ctk.CTkLabel(
            left,
            text=helper_text,
            font=F_HELPER,
            text_color=FG_DIM,
            anchor="w",
        ).pack(fill="x")

        ctk.CTkEntry(
            row,
            textvariable=var,
            width=width,
            height=30,
            corner_radius=6,
            fg_color=SURFACE,
            border_color=BORDER,
            text_color=FG,
            font=F_MONO,
        ).pack(side="right")

    def _on_resolution_dropdown_changed(self, choice: str) -> None:
        if choice == "Custom...":
            self._custom_res_frame.pack(fill="x")
        else:
            self._custom_res_frame.pack_forget()
            self._custom_res_var.set(choice.split()[0])

    def _clear_save_notification(self) -> None:
        self._msg_clear_after_id = None
        if self._save_msg_lbl.winfo_exists():
            self._save_msg_lbl.configure(text="")

    def _show_transient_notification(self, text: str, color: str = GREEN, duration_ms: int = 2000) -> None:
        if self._msg_clear_after_id is not None:
            try:
                self.after_cancel(self._msg_clear_after_id)
            except Exception:
                pass
            self._msg_clear_after_id = None
        self._save_msg_lbl.configure(text=text, text_color=color)
        if duration_ms > 0:
            self._msg_clear_after_id = self.after(duration_ms, self._clear_save_notification)

    def reset_to_defaults(self) -> None:
        """Reset all UI form inputs and sliders to default AppConfig values."""
        default_cfg = AppConfig(camera=CameraConfig(), model=ModelConfig())
        ms = default_cfg.model

        self._model_var.set(ms.model_path)
        self._device_var.set(ms.device)
        self._thr_slider.set(max(0.10, min(0.90, ms.threshold)))
        self._thr_val_lbl.configure(text=f"{ms.threshold:.2f}")

        preset_map = {640: "640", 800: "800", 880: "880 (Recommended)", 1040: "1040", 1280: "1280"}
        init_res_choice = preset_map.get(ms.resolution, "Custom...")
        self._res_choice_var.set(init_res_choice)
        self._custom_res_var.set(str(ms.resolution))
        if init_res_choice == "Custom...":
            self._custom_res_frame.pack(fill="x")
        else:
            self._custom_res_frame.pack_forget()

        for key, _label, _desc, _val_type, _min_v, _max_v in SETTINGS_METADATA:
            sec_name, attr_name = key.split(".", 1)
            sec_obj = getattr(default_cfg, sec_name)
            val = getattr(sec_obj, attr_name)
            if key in self._field_vars:
                self._field_vars[key].set(str(val))

        msg = "Settings reset to factory defaults. Click Save to persist."
        self._show_transient_notification(msg, color=YELLOW, duration_ms=4000)
        self._on_status_message(msg, YELLOW)

    def save_settings_to_disk(self, quiet: bool = False) -> bool:
        try:
            cfg = load_config(str(self._config_path))
            choice = self._res_choice_var.get()
            raw_res = self._custom_res_var.get() if choice == "Custom..." else choice.split()[0]
            snapped_res, _was_snapped = snap_resolution_to_block40(raw_res)
            self._custom_res_var.set(str(snapped_res))

            parsed: Dict[str, Dict[str, Any]] = {
                "camera": {
                    "source_id": cfg.camera.source_id,
                    "video_path": cfg.camera.video_path,
                    "display_width": cfg.camera.display_width,
                },
                "model": {
                    "model_path": self._model_var.get().strip() or "best.pt",
                    "threshold": round(float(self._thr_slider.get()), 2),
                    "resolution": snapped_res,
                    "batch_size": cfg.model.batch_size,
                    "device": self._device_var.get().strip() or "cuda",
                },
                "tracking": {},
                "kinematics": {},
                "anomaly": {},
                "scene_thresholds": {},
            }

            for key, _label, _desc, val_type, min_v, max_v in SETTINGS_METADATA:
                sec_name, attr_name = key.split(".", 1)
                raw_str = self._field_vars[key].get().strip()
                num = float(raw_str)
                clamped = max(min_v, min(max_v, num))
                final_val = int(round(clamped)) if val_type is int else round(clamped, 4)
                parsed[sec_name][attr_name] = final_val
                self._field_vars[key].set(str(final_val))

            cfg.camera = CameraConfig(**parsed["camera"])
            cfg.model = ModelConfig(**parsed["model"])
            cfg.tracking = TrackingConfig(**parsed["tracking"])
            cfg.kinematics = KinematicsConfig(**parsed["kinematics"])
            cfg.anomaly = AnomalyConfig(**parsed["anomaly"])
            cfg.scene_thresholds = SceneThresholdsConfig(**parsed["scene_thresholds"])

            save_config(cfg, self._config_path)
            # Re-read from disk to guarantee in-memory state matches persisted config.json
            self._cfg = load_config(str(self._config_path))

            if not quiet:
                msg = "✓ Settings successfully saved to config.json"
                self._show_transient_notification(msg, color=GREEN, duration_ms=2000)
                self._on_status_message(msg, GREEN)
            return True
        except Exception as exc:
            if not quiet:
                self._show_transient_notification(f"Validation error: {exc}", color=RED, duration_ms=4000)
                self._on_status_message(f"Settings error: {exc}", RED)
            return False
