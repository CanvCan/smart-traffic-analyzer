"""
launcher/tabs/services_tab.py

Displays Docker container and port health statuses using ServiceManager.
"""

from __future__ import annotations

import tkinter as tk
import webbrowser
from typing import Callable, Dict, List

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
    F_MONO,
    F_SMALL,
    FG,
    FG_MED,
    FONT_FAMILY,
    GREEN,
    GREEN_D,
    RED,
    RED_D,
    SURFACE,
    YELLOW,
    YELLOW_D,
)
from traffic_analyzer.infrastructure.service_manager import ServiceManager


class ServicesTab(ctk.CTkFrame):
    """Tab showing Docker, Kafka, Spark, InfluxDB, and Grafana health."""

    def __init__(
        self,
        parent: tk.Misc,
        service_mgr: ServiceManager,
        on_status_message: Callable[[str, str], None],
    ) -> None:
        super().__init__(parent, fg_color=BG, corner_radius=0)
        self._service_mgr = service_mgr
        self._on_status_message = on_status_message
        self._service_check_busy = False
        self._service_badges: Dict[str, ctk.CTkLabel] = {}
        self._service_detail_lbls: Dict[str, ctk.CTkLabel] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=28, pady=18)

        ctk.CTkLabel(
            container,
            text="Infrastructure Services & Control Plane",
            font=(FONT_FAMILY, 15, "bold"),
            text_color=FG,
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            container,
            text="Monitor and start Docker containers for Kafka event streaming, Spark Structured Streaming, InfluxDB, and Grafana.",
            font=F_SMALL,
            text_color=FG_MED,
            anchor="w",
        ).pack(fill="x", pady=(2, 12))

        service_defs = [
            ("Docker", "Docker Daemon", "Container Engine & Runtime"),
            ("Kafka", "Kafka Broker", f"Event Stream (localhost:{self._service_mgr.kafka_port})"),
            ("Spark", "Spark Streaming", "Windowed Analytics Q1–Q11 (spark_processor)"),
            ("InfluxDB", "InfluxDB", f"Time-Series Storage ({self._service_mgr.influxdb_url})"),
            ("Grafana", "Grafana", f"Visualization Dashboards ({self._service_mgr.grafana_url})"),
        ]

        cards_frame = ctk.CTkFrame(container, fg_color="transparent")
        cards_frame.pack(fill="x")

        for key, title, subtitle in service_defs:
            card = ctk.CTkFrame(
                cards_frame,
                fg_color=CARD,
                corner_radius=8,
                border_width=1,
                border_color=BORDER,
                height=50,
            )
            card.pack(fill="x", pady=3)
            card.pack_propagate(False)

            left = ctk.CTkFrame(card, fg_color="transparent")
            left.pack(side="left", fill="both", expand=True, padx=16, pady=5)

            ctk.CTkLabel(left, text=title, font=F_HEAD, text_color=FG, anchor="w").pack(fill="x")
            detail_lbl = ctk.CTkLabel(left, text=subtitle, font=F_SMALL, text_color=FG_MED, anchor="w")
            detail_lbl.pack(fill="x")
            self._service_detail_lbls[key] = detail_lbl

            badge = ctk.CTkLabel(
                card,
                text="● Checking...",
                font=F_HEAD,
                text_color=YELLOW,
                fg_color=YELLOW_D,
                corner_radius=6,
                width=116,
                height=28,
            )
            badge.pack(side="right", padx=16)
            self._service_badges[key] = badge

        action_row = ctk.CTkFrame(container, fg_color="transparent")
        action_row.pack(fill="x", pady=(12, 8))

        self._start_services_btn = ctk.CTkButton(
            action_row,
            text="▶  Start Services",
            width=176,
            height=38,
            corner_radius=8,
            fg_color=ACCENT,
            hover_color=ACCENT_H,
            text_color="#ffffff",
            font=F_HEAD,
            command=self._handle_start_services,
        )
        self._start_services_btn.pack(side="left")

        ctk.CTkButton(
            action_row,
            text="↻  Refresh Status",
            width=132,
            height=38,
            corner_radius=8,
            fg_color=SURFACE,
            hover_color=CARD_HI,
            border_width=1,
            border_color=BORDER,
            text_color=FG,
            font=F_BODY,
            command=self.trigger_check_now,
        ).pack(side="left", padx=(10, 0))

        ctk.CTkButton(
            action_row,
            text="Open InfluxDB",
            width=130,
            height=38,
            corner_radius=8,
            fg_color=CARD,
            hover_color=CARD_HI,
            border_width=1,
            border_color=BORDER,
            text_color=FG,
            font=F_SMALL,
            command=lambda: webbrowser.open(self._service_mgr.influxdb_url),
        ).pack(side="right")

        ctk.CTkButton(
            action_row,
            text="Open Grafana",
            width=130,
            height=38,
            corner_radius=8,
            fg_color=CARD,
            hover_color=CARD_HI,
            border_width=1,
            border_color=BORDER,
            text_color=FG,
            font=F_SMALL,
            command=lambda: webbrowser.open(self._service_mgr.grafana_url),
        ).pack(side="right", padx=(0, 10))

        self._diag_box = ctk.CTkFrame(
            container,
            fg_color=SURFACE,
            corner_radius=8,
            border_width=1,
            border_color=BORDER,
        )
        self._diag_box.pack(fill="both", expand=True, pady=(4, 0))

        diag_hdr = ctk.CTkFrame(self._diag_box, fg_color="transparent")
        diag_hdr.pack(fill="x", padx=14, pady=(8, 2))
        ctk.CTkLabel(
            diag_hdr,
            text="Diagnostics & Service Callout (Selectable / Copyable)",
            font=F_HEAD,
            text_color=FG,
            anchor="w",
        ).pack(side="left")

        ctk.CTkButton(
            diag_hdr,
            text="Copy Text",
            width=84,
            height=24,
            corner_radius=5,
            fg_color=CARD,
            hover_color=CARD_HI,
            border_width=1,
            border_color=BORDER,
            text_color=FG_MED,
            font=F_SMALL,
            command=self._copy_diagnostics_to_clipboard,
        ).pack(side="right")

        self._diag_text = ctk.CTkTextbox(
            self._diag_box,
            font=F_MONO,
            fg_color=BG,
            text_color=FG_MED,
            corner_radius=6,
            border_width=1,
            border_color=BORDER,
            wrap="word",
            height=110,
        )
        self._diag_text.pack(fill="both", expand=True, padx=14, pady=(2, 12))
        self._set_diagnostics_text("Checking infrastructure service health...", FG_MED)

    def _set_diagnostics_text(self, message: str, color: str = FG_MED) -> None:
        self._diag_text.configure(state="normal", text_color=color)
        self._diag_text.delete("1.0", "end")
        self._diag_text.insert("1.0", message)
        self._diag_text.configure(state="disabled")

    def _copy_diagnostics_to_clipboard(self) -> None:
        try:
            content = self._diag_text.get("1.0", "end").strip()
            self.clipboard_clear()
            self.clipboard_append(content)
            self._on_status_message("Diagnostics copied to clipboard", GREEN)
        except Exception:
            pass

    def poll_services(self) -> None:
        if not self._service_check_busy:
            self._service_check_busy = True
            self._service_mgr.check_all_async(
                lambda statuses, details, summary: self.after(
                    0, lambda: self._apply_service_statuses(statuses, details, summary)
                )
            )

    def trigger_check_now(self) -> None:
        if self._service_check_busy:
            return
        self._service_check_busy = True
        self._set_diagnostics_text("Refreshing service status...", FG_MED)
        self._service_mgr.check_all_async(
            lambda statuses, details, summary: self.after(
                0, lambda: self._apply_service_statuses(statuses, details, summary)
            )
        )

    def _apply_service_statuses(
        self,
        statuses: Dict[str, bool],
        details: Dict[str, str],
        summary: str,
    ) -> None:
        self._service_check_busy = False
        all_ok = True
        lines: List[str] = []
        for key, badge in self._service_badges.items():
            is_up = bool(statuses.get(key, False))
            if not is_up:
                all_ok = False
            badge.configure(
                text="● Running" if is_up else "● Offline",
                text_color=GREEN if is_up else RED,
                fg_color=GREEN_D if is_up else RED_D,
            )
            if key in self._service_detail_lbls and key in details:
                self._service_detail_lbls[key].configure(text=details[key])
            lines.append(f"[{'OK' if is_up else 'OFFLINE'}] {key}: {details.get(key, '')}")

        detail_block = "\n".join(lines)
        if not statuses.get("Docker", False):
            self._diag_box.configure(border_color=RED)
            msg = f"Docker Desktop is not running, please start it and retry.\n\n{detail_block}"
            self._set_diagnostics_text(msg, RED)
        elif not all_ok:
            self._diag_box.configure(border_color=YELLOW)
            self._set_diagnostics_text(f"{summary}\n\n{detail_block}", YELLOW)
        else:
            self._diag_box.configure(border_color=GREEN)
            self._set_diagnostics_text(f"{summary}\n\n{detail_block}", GREEN)

    def _handle_start_services(self) -> None:
        self._start_services_btn.configure(state="disabled", text="Starting Services...")
        self._diag_box.configure(border_color=ACCENT)
        self._set_diagnostics_text(
            "Running 'docker compose up -d' in background... Please wait.",
            FG,
        )

        def _on_done(ok: bool, msg: str, statuses: Dict[str, bool], details: Dict[str, str]) -> None:
            def _ui() -> None:
                self._start_services_btn.configure(state="normal", text="▶  Start Services")
                self._apply_service_statuses(statuses, details, msg)
                if not ok:
                    self._diag_box.configure(border_color=RED)
                    detail_lines = "\n".join(f"• {k}: {v}" for k, v in details.items())
                    self._set_diagnostics_text(f"{msg}\n\n{detail_lines}", RED)

            self.after(0, _ui)

        self._service_mgr.start_services_async(_on_done)
