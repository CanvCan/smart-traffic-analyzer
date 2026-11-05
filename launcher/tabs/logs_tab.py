"""
launcher/tabs/logs_tab.py

Live Kafka Telemetry Event stream (Producer via IPC Queue) and Spark Structured Streaming
container log viewer (Consumer / Q1-Q11).
"""

from __future__ import annotations

from datetime import datetime
import queue
import tkinter as tk
from typing import Any, Dict, List, Optional, Tuple

import customtkinter as ctk

from launcher.theme import (
    ACCENT,
    ACCENT_H,
    BG,
    BORDER,
    CARD,
    CARD_HI,
    F_HEAD,
    F_MONO,
    F_SMALL,
    FG,
    FG_MED,
    FONT_FAMILY,
    GREEN,
    RED,
    SURFACE,
    YELLOW,
)
from traffic_analyzer.infrastructure.docker_logs import DockerLogStreamer


def _format_ts(raw_ts: Any) -> str:
    if isinstance(raw_ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw_ts)).strftime("%H:%M:%S.%f")[:-3]
        except Exception:
            return f"{float(raw_ts):.3f}"
    if raw_ts:
        return str(raw_ts)
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def format_telemetry_event(event: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Format a structured telemetry event dict from EventBuilder into one or more
    `(formatted_line_with_newline, tag_name)` tuples for the Kafka Telemetry console.
    """
    ev_type = str(event.get("event_type", "vehicle_detected"))
    ts_str = _format_ts(event.get("timestamp"))
    lines: List[Tuple[str, str]] = []

    if ev_type == "vehicle_detected":
        veh = event.get("vehicle", {})
        kin = event.get("kinematics", {})
        anom = event.get("anomaly", {})

        track_id = veh.get("id", event.get("track_id", 0))
        class_name = veh.get("class", event.get("class_name", "Vehicle"))
        lane_id = veh.get("lane") or event.get("lane_id") or "UNASSIGNED"
        speed_px = float(kin.get("speed_px_per_sec", event.get("speed_px", 0.0)))
        confidence = float(veh.get("confidence", event.get("confidence", 1.0)))

        line = (
            f"[{ts_str}] [EVENT] ID: #{track_id} | Class: {class_name} | "
            f"Lane: {lane_id} | Speed: {speed_px:.1f} px/s | Conf: {confidence:.2f}\n"
        )
        lines.append((line, "EVENT"))

        if anom.get("is_anomaly"):
            anom_type = str(anom.get("type", "anomaly")).upper()
            anom_line = (
                f"[{ts_str}] [ANOMALY] ID: #{track_id} | Type: {anom_type} | "
                f"Class: {class_name} | Lane: {lane_id} | Speed: {speed_px:.1f} px/s\n"
            )
            lines.append((anom_line, "ANOMALY"))

    elif ev_type == "anomaly_detected":
        track_id = event.get("track_id", event.get("vehicle_id", 0))
        class_name = event.get("class", event.get("class_name", "Vehicle"))
        lane_id = event.get("lane") or event.get("lane_id") or "UNASSIGNED"
        speed_px = float(event.get("speed_px", 0.0))
        anom_type = str(event.get("type", event.get("anomaly_type", "ANOMALY"))).upper()
        line = (
            f"[{ts_str}] [ANOMALY] ID: #{track_id} | Type: {anom_type} | "
            f"Class: {class_name} | Lane: {lane_id} | Speed: {speed_px:.1f} px/s\n"
        )
        lines.append((line, "ANOMALY"))

    elif ev_type == "traffic_snapshot":
        frame_id = event.get("frame_id", 0)
        counts = event.get("counts", {})
        speed = event.get("speed", {})
        density = event.get("density", {})

        total = int(counts.get("total", 0))
        status = str(density.get("status", "FREE"))
        avg_speed = float(speed.get("average_px", 0.0))
        occupancy = float(density.get("occupancy_ratio", 0.0))

        line = (
            f"[{ts_str}] [SNAPSHOT] Frame: #{frame_id} | Total: {total} | "
            f"Status: {status} | AvgSpeed: {avg_speed:.1f} px/s | Occ: {occupancy:.2f}\n"
        )
        lines.append((line, "SNAPSHOT"))

    else:
        lines.append((f"[{ts_str}] [{ev_type.upper()}] {event}\n", "EVENT"))

    return lines


class LogsTab(ctk.CTkFrame):
    """Dual-pane live telemetry producer console and Spark Structured Streaming consumer log viewer."""

    def __init__(self, parent: tk.Misc, event_queue: Optional[Any] = None) -> None:
        super().__init__(parent, fg_color=BG, corner_radius=0)
        self._event_queue = event_queue
        self._local_event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=2000)
        self._spark_streamer = DockerLogStreamer(
            container_candidates=["spark_streaming", "spark_processor"],
            service_name="spark-processor",
        )
        self._logs_autoscroll_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._spark_streamer.start()

    def set_event_queue(self, event_queue: Any) -> None:
        self._event_queue = event_queue

    def push_telemetry_event(self, event: Dict[str, Any]) -> None:
        try:
            self._local_event_queue.put_nowait(event)
        except queue.Full:
            try:
                self._local_event_queue.get_nowait()
                self._local_event_queue.put_nowait(event)
            except Exception:
                pass

    def _build_ui(self) -> None:
        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", padx=28, pady=(14, 8))

        ctk.CTkLabel(
            top_bar,
            text="Live Telemetry & Streaming Consoles",
            font=(FONT_FAMILY, 14, "bold"),
            text_color=FG,
        ).pack(side="left")

        ctk.CTkButton(
            top_bar,
            text="Clear Consoles",
            width=118,
            height=30,
            corner_radius=6,
            fg_color=CARD,
            hover_color=CARD_HI,
            border_width=1,
            border_color=BORDER,
            text_color=FG,
            font=F_SMALL,
            command=self.clear_consoles,
        ).pack(side="right")

        ctk.CTkCheckBox(
            top_bar,
            text="Auto-scroll",
            variable=self._logs_autoscroll_var,
            font=F_SMALL,
            text_color=FG,
            fg_color=ACCENT,
            hover_color=ACCENT_H,
            checkbox_width=18,
            checkbox_height=18,
        ).pack(side="right", padx=(0, 16))

        panes = ctk.CTkFrame(self, fg_color="transparent")
        panes.pack(fill="both", expand=True, padx=28, pady=(0, 18))
        panes.columnconfigure(0, weight=1)
        panes.columnconfigure(1, weight=1)
        panes.rowconfigure(0, weight=1)

        left_card = ctk.CTkFrame(panes, fg_color=SURFACE, corner_radius=8, border_width=1, border_color=BORDER)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self._kafka_hdr_lbl = ctk.CTkLabel(
            left_card,
            text="Kafka Telemetry Stream (Producer)",
            font=F_HEAD,
            text_color=ACCENT,
            anchor="w",
        )
        self._kafka_hdr_lbl.pack(fill="x", padx=12, pady=(8, 4))

        self._kafka_log_box = ctk.CTkTextbox(
            left_card,
            font=F_MONO,
            fg_color=BG,
            text_color=FG,
            corner_radius=6,
            border_width=1,
            border_color=BORDER,
            wrap="none",
        )
        self._kafka_log_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._configure_telemetry_tags()
        self._insert_initial_telemetry_banner()

        right_card = ctk.CTkFrame(panes, fg_color=SURFACE, corner_radius=8, border_width=1, border_color=BORDER)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self._spark_hdr_lbl = ctk.CTkLabel(
            right_card,
            text="Spark Structured Streaming (Consumer / Q1-Q11)",
            font=F_HEAD,
            text_color=GREEN,
            anchor="w",
        )
        self._spark_hdr_lbl.pack(fill="x", padx=12, pady=(8, 4))

        self._spark_log_box = ctk.CTkTextbox(
            right_card,
            font=F_MONO,
            fg_color=BG,
            text_color=FG,
            corner_radius=6,
            border_width=1,
            border_color=BORDER,
            wrap="none",
        )
        self._spark_log_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._spark_log_box.configure(state="disabled")

    def _configure_telemetry_tags(self) -> None:
        inner = getattr(self._kafka_log_box, "_textbox", None)
        if inner is not None:
            inner.tag_config("EVENT", foreground=FG)
            inner.tag_config("ANOMALY", foreground=RED)
            inner.tag_config("SNAPSHOT", foreground=YELLOW)
            inner.tag_config("INFO", foreground=FG_MED)

    def _insert_initial_telemetry_banner(self) -> None:
        self._kafka_log_box.configure(state="normal")
        inner = getattr(self._kafka_log_box, "_textbox", None)
        msg = "[Telemetry Ready] Waiting for analysis pipeline events (vehicle_detected, traffic_snapshot, anomaly)...\n"
        if inner is not None:
            inner.insert("end", msg, "INFO")
        else:
            self._kafka_log_box.insert("end", msg)
        self._kafka_log_box.configure(state="disabled")

    def _drain_telemetry_events(self) -> None:
        events: List[Dict[str, Any]] = []
        for q in (self._event_queue, self._local_event_queue):
            if q is None:
                continue
            while len(events) < 120:
                try:
                    ev = q.get_nowait()
                    if isinstance(ev, dict):
                        events.append(ev)
                except queue.Empty:
                    break
                except Exception:
                    break

        if not events:
            return

        self._kafka_log_box.configure(state="normal")
        inner = getattr(self._kafka_log_box, "_textbox", None)
        for ev in events:
            for text_line, tag in format_telemetry_event(ev):
                if inner is not None:
                    inner.insert("end", text_line, tag)
                else:
                    self._kafka_log_box.insert("end", text_line)

        line_count = int(self._kafka_log_box.index("end-1c").split(".")[0])
        if line_count > 800:
            self._kafka_log_box.delete("1.0", f"{line_count - 600}.0")
        if self._logs_autoscroll_var.get():
            self._kafka_log_box.see("end")
        self._kafka_log_box.configure(state="disabled")

    def _drain_stream_to_textbox(self, q: "queue.Queue[str]", box: ctk.CTkTextbox) -> None:
        batch: List[str] = []
        while len(batch) < 120:
            try:
                batch.append(q.get_nowait())
            except queue.Empty:
                break
        if not batch:
            return
        box.configure(state="normal")
        box.insert("end", "".join(batch))
        line_count = int(box.index("end-1c").split(".")[0])
        if line_count > 800:
            box.delete("1.0", f"{line_count - 600}.0")
        if self._logs_autoscroll_var.get():
            box.see("end")
        box.configure(state="disabled")

    def poll_logs(self) -> None:
        self._drain_telemetry_events()
        self._drain_stream_to_textbox(self._spark_streamer.queue, self._spark_log_box)

    def clear_consoles(self) -> None:
        for box in (self._kafka_log_box, self._spark_log_box):
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.configure(state="disabled")

    def stop_streamers(self) -> None:
        self._spark_streamer.stop()
