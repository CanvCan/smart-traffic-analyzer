"""
launcher/app.py

Main desktop window and tab router (Composition Root) for Smart Traffic Analyzer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import customtkinter as ctk

from launcher.ipc_runner import AnalysisProcessManager
from launcher.tabs.logs_tab import LogsTab
from launcher.tabs.services_tab import ServicesTab
from launcher.tabs.settings_tab import SettingsTab
from launcher.tabs.source_tab import SourceTab
from launcher.theme import (
    ACCENT,
    BG,
    BORDER,
    CARD_HI,
    F_HEAD,
    F_SMALL,
    F_TITLE,
    FG,
    FG_MED,
    SURFACE,
)
from traffic_analyzer.infrastructure.config_loader import (
    AppConfig,
    CameraConfig,
    ModelConfig,
    save_config,
)
from traffic_analyzer.infrastructure.service_manager import ServiceManager

_DEFAULT_CFG_PATH = _ROOT / "traffic_analyzer" / "config.json"
_DEFAULT_CAMERAS_DIR = _ROOT / "traffic_analyzer" / "cameras"


class SmartTrafficLauncherApp(ctk.CTk):
    """Desktop control plane window and tab router."""

    TAB_SERVICES = "services"
    TAB_SOURCE = "source"
    TAB_SETTINGS = "settings"
    TAB_LOGS = "logs"
    TAB_MODEL = TAB_SETTINGS

    def __init__(
        self,
        config_path: str | Path = _DEFAULT_CFG_PATH,
        cameras_dir: str | Path = _DEFAULT_CAMERAS_DIR,
    ) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")

        self._config_path = Path(config_path).resolve()
        self._project_root = _ROOT
        self._cameras_dir = Path(cameras_dir).resolve()
        self._cameras_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_config_exists()

        self.title("Smart Traffic Analyzer — Control Plane")
        self.minsize(960, 660)
        self.configure(fg_color=BG)
        self._center_window(1080, 730)

        self._service_mgr = ServiceManager(self._project_root)
        self._proc_mgr = AnalysisProcessManager(on_state_change=self._on_worker_state_change)

        self._tab_frames: Dict[str, ctk.CTkFrame] = {}
        self._tab_buttons: Dict[str, ctk.CTkButton] = {}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close_window)
        self._switch_tab(self.TAB_SERVICES)

        self.after(120, self._poll_services_loop)
        self.after(150, self._poll_worker_loop)
        self.after(200, self._poll_logs_loop)

    def _ensure_config_exists(self) -> None:
        if not self._config_path.exists():
            default_cfg = AppConfig(camera=CameraConfig(), model=ModelConfig())
            save_config(default_cfg, self._config_path)

    def _center_window(self, w: int, h: int) -> None:
        self.geometry(f"{w}x{h}")
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2)}")

    def _set_top_status(self, text: str, color: str = FG_MED) -> None:
        self._top_status_lbl.configure(text=text, text_color=color)

    def _build_ui(self) -> None:
        header = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0, height=58)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="Smart Traffic Analyzer",
            font=F_TITLE,
            text_color=FG,
        ).pack(side="left", padx=22)

        nav_bar = ctk.CTkFrame(header, fg_color=BG, corner_radius=8, border_width=1, border_color=BORDER)
        nav_bar.pack(side="left", padx=12, pady=10)

        tabs = [
            (self.TAB_SERVICES, "1. Services"),
            (self.TAB_SOURCE, "2. Analysis & Source"),
            (self.TAB_SETTINGS, "3. Settings"),
            (self.TAB_LOGS, "4. Logs & Streams"),
        ]
        for key, title in tabs:
            btn = ctk.CTkButton(
                nav_bar,
                text=title,
                width=142,
                height=32,
                corner_radius=6,
                font=F_HEAD,
                fg_color="transparent",
                hover_color=CARD_HI,
                text_color=FG_MED,
                command=lambda k=key: self._switch_tab(k),
            )
            btn.pack(side="left", padx=3, pady=3)
            self._tab_buttons[key] = btn

        self._top_status_lbl = ctk.CTkLabel(
            header,
            text="Ready",
            font=F_SMALL,
            text_color=FG_MED,
        )
        self._top_status_lbl.pack(side="right", padx=22)

        ctk.CTkFrame(self, height=1, fg_color=BORDER, corner_radius=0).pack(fill="x")

        self._content_area = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self._content_area.pack(fill="both", expand=True)

        self._services_tab = ServicesTab(
            self._content_area,
            service_mgr=self._service_mgr,
            on_status_message=self._set_top_status,
        )
        self._settings_tab = SettingsTab(
            self._content_area,
            project_root=self._project_root,
            config_path=self._config_path,
            on_status_message=self._set_top_status,
        )
        self._source_tab = SourceTab(
            self._content_area,
            project_root=self._project_root,
            config_path=self._config_path,
            cameras_dir=self._cameras_dir,
            proc_mgr=self._proc_mgr,
            on_before_start=lambda: self._settings_tab.save_settings_to_disk(quiet=True),
            on_status_message=self._set_top_status,
        )
        self._logs_tab = LogsTab(
            self._content_area,
            event_queue=self._proc_mgr.event_queue,
        )

        self._tab_frames = {
            self.TAB_SERVICES: self._services_tab,
            self.TAB_SOURCE: self._source_tab,
            self.TAB_SETTINGS: self._settings_tab,
            self.TAB_LOGS: self._logs_tab,
        }

    def _switch_tab(self, tab_key: str) -> None:
        for k, frame in self._tab_frames.items():
            if k == tab_key:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()

        for k, btn in self._tab_buttons.items():
            active = k == tab_key
            btn.configure(
                fg_color=ACCENT if active else "transparent",
                text_color="#ffffff" if active else FG_MED,
            )

    def _poll_services_loop(self) -> None:
        self._services_tab.poll_services()
        self.after(5000, self._poll_services_loop)

    def _poll_worker_loop(self) -> None:
        self._proc_mgr.poll()
        self.after(150, self._poll_worker_loop)

    def _poll_logs_loop(self) -> None:
        self._logs_tab.poll_logs()
        self.after(250, self._poll_logs_loop)

    def _on_worker_state_change(self, state: str, payload: Dict[str, Any]) -> None:
        detail = str(payload.get("detail", ""))
        self._source_tab.apply_worker_state(state, detail)
        if state in ("STOPPED", "ERROR"):
            self.deiconify()
            self.lift()

    def _on_close_window(self) -> None:
        self._logs_tab.stop_streamers()
        if self._proc_mgr.is_running:
            self._proc_mgr.request_stop()
        self.destroy()


LauncherApp = SmartTrafficLauncherApp


def main() -> None:
    app = SmartTrafficLauncherApp()
    app.mainloop()


if __name__ == "__main__":
    main()
