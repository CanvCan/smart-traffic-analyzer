"""
infrastructure/video_loop.py

Manages video stream reading, display, and user input for the analysis pipeline
using a 3-Thread decoupled architecture:
  1. CaptureThread         — reads frames at native source cadence and stamps FramePacket
  2. InferenceTrackWorker  — pops latest FramePacket via AtomicLatestSlot, runs RF-DETR + ByteTrack
  3. RenderMainLoop        — composites latest OverlaySnapshot onto display frames at 25-30+ FPS

Includes:
  - AtomicLatestSlot for O(1) latest-frame handoff without queue backlog
  - Windows 8.3 short-path resolution for non-ASCII local file paths
  - Hybrid timestamp strategy (monotonic clock for live streams, frame_id / fps for files)
  - Separate-process Queue IPC (cmd_queue / status_queue)
  - Window X button close detection (cv2.WND_PROP_VISIBLE < 1)
  - Interactive HUD bar & keyboard shortcuts:
      R -> Return to Launcher
      Q / Esc -> Quit
      L -> Toggle Lane overlay
      C -> Toggle Class legend
      G -> Toggle Ghost boxes
      T -> Toggle Vehicle labels
      H -> Toggle HUD bar
      F -> Toggle Fullscreen
      S -> Save Screenshot
      Space / [/] / ,/. -> Pause, Seek ±5s, Step ±1 frame (Video File only)
"""

import inspect
import os
from pathlib import Path
import queue
import threading
import time
import traceback
from typing import Any, Callable, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from traffic_analyzer.domain.models import FramePacket, OverlaySnapshot

WIN_NAME = "Smart Traffic Analyzer"


class AtomicLatestSlot:
    """
    Thread-safe single-item slot that always retains only the most recent item.
    Eliminates multi-frame queue lag between capture, inference, and rendering.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: Any = None
        self._event = threading.Event()

    def put(self, item: Any) -> None:
        with self._lock:
            self._item = item
            self._event.set()

    def pop_latest(self, timeout: float = 0.0) -> Any:
        if timeout > 0.0:
            self._event.wait(timeout=timeout)
        with self._lock:
            item = self._item
            self._item = None
            self._event.clear()
            return item

    def peek_latest(self) -> Any:
        with self._lock:
            return self._item

    def clear(self) -> None:
        with self._lock:
            self._item = None
            self._event.clear()


class CaptureThread(threading.Thread):
    """Dedicated frame-capture thread writing FramePackets into AtomicLatestSlots."""

    def __init__(self, target: Callable[[], None], name: str = "CaptureThread") -> None:
        super().__init__(target=target, daemon=True, name=name)


class InferenceTrackWorker(threading.Thread):
    """Dedicated single-consumer inference and ByteTrack worker thread."""

    def __init__(self, target: Callable[[], None], name: str = "InferenceTrackWorker") -> None:
        super().__init__(target=target, daemon=True, name=name)


def _safe_win32_video_path(path_str: str) -> str:
    """Resolve Windows 8.3 short path for local files containing Turkish/non-ASCII characters."""
    if os.name != "nt" or path_str.startswith("http"):
        return path_str
    try:
        path_str.encode("ascii")
        return path_str
    except UnicodeEncodeError:
        pass
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        res = ctypes.windll.kernel32.GetShortPathNameW(path_str, buf, 1024)
        if res > 0:
            return buf.value
    except Exception:
        pass
    return path_str


def _setup_cv2_window(win_w: int, win_h: int) -> None:
    """Center the OpenCV window on screen and bring it to the foreground."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        scr_w = user32.GetSystemMetrics(0)
        scr_h = user32.GetSystemMetrics(1)
        x = max((scr_w - win_w) // 2, 0)
        y = max((scr_h - win_h) // 2, 0)
        cv2.moveWindow(WIN_NAME, x, y)
        hwnd = user32.FindWindowW(None, WIN_NAME)
        if hwnd:
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


_MJPEG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://izum.izmir.bel.tr/",
    "Accept": "multipart/x-mixed-replace,image/jpeg,*/*",
    "Connection": "keep-alive",
}

_MJPEG_MAX_RETRIES = 10
_MJPEG_RETRY_DELAY = 2


def _iter_mjpeg(url: str, stop_event: Optional[threading.Event] = None) -> Iterator[np.ndarray]:
    """Read an MJPEG-over-HTTP(S) stream and yield decoded BGR frames."""
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    attempt = 0
    while attempt < _MJPEG_MAX_RETRIES:
        if stop_event is not None and stop_event.is_set():
            return
        try:
            with requests.get(
                url, stream=True, verify=False, timeout=15, headers=_MJPEG_HEADERS
            ) as r:
                r.raise_for_status()
                ct = r.headers.get("Content-Type", "unknown")
                print(f"[VideoLoop] Content-Type: {ct}")

                buf = b""
                frame_count = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if stop_event is not None and stop_event.is_set():
                        return
                    if not chunk:
                        continue
                    buf += chunk
                    while True:
                        start = buf.find(b"\xff\xd8")
                        end = buf.find(b"\xff\xd9")
                        if start == -1 or end == -1 or end < start:
                            break
                        jpg = buf[start:end + 2]
                        buf = buf[end + 2:]
                        frame = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        if frame is not None:
                            frame_count += 1
                            attempt = 0
                            yield frame

                print(f"[VideoLoop] Stream ended after {frame_count} frames. Reconnecting...")

        except requests.exceptions.RequestException as e:
            attempt += 1
            print(f"[VideoLoop] Stream error (attempt {attempt}/{_MJPEG_MAX_RETRIES}): {e}")

        if stop_event is not None and stop_event.wait(_MJPEG_RETRY_DELAY):
            return
        elif stop_event is None:
            time.sleep(_MJPEG_RETRY_DELAY)

    print("[VideoLoop] Max retries reached. Giving up.")


class VideoLoop:
    """
    Reads frames from a video file, camera stream, or MJPEG URL and drives
    the 3-thread decoupled pipeline (CaptureThread -> InferenceTrackWorker -> RenderMainLoop).
    """

    @staticmethod
    def _detect_is_live(video_path: str) -> bool:
        vp = (video_path or "").strip().lower()
        return (
            vp.startswith(("http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://"))
            or "mjpeg" in vp
            or "mjpg" in vp
        )

    def __init__(
        self,
        video_path: str,
        display_width: int,
        cmd_queue: Any = None,
        status_queue: Any = None,
        renderer: Any = None,
        source_name: str = "",
        lanes: Optional[list] = None,
        is_live: Optional[bool] = None,
    ):
        self._video_path = video_path
        self._display_width = display_width
        self._cmd_queue = cmd_queue
        self._status_queue = status_queue
        self._renderer = renderer
        self._source_name = source_name or Path(video_path).stem
        self._lanes = lanes if lanes is not None else []
        self.is_live: bool = self._detect_is_live(video_path) if is_live is None else bool(is_live)
        if self._renderer is not None and hasattr(self._renderer, "is_live"):
            self._renderer.is_live = self.is_live

        self._show_hud = True
        self._fullscreen = False
        self._fps_ema = 0.0
        self._inf_fps_ema = 0.0
        self._last_ts = time.perf_counter()
        self._hud_buttons: List[Tuple[int, int, int, int, str]] = []
        self._pending_action: Optional[str] = None
        self._exit_reason = "completed"

    def run(
        self,
        frame_callback: Callable,
        on_events: Callable,
    ) -> None:
        is_mjpeg = "mjpeg" in self._video_path.lower()
        if is_mjpeg:
            self._run_mjpeg(frame_callback, on_events)
        else:
            self._run_cv2(frame_callback, on_events)

    @staticmethod
    def _invoke_frame_callback(
        frame_callback: Callable,
        frame: np.ndarray,
        frame_id: int,
        frame_timestamp: float,
    ) -> list:
        try:
            sig = inspect.signature(frame_callback)
            if len(sig.parameters) >= 3:
                return frame_callback(frame, frame_id, frame_timestamp)
        except Exception:
            pass
        return frame_callback(frame, frame_id)

    def _start_inference_worker(
        self,
        stop_event: threading.Event,
        capture_to_infer_slot: AtomicLatestSlot,
        overlay_slot: AtomicLatestSlot,
        frame_callback: Callable,
        on_events: Callable,
    ) -> threading.Thread:
        bound_self = getattr(frame_callback, "__self__", None)
        has_packet_api = bound_self is not None and hasattr(bound_self, "process_packet")
        if has_packet_api and not self._lanes and hasattr(bound_self, "config"):
            self._lanes = getattr(bound_self.config, "lanes", [])

        def _worker() -> None:
            while not stop_event.is_set():
                packet: Optional[FramePacket] = capture_to_infer_slot.pop_latest(timeout=0.02)
                if packet is None:
                    continue
                try:
                    if has_packet_api:
                        events, snapshot, inf_fps = bound_self.process_packet(packet)
                        overlay_slot.put(snapshot)
                        self._inf_fps_ema = (
                            inf_fps
                            if self._inf_fps_ema <= 0.0
                            else (0.80 * self._inf_fps_ema + 0.20 * inf_fps)
                        )
                    else:
                        t0 = time.perf_counter()
                        work_frame = packet.frame.copy()
                        res = self._invoke_frame_callback(
                            frame_callback, work_frame, packet.frame_id, packet.timestamp
                        )
                        if isinstance(res, tuple) and len(res) == 2:
                            events, snapshot = res
                            overlay_slot.put(snapshot)
                        else:
                            events = res
                        inf_fps = 1.0 / max(1e-4, time.perf_counter() - t0)
                        self._inf_fps_ema = (
                            inf_fps
                            if self._inf_fps_ema <= 0.0
                            else (0.80 * self._inf_fps_ema + 0.20 * inf_fps)
                        )
                    if events:
                        on_events(events)
                except Exception as exc:
                    print(f"[VideoLoop] ERROR in InferenceTrackWorker frame {packet.frame_id}: {exc}")
                    traceback.print_exc()

        worker = threading.Thread(
            target=_worker,
            daemon=True,
            name="InferenceTrackWorker",
        )
        worker.start()
        return worker

    def _on_mouse_click(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self._show_hud:
            for x1, y1, x2, y2, action in self._hud_buttons:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._pending_action = action
                    break

    def _check_ipc_stop_or_window_closed(self, window_ready: bool) -> bool:
        if self._cmd_queue is not None:
            try:
                msg = self._cmd_queue.get_nowait()
                if isinstance(msg, dict) and msg.get("cmd") == "STOP":
                    self._exit_reason = "user_stopped"
                    return True
            except queue.Empty:
                pass
            except Exception:
                pass

        if window_ready:
            try:
                if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    self._exit_reason = "window_closed"
                    return True
            except cv2.error:
                self._exit_reason = "window_closed"
                return True
        return False

    def _notify_stopped(self) -> None:
        if self._status_queue is not None:
            try:
                self._status_queue.put_nowait({
                    "state": "STOPPED",
                    "reason": self._exit_reason,
                    "detail": f"Analysis exited ({self._exit_reason}).",
                })
            except Exception:
                pass

    def _update_fps(self) -> None:
        now = time.perf_counter()
        dt = max(1e-4, now - self._last_ts)
        self._last_ts = now
        inst_fps = 1.0 / dt
        self._fps_ema = inst_fps if self._fps_ema <= 0.0 else (0.85 * self._fps_ema + 0.15 * inst_fps)

    def _draw_hud_bar(self, disp_frame: np.ndarray, is_live: bool, paused: bool) -> None:
        self._hud_buttons.clear()
        if not self._show_hud:
            return
        if self._renderer is not None and hasattr(self._renderer, "draw_hud"):
            self._renderer.draw_hud(
                disp_frame,
                fps=self._fps_ema,
                paused=(paused and not is_live),
                is_live=is_live,
            )
            return

        if is_live:
            info_txt = f"FPS: {self._fps_ema:4.1f}  |  LIVE | Q: Quit"
        else:
            status_part = "  |  PAUSED" if paused else ""
            info_txt = f"FPS: {self._fps_ema:4.1f}{status_part}  |  Space: Pause | Q: Quit"
        (tw, th), _ = cv2.getTextSize(info_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        x1, y1, x2, y2 = 12, 12, 28 + tw, 24 + th
        roi = disp_frame[y1:y2, x1:x2]
        fill = np.empty_like(roi)
        fill[:] = (16, 19, 42)
        cv2.addWeighted(fill, 0.78, roi, 0.22, 0.0, roi)
        cv2.rectangle(disp_frame, (x1, y1), (x2, y2), (58, 66, 110), 1)
        cv2.putText(
            disp_frame,
            info_txt,
            (x1 + 8, y1 + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (225, 230, 248),
            1,
            cv2.LINE_AA,
        )

    def _save_screenshot(self, frame: np.ndarray) -> None:
        out_dir = Path("screenshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{self._source_name}_{ts}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"[VideoLoop] Screenshot saved -> {out_path}")

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            WIN_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if self._fullscreen else cv2.WINDOW_NORMAL,
        )

    def _handle_action(
        self,
        action: str,
        disp_frame: np.ndarray,
        is_live: bool,
        paused: bool,
        seek_queue: Optional[queue.Queue] = None,
    ) -> Tuple[bool, bool, bool]:
        r = self._renderer
        if action in ("quit", "return_launcher"):
            self._exit_reason = "return_to_launcher" if action == "return_launcher" else "user_quit"
            return True, paused, False
        elif action == "toggle_lanes" and r is not None:
            r.show_lanes = not getattr(r, "show_lanes", True)
        elif action == "toggle_legend" and r is not None:
            r.show_legend = not getattr(r, "show_legend", True)
        elif action == "toggle_ghosts" and r is not None:
            r.show_ghosts = not getattr(r, "show_ghosts", True)
        elif action == "toggle_labels" and r is not None:
            r.show_labels = not getattr(r, "show_labels", True)
        elif action == "toggle_hud":
            self._show_hud = not self._show_hud
        elif action == "screenshot":
            self._save_screenshot(disp_frame)
        elif action == "fullscreen":
            self._toggle_fullscreen()
        elif not is_live:
            if action == "toggle_pause":
                paused = not paused
                print(f"[VideoLoop] {'PAUSED' if paused else 'RESUMED'}")
            elif action in ("seek_back", "seek_fwd") and seek_queue is not None:
                seek_queue.put(action)
                return False, paused, True
            elif action in ("step_back", "step_fwd") and paused and seek_queue is not None:
                seek_queue.put(action)
                return False, paused, True
        return False, paused, False

    @staticmethod
    def _key_to_action(key: int, is_live: bool = False) -> Optional[str]:
        if key in (ord('q'), ord('Q'), 27):
            return "quit"
        if key in (ord('r'), ord('R')):
            return "return_launcher"
        if key in (ord('l'), ord('L')):
            return "toggle_lanes"
        if key in (ord('c'), ord('C')):
            return "toggle_legend"
        if key in (ord('g'), ord('G')):
            return "toggle_ghosts"
        if key in (ord('t'), ord('T')):
            return "toggle_labels"
        if key in (ord('h'), ord('H')):
            return "toggle_hud"
        if key in (ord('s'), ord('S')):
            return "screenshot"
        if key in (ord('f'), ord('F')):
            return "fullscreen"
        if not is_live:
            if key == ord(' '):
                return "toggle_pause"
            if key == ord('['):
                return "seek_back"
            if key == ord(']'):
                return "seek_fwd"
            if key == ord(','):
                return "step_back"
            if key == ord('.'):
                return "step_fwd"
        return None

    # ── 3-Thread MJPEG path (Live Camera Stream — monotonic timestamp) ────────

    def _run_mjpeg(self, frame_callback: Callable, on_events: Callable) -> None:
        print(f"[VideoLoop] MJPEG stream: {self._video_path}")
        print("[VideoLoop] q/Esc=quit | r=return to launcher | l=lanes | c=legend | g=ghosts | t=labels | s=snap")

        stop_event = threading.Event()
        capture_eof = threading.Event()
        display_slot = AtomicLatestSlot()
        capture_to_infer_slot = AtomicLatestSlot()
        overlay_slot = AtomicLatestSlot()

        def _capture_thread() -> None:
            frame_count = 0
            try:
                for frame in _iter_mjpeg(self._video_path, stop_event=stop_event):
                    if stop_event.is_set():
                        break
                    frame_count += 1
                    packet = FramePacket(
                        frame_id=frame_count,
                        timestamp=time.monotonic(),
                        frame=frame,
                        is_live=True,
                    )
                    display_slot.put(packet)
                    capture_to_infer_slot.put(packet)
            except Exception as exc:
                print(f"[VideoLoop] CaptureThread MJPEG error: {exc}")
            finally:
                capture_eof.set()

        cap_thread = threading.Thread(target=_capture_thread, daemon=True, name="CaptureThread")
        cap_thread.start()
        inf_thread = self._start_inference_worker(
            stop_event, capture_to_infer_slot, overlay_slot, frame_callback, on_events
        )

        dw = self._display_width
        window_ready = False
        last_disp: Optional[np.ndarray] = None

        try:
            while not stop_event.is_set():
                if self._check_ipc_stop_or_window_closed(window_ready):
                    stop_event.set()
                    break

                pkt: Optional[FramePacket] = display_slot.pop_latest(timeout=0.015)
                if pkt is not None:
                    self._update_fps()
                    canvas = pkt.frame.copy()
                    snapshot: Optional[OverlaySnapshot] = overlay_slot.peek_latest()
                    if self._renderer is not None and hasattr(self._renderer, "draw_snapshot"):
                        self._renderer.draw_snapshot(canvas, self._lanes, snapshot)

                    h, w = canvas.shape[:2]
                    win_h = int(dw * h / w)
                    disp = cv2.resize(canvas, (dw, win_h))
                    self._draw_hud_bar(disp, is_live=True, paused=False)
                    last_disp = disp
                    cv2.imshow(WIN_NAME, disp)

                    if not window_ready:
                        _setup_cv2_window(dw, win_h)
                        cv2.setMouseCallback(WIN_NAME, self._on_mouse_click)
                        window_ready = True
                elif capture_eof.is_set():
                    break

                key = cv2.waitKey(1) & 0xFF
                action = self._pending_action or self._key_to_action(key, is_live=True)
                self._pending_action = None
                if action and last_disp is not None:
                    should_break, _, _ = self._handle_action(action, last_disp, True, False)
                    if should_break:
                        stop_event.set()
                        break

                if self._check_ipc_stop_or_window_closed(window_ready):
                    stop_event.set()
                    break
        finally:
            stop_event.set()
            cap_thread.join(timeout=1.0)
            inf_thread.join(timeout=1.0)
            cv2.destroyAllWindows()
            self._notify_stopped()
            print("[VideoLoop] Stream closed.")

    # ── 3-Thread OpenCV / file path (Deterministic PTS for local files) ───────

    def _run_cv2(self, frame_callback: Callable, on_events: Callable) -> None:
        is_live = self.is_live
        if is_live:
            cap = cv2.VideoCapture(self._video_path, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            safe_path = _safe_win32_video_path(self._video_path)
            cap = cv2.VideoCapture(safe_path)

        if not cap.isOpened():
            print(f"[VideoLoop] ERROR — Cannot open: {self._video_path}")
            self._exit_reason = "open_error"
            self._notify_stopped()
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if source_fps <= 0 or source_fps > 240:
            source_fps = 25.0
        frame_interval = 1.0 / source_fps

        print(f"[VideoLoop] Processing: {self._video_path} (fps={source_fps:.1f}, is_live={is_live})")
        if is_live:
            print("[VideoLoop] q/Esc=quit | r=return to launcher | l=lanes | c=legend | g=ghosts | t=labels | s=snap")
        else:
            print("[VideoLoop] q/Esc=quit | r=return to launcher | l=lanes | c=legend | g=ghosts | t=labels | space=pause")

        stop_event = threading.Event()
        capture_eof = threading.Event()
        pause_state = {"paused": False}
        seek_queue: queue.Queue = queue.Queue()

        display_slot = AtomicLatestSlot()
        capture_to_infer_slot = AtomicLatestSlot()
        overlay_slot = AtomicLatestSlot()

        def _capture_thread() -> None:
            frame_count = 0
            next_deadline = time.perf_counter()
            try:
                while not stop_event.is_set():
                    force_read = False
                    if not is_live:
                        try:
                            seek_cmd = seek_queue.get_nowait()
                            cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
                            if seek_cmd in ("seek_back", "seek_fwd"):
                                delta = int(round(5.0 * source_fps)) * (-1 if seek_cmd == "seek_back" else 1)
                                target = max(0, int(cur + delta))
                                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                                frame_count = target
                                overlay_slot.clear()
                                force_read = True
                            elif seek_cmd in ("step_back", "step_fwd"):
                                target = max(0, int(cur - 2)) if seek_cmd == "step_back" else max(0, int(cur))
                                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                                frame_count = target
                                force_read = True
                        except queue.Empty:
                            pass

                        if pause_state["paused"] and not force_read:
                            time.sleep(0.015)
                            next_deadline = time.perf_counter()
                            continue

                    ret, frame = cap.read()
                    if not ret:
                        capture_eof.set()
                        break

                    frame_count += 1
                    frame_ts = time.monotonic() if is_live else (frame_count / source_fps)
                    packet = FramePacket(
                        frame_id=frame_count,
                        timestamp=frame_ts,
                        frame=frame,
                        is_live=is_live,
                    )
                    display_slot.put(packet)
                    capture_to_infer_slot.put(packet)

                    if not is_live and not force_read:
                        next_deadline += frame_interval
                        sleep_for = next_deadline - time.perf_counter()
                        if sleep_for > 0.0:
                            time.sleep(sleep_for)
                        else:
                            next_deadline = time.perf_counter()
            except Exception as exc:
                print(f"[VideoLoop] CaptureThread error: {exc}")
                capture_eof.set()

        cap_thread = threading.Thread(target=_capture_thread, daemon=True, name="CaptureThread")
        cap_thread.start()
        inf_thread = self._start_inference_worker(
            stop_event, capture_to_infer_slot, overlay_slot, frame_callback, on_events
        )

        dw = self._display_width
        window_ready = False
        last_raw: Optional[np.ndarray] = None
        last_disp: Optional[np.ndarray] = None

        try:
            while not stop_event.is_set():
                if self._check_ipc_stop_or_window_closed(window_ready):
                    stop_event.set()
                    break

                pkt: Optional[FramePacket] = display_slot.pop_latest(timeout=0.012)
                if pkt is not None:
                    last_raw = pkt.frame
                    self._update_fps()
                elif capture_eof.is_set() and capture_to_infer_slot.peek_latest() is None:
                    break

                if last_raw is not None:
                    canvas = last_raw.copy()
                    snapshot: Optional[OverlaySnapshot] = overlay_slot.peek_latest()
                    if self._renderer is not None and hasattr(self._renderer, "draw_snapshot"):
                        self._renderer.draw_snapshot(canvas, self._lanes, snapshot)

                    h, w = canvas.shape[:2]
                    win_h = int(dw * h / w)
                    disp = cv2.resize(canvas, (dw, win_h))
                    self._draw_hud_bar(disp, is_live=is_live, paused=(not is_live and pause_state["paused"]))
                    last_disp = disp
                    cv2.imshow(WIN_NAME, disp)

                    if not window_ready:
                        _setup_cv2_window(dw, win_h)
                        cv2.setMouseCallback(WIN_NAME, self._on_mouse_click)
                        window_ready = True

                wait_ms = 30 if (not is_live and pause_state["paused"]) else 1
                key = cv2.waitKey(wait_ms) & 0xFF
                action = self._pending_action or self._key_to_action(key, is_live=is_live)
                self._pending_action = None
                if action and last_disp is not None:
                    should_break, new_paused, _ = self._handle_action(
                        action, last_disp, is_live, pause_state["paused"], seek_queue=seek_queue
                    )
                    pause_state["paused"] = False if is_live else new_paused
                    if should_break:
                        stop_event.set()
                        break

                if self._check_ipc_stop_or_window_closed(window_ready):
                    stop_event.set()
                    break
        finally:
            stop_event.set()
            cap_thread.join(timeout=1.0)
            inf_thread.join(timeout=1.0)
            cap.release()
            cv2.destroyAllWindows()
            self._notify_stopped()
            print("[VideoLoop] Stream closed.")
