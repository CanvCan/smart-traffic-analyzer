"""
core/video_loop.py

Manages video stream reading, display, and user input for the analysis pipeline.

Responsibilities:
  - Open and release VideoCapture
  - Read frames sequentially and drive the processing pipeline via callbacks
  - Display the rendered frame in a resizable window
  - Handle keyboard shortcuts: q → quit, space → pause / resume
  - Catch and log per-frame errors without interrupting the stream

Part of a three-component pipeline:
  FrameProcessor  — single-frame processing
  VideoLoop       — video I/O and display
  Analyzer        — top-level orchestration
"""

import traceback
from typing import Callable, Iterator

import cv2
import numpy as np

WIN_NAME = "Smart Traffic Analyzer"


def _setup_cv2_window(win_w: int, win_h: int) -> None:
    """Center the OpenCV window on screen and bring it to the foreground.

    Called once after the first imshow.  Uses ctypes so no extra dependencies
    are needed beyond the standard Windows API.

    Args:
        win_w: Displayed frame width in pixels.
        win_h: Displayed frame height in pixels.
    """
    try:
        import ctypes
        user32 = ctypes.windll.user32
        scr_w  = user32.GetSystemMetrics(0)   # SM_CXSCREEN
        scr_h  = user32.GetSystemMetrics(1)   # SM_CYSCREEN
        x = max((scr_w - win_w) // 2, 0)
        y = max((scr_h - win_h) // 2, 0)
        cv2.moveWindow(WIN_NAME, x, y)
        hwnd = user32.FindWindowW(None, WIN_NAME)
        if hwnd:
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
    except Exception:
        pass  # Non-Windows or permission denied — silently skip


_MJPEG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer":    "https://izum.izmir.bel.tr/",
    "Accept":     "multipart/x-mixed-replace,image/jpeg,*/*",
    "Connection": "keep-alive",
}

_MJPEG_MAX_RETRIES = 10
_MJPEG_RETRY_DELAY = 2   # seconds between reconnect attempts


def _iter_mjpeg(url: str) -> Iterator[np.ndarray]:
    """
    Read an MJPEG-over-HTTP(S) stream and yield decoded BGR frames.
    - Locates JPEG frames via SOI/EOI byte markers (works with any boundary format).
    - Automatically reconnects up to _MJPEG_MAX_RETRIES times on network errors.
    - Sends browser-like headers so servers that check Referer/User-Agent accept us.
    """
    import time
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    attempt = 0
    while attempt < _MJPEG_MAX_RETRIES:
        try:
            with requests.get(url, stream=True, verify=False,
                              timeout=15, headers=_MJPEG_HEADERS) as r:
                r.raise_for_status()
                ct = r.headers.get("Content-Type", "unknown")
                print(f"[VideoLoop] Content-Type: {ct}")

                buf        = b""
                frame_count = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    buf += chunk
                    while True:
                        start = buf.find(b"\xff\xd8")   # JPEG SOI
                        end   = buf.find(b"\xff\xd9")   # JPEG EOI
                        if start == -1 or end == -1 or end < start:
                            break
                        jpg   = buf[start:end + 2]
                        buf   = buf[end + 2:]
                        frame = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        if frame is not None:
                            frame_count += 1
                            attempt = 0   # reset retry counter on success
                            yield frame

                # Stream ended cleanly — try to reconnect
                print(f"[VideoLoop] Stream ended after {frame_count} frames. Reconnecting...")

        except requests.exceptions.RequestException as e:
            attempt += 1
            print(f"[VideoLoop] Stream error (attempt {attempt}/{_MJPEG_MAX_RETRIES}): {e}")

        time.sleep(_MJPEG_RETRY_DELAY)

    print(f"[VideoLoop] Max retries reached. Giving up.")


class VideoLoop:
    """
    Reads frames from a video file, camera stream, or MJPEG URL and drives
    the processing pipeline via callbacks.

    Args:
        video_path:    Path to the video file, camera index, or stream URL.
        display_width: Width (px) to which the display window is resized.
    """

    def __init__(self, video_path: str, display_width: int):
        self._video_path    = video_path
        self._display_width = display_width

    def run(self,
            frame_callback: Callable,
            on_events: Callable) -> None:
        """
        Open the video, loop over frames, call callbacks, handle UI.

        Args:
            frame_callback: Called as frame_callback(frame, frame_id) → list[dict].
                            Allowed to modify frame in-place (render overlays).
            on_events:      Called as on_events(events) after each frame.
                            Responsible for publishing / storing events.
        """
        is_mjpeg = "mjpeg" in self._video_path.lower()

        if is_mjpeg:
            self._run_mjpeg(frame_callback, on_events)
        else:
            self._run_cv2(frame_callback, on_events)

    # ── MJPEG path ────────────────────────────────────────────────────────────

    def _run_mjpeg(self, frame_callback: Callable, on_events: Callable) -> None:
        print(f"[VideoLoop] MJPEG stream: {self._video_path}")
        print("[VideoLoop] q=quit  |  space=pause/resume")

        frame_count  = 0
        paused       = False
        dw           = self._display_width
        window_ready = False

        try:
            for frame in _iter_mjpeg(self._video_path):
                if paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        paused = False
                        print("[VideoLoop] RESUMED")
                    continue

                frame_count += 1
                try:
                    events = frame_callback(frame, frame_count)
                    on_events(events)
                except Exception as e:
                    print(f"[VideoLoop] ERROR frame {frame_count}: {e}")
                    traceback.print_exc()

                h, w   = frame.shape[:2]
                win_h  = int(dw * h / w)
                cv2.imshow(WIN_NAME, cv2.resize(frame, (dw, win_h)))

                if not window_ready:
                    _setup_cv2_window(dw, win_h)
                    window_ready = True

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = True
                    print("[VideoLoop] PAUSED")

        except Exception as e:
            print(f"[VideoLoop] Stream error: {e}")
        finally:
            cv2.destroyAllWindows()
            print("[VideoLoop] Stream closed.")

    # ── OpenCV / file path ────────────────────────────────────────────────────

    def _run_cv2(self, frame_callback: Callable, on_events: Callable) -> None:
        is_http = self._video_path.startswith("http")
        if is_http:
            cap = cv2.VideoCapture(self._video_path, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(self._video_path)

        if not cap.isOpened():
            print(f"[VideoLoop] ERROR — Cannot open: {self._video_path}")
            return

        print(f"[VideoLoop] Processing: {self._video_path}")
        print("[VideoLoop] q=quit  |  space=pause/resume")

        frame_count  = 0
        paused       = False
        dw           = self._display_width
        window_ready = False

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    try:
                        events = frame_callback(frame, frame_count)
                        on_events(events)
                    except Exception as e:
                        print(f"[VideoLoop] ERROR frame {frame_count}: {e}")
                        traceback.print_exc()

                    h, w  = frame.shape[:2]
                    win_h = int(dw * h / w)
                    cv2.imshow(WIN_NAME, cv2.resize(frame, (dw, win_h)))

                    if not window_ready:
                        _setup_cv2_window(dw, win_h)
                        window_ready = True

                wait_ms = 100 if paused else 1
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"[VideoLoop] {'PAUSED' if paused else 'RESUMED'}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[VideoLoop] Stream closed.")
