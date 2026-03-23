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
from typing import Callable

import cv2


class VideoLoop:
    """
    Reads frames from a video file or camera stream and drives
    the processing pipeline via callbacks.

    Args:
        video_path:    Path to the video file or camera index.
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
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            print(f"[VideoLoop] ERROR — Cannot open: {self._video_path}")
            return

        print(f"[VideoLoop] Processing: {self._video_path}")
        print("[VideoLoop] q → quit  |  space → pause/resume")

        frame_count = 0
        paused      = False
        dw          = self._display_width

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

                    h, w = frame.shape[:2]
                    cv2.imshow("Smart Traffic Analyzer",
                               cv2.resize(frame, (dw, int(dw * h / w))))

                # Longer wait when paused to avoid CPU spin
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
