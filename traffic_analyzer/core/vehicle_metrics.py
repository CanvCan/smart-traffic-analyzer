"""
core/vehicle_metrics.py

Per-vehicle kinematic metrics: speed, direction, stopped detection, slowdown.

Design decisions:
- Speed    : EMA-smoothed, perspective-compensated px/s
- Direction: Linear regression over last N positions (noise-resistant)
- Stopped  : Positional std-dev over a time window (camera-shake resistant)
- Slowdown : Two-window EMA comparison with low-speed guard
"""

import time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

# ── TUNABLE CONSTANTS ────────────────────────────────────────────────────────

EMA_ALPHA = 0.25  # EMA smoothing factor for speed (0=heavy smooth, 1=raw)
POSITION_HISTORY = 45  # Max position history per track (must be >= STOP_MIN_FRAMES)
SPEED_HISTORY = 30  # Max speed history per track
DIRECTION_WINDOW = 10  # Positions used for direction regression
STOP_WINDOW = 30  # Frames to analyse for stopped detection
STOP_STD_THRESHOLD = 3.0  # Max positional std-dev (px) to be "stopped"
STOP_MIN_FRAMES = 30  # Must have at least this many frames before stopped check
# Note: must be <= POSITION_HISTORY and <= STOP_WINDOW
SLOWDOWN_PAST_WIN = 15  # Past window for slowdown comparison
SLOWDOWN_NOW_WIN = 5  # Current window for slowdown comparison
SLOWDOWN_RATIO = 0.50  # Speed must drop by this fraction to flag anomaly
SLOWDOWN_MIN_SPEED = 30.0  # px/s — below this, slowdown is expected (stop-go)


class VehicleMetrics:
    """
    Maintains per-track kinematic state and computes metrics on demand.
    Instantiated once and shared across all frames.
    """

    def __init__(self):
        # (cx, cy, wall_clock_time)
        self._pos: Dict[int, deque] = defaultdict(lambda: deque(maxlen=POSITION_HISTORY))
        # EMA-smoothed perspective-corrected speed in px/s
        self._speed: Dict[int, deque] = defaultdict(lambda: deque(maxlen=SPEED_HISTORY))
        # EMA state (last smoothed value)
        self._ema: Dict[int, float] = {}
        # Entry info
        self._entry_frame: Dict[int, int] = {}
        self._entry_time: Dict[int, float] = {}

    # ── UPDATE ───────────────────────────────────────────────────────────────

    def update(self, tid: int, box: Tuple[int, int, int, int],
               frame_id: int, frame_height: int) -> None:
        """
        Record new observation for track `tid`.
        Must be called exactly once per frame per active track.
        """
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bh = max(y2 - y1, 1)  # bounding-box height
        now = time.time()

        # Register entry
        if tid not in self._entry_frame:
            self._entry_frame[tid] = frame_id
            self._entry_time[tid] = now

        pos = self._pos[tid]

        if pos:
            prev_cx, prev_cy, prev_t = pos[-1]
            dt = now - prev_t

            if dt > 1e-4:  # guard against duplicate timestamps
                # Raw pixel displacement per second
                raw_speed = np.hypot(cx - prev_cx, cy - prev_cy) / dt

                # Perspective compensation:
                # Vehicles higher in the frame (smaller bh) are further away.
                # A vehicle with bh=ref_height is at "reference distance".
                # Scale factor = ref_height / actual_bh  (larger when far away)
                ref_height = frame_height * 0.15  # tunable reference bbox height
                scale = ref_height / bh
                scale = np.clip(scale, 0.5, 4.0)  # bound to avoid wild values
                comp_speed = raw_speed * scale

                # EMA smoothing
                prev_ema = self._ema.get(tid, comp_speed)
                new_ema = EMA_ALPHA * comp_speed + (1 - EMA_ALPHA) * prev_ema
                self._ema[tid] = new_ema
                self._speed[tid].append(new_ema)

        pos.append((cx, cy, now))

    # ── SPEED ────────────────────────────────────────────────────────────────

    def get_speed(self, tid: int) -> float:
        """
        Current EMA speed in px/s.
        Returns 0.0 if not enough history.
        """
        return round(self._ema.get(tid, 0.0), 2)

    # ── DIRECTION ────────────────────────────────────────────────────────────

    def get_direction(self, tid: int) -> str:
        """
        Estimate movement direction via linear regression over the last
        DIRECTION_WINDOW positions.

        Regression is more noise-resistant than a two-point delta:
        it uses all available positions and finds the best-fit slope.
        """
        pos = list(self._pos[tid])
        if len(pos) < 3:
            return "unknown"

        recent = pos[-DIRECTION_WINDOW:]
        xs = np.array([p[0] for p in recent])
        ys = np.array([p[1] for p in recent])
        ts = np.arange(len(recent), dtype=float)

        # Fit x(t) and y(t) independently
        dx = float(np.polyfit(ts, xs, 1)[0])  # slope of x over time
        dy = float(np.polyfit(ts, ys, 1)[0])  # slope of y over time

        mag = np.hypot(dx, dy)
        if mag < 0.5:  # effectively stationary
            return "stopped"

        # Return primary axis direction
        if abs(dx) >= abs(dy):
            return "left_to_right" if dx > 0 else "right_to_left"
        else:
            return "top_to_bottom" if dy > 0 else "bottom_to_top"

    # ── STOPPED DETECTION ────────────────────────────────────────────────────

    def is_stopped(self, tid: int) -> bool:
        """
        A vehicle is stopped when the standard deviation of its centroid
        positions over the last STOP_WINDOW frames is below STOP_STD_THRESHOLD.

        Using std-dev (not range) makes this robust to camera micro-vibrations
        which cause small random displacements but not a consistent drift.
        Requires at least STOP_MIN_FRAMES observations.
        """
        pos = list(self._pos[tid])
        if len(pos) < STOP_MIN_FRAMES:
            return False

        recent = pos[-STOP_WINDOW:]
        xs = np.array([p[0] for p in recent])
        ys = np.array([p[1] for p in recent])
        std = float(np.sqrt(np.var(xs) + np.var(ys)))
        return std < STOP_STD_THRESHOLD

    def get_stop_duration(self, tid: int) -> float:
        """
        Seconds since the vehicle was last moving.
        Returns 0.0 if not stopped.

        Uses the oldest available position capped to STOP_MIN_FRAMES
        so this is safe regardless of deque length.
        """
        if not self.is_stopped(tid):
            return 0.0
        pos = list(self._pos[tid])
        if len(pos) < 2:
            return 0.0
        lookback = min(STOP_MIN_FRAMES, len(pos))
        return round(pos[-1][2] - pos[-lookback][2], 2)

    # ── SUDDEN SLOWDOWN ──────────────────────────────────────────────────────

    def is_sudden_slowdown(self, tid: int) -> bool:
        """
        Compares the mean speed of the past window against the current window.

        past_mean  = mean(speed[-PAST_WIN - NOW_WIN : -NOW_WIN])
        curr_mean  = mean(speed[-NOW_WIN:])

        Flagged when:
          curr_mean < past_mean * (1 - SLOWDOWN_RATIO)
          AND past_mean > SLOWDOWN_MIN_SPEED   (not already in stop-go)
          AND NOT is_stopped()                 (stopped has its own flag)
        """
        spd = list(self._speed[tid])
        needed = SLOWDOWN_PAST_WIN + SLOWDOWN_NOW_WIN
        if len(spd) < needed:
            return False

        past_mean = float(np.mean(spd[-needed:-SLOWDOWN_NOW_WIN]))
        curr_mean = float(np.mean(spd[-SLOWDOWN_NOW_WIN:]))

        if past_mean < SLOWDOWN_MIN_SPEED:
            return False
        if self.is_stopped(tid):
            return False

        return curr_mean < past_mean * (1 - SLOWDOWN_RATIO)

    # ── RESIDENCE ────────────────────────────────────────────────────────────

    def get_residence(self, tid: int, frame_id: int) -> Dict:
        entry_frame = self._entry_frame.get(tid, frame_id)
        entry_time = self._entry_time.get(tid, time.time())
        return {
            "entry_frame": entry_frame,
            "frames_in_roi": frame_id - entry_frame,
            "seconds_in_roi": round(time.time() - entry_time, 2),
        }

    # ── CLEANUP ──────────────────────────────────────────────────────────────

    def cleanup(self, active_ids: set) -> None:
        """Remove state for tracks that are no longer active."""
        for tid in list(self._pos.keys()):
            if tid not in active_ids:
                del self._pos[tid]
                self._speed.pop(tid, None)
                self._ema.pop(tid, None)
                self._entry_frame.pop(tid, None)
                self._entry_time.pop(tid, None)