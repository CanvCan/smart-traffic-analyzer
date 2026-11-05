"""
services/vehicle_metrics.py

Per-vehicle kinematic metrics: speed, direction, stopped detection, slowdown,
and class majority voting.

Includes:
- Hybrid injected timestamp support (live monotonic clock vs. deterministic file PTS)
- Continuous-time speed EMA: alpha_v(dt) = 1 - exp(-dt / SPEED_TAU_SEC)
- Continuous-time bounding-box height EMA: alpha_h(dt) = 1 - exp(-dt / HEIGHT_TAU_SEC)
- Physical-time closed-form O(N) linear regression for direction estimation
- Frame-rate independent stopped vehicle detection and duration tracking
"""

import math
import numpy as np
from collections import Counter, defaultdict, deque
from typing import Dict, Optional, Tuple

SPEED_TAU_SEC = 0.35
HEIGHT_TAU_SEC = 0.50
POSITION_HISTORY = 45
SPEED_HISTORY = 30
DIRECTION_WINDOW = 10
MIN_MOVING_SLOPE_PX_SEC = 8.0
STOP_WINDOW = 30
STOP_STD_THRESHOLD = 3.0
STOP_MIN_OBSERVATIONS = 8
STOP_MIN_ELAPSED_SEC = 1.0
SLOWDOWN_PAST_WIN = 15
SLOWDOWN_NOW_WIN = 5
SLOWDOWN_RATIO = 0.50
SLOWDOWN_MIN_SPEED = 30.0
CLASS_VOTE_WINDOW = 20


class VehicleMetrics:
    """
    Maintains per-track kinematic state and computes metrics on demand.
    Instantiated once and shared across all frames.
    Reads parameters dynamically from `kinematics_settings` and `anomaly_settings`.
    """

    def __init__(
        self,
        speed_tau_sec: float = SPEED_TAU_SEC,
        position_history: int = POSITION_HISTORY,
        direction_window: int = DIRECTION_WINDOW,
        class_vote_window: int = CLASS_VOTE_WINDOW,
        stop_std_threshold: float = STOP_STD_THRESHOLD,
        stop_min_frames: int = STOP_WINDOW,
        stop_min_elapsed_sec: float = STOP_MIN_ELAPSED_SEC,
        slowdown_ratio: float = SLOWDOWN_RATIO,
        slowdown_min_speed: float = SLOWDOWN_MIN_SPEED,
    ) -> None:
        self._speed_tau_sec = max(0.05, float(speed_tau_sec))

        self._position_history = max(10, int(position_history))
        self._direction_window = max(3, int(direction_window))
        self._class_vote_window = max(1, int(class_vote_window))
        self._stop_window = max(STOP_MIN_OBSERVATIONS, int(stop_min_frames))
        self._stop_std_threshold = max(0.5, float(stop_std_threshold))
        self._stop_min_elapsed_sec = max(0.1, float(stop_min_elapsed_sec))
        self._slowdown_ratio = min(0.95, max(0.05, float(slowdown_ratio)))
        self._slowdown_min_speed = max(1.0, float(slowdown_min_speed))

        self._pos: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self._position_history))
        self._speed: Dict[int, deque] = defaultdict(lambda: deque(maxlen=SPEED_HISTORY))
        self._ema: Dict[int, float] = {}
        self._bh_ema: Dict[int, float] = {}
        self._entry_frame: Dict[int, int] = {}
        self._entry_time: Dict[int, float] = {}
        self._last_time: Dict[int, float] = {}
        self._stop_since: Dict[int, float] = {}
        self._class_votes: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self._class_vote_window))

    def update(
        self,
        tid: int,
        box: Tuple[int, int, int, int],
        frame_id: int,
        frame_height: int,
        cls_id: int = -1,
        frame_timestamp: Optional[float] = None,
    ) -> None:
        """
        Record new observation for track `tid` using injected `frame_timestamp`.
        """
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        raw_bh = float(max(y2 - y1, 1))
        now = float(frame_timestamp) if frame_timestamp is not None else (float(frame_id) / 25.0)
        self._last_time[tid] = now

        if tid not in self._entry_frame:
            self._entry_frame[tid] = frame_id
            self._entry_time[tid] = now
            self._bh_ema[tid] = raw_bh

        pos = self._pos[tid]

        if pos:
            prev_cx, prev_cy, prev_t = pos[-1]
            dt = now - prev_t

            if dt > 1e-4:
                dt_clamped = float(np.clip(dt, 0.001, 0.5))
                alpha_h = 1.0 - math.exp(-dt_clamped / HEIGHT_TAU_SEC)
                prev_bh = self._bh_ema.get(tid, raw_bh)
                smooth_bh = alpha_h * raw_bh + (1.0 - alpha_h) * prev_bh
                self._bh_ema[tid] = smooth_bh

                raw_speed = math.hypot(cx - prev_cx, cy - prev_cy) / dt

                ref_height = frame_height * 0.15
                scale = ref_height / max(smooth_bh, 1.0)
                scale = float(np.clip(scale, 0.5, 4.0))
                comp_speed = raw_speed * scale

                alpha_v = 1.0 - math.exp(-dt_clamped / self._speed_tau_sec)
                prev_ema = self._ema.get(tid, comp_speed)
                new_ema = alpha_v * comp_speed + (1.0 - alpha_v) * prev_ema
                self._ema[tid] = new_ema
                self._speed[tid].append(new_ema)
        else:
            self._bh_ema[tid] = raw_bh

        pos.append((cx, cy, now))

        if cls_id >= 0:
            self._class_votes[tid].append(cls_id)

        if self.is_stopped(tid):
            if tid not in self._stop_since:
                self._stop_since[tid] = now
        else:
            self._stop_since.pop(tid, None)

    def get_speed(self, tid: int) -> float:
        return float(round(self._ema.get(tid, 0.0), 2))

    def get_stable_class(self, tid: int, raw_cls_id: int) -> int:
        votes = self._class_votes.get(tid)
        if not votes:
            return raw_cls_id
        return Counter(votes).most_common(1)[0][0]

    def get_direction(self, tid: int) -> str:
        pos = list(self._pos[tid])
        if len(pos) < 3:
            return "unknown"

        recent = pos[-self._direction_window:]
        n = len(recent)
        t0 = recent[0][2]
        taus = [p[2] - t0 for p in recent]
        if taus[-1] <= 1e-4:
            taus = [i / 25.0 for i in range(n)]

        mean_t = sum(taus) / n
        mean_x = sum(p[0] for p in recent) / n
        mean_y = sum(p[1] for p in recent) / n

        denom = sum((t - mean_t) ** 2 for t in taus)
        if denom <= 1e-9:
            return "stopped"

        vx = sum((taus[i] - mean_t) * (recent[i][0] - mean_x) for i in range(n)) / denom
        vy = sum((taus[i] - mean_t) * (recent[i][1] - mean_y) for i in range(n)) / denom

        mag = math.hypot(vx, vy)
        if mag < MIN_MOVING_SLOPE_PX_SEC:
            return "stopped"

        if abs(vx) >= abs(vy):
            return "left_to_right" if vx > 0 else "right_to_left"
        else:
            return "top_to_bottom" if vy > 0 else "bottom_to_top"

    def is_stopped(self, tid: int) -> bool:
        pos = list(self._pos[tid])
        recent = pos[-self._stop_window:]
        if len(recent) < STOP_MIN_OBSERVATIONS:
            return False

        dt_window = recent[-1][2] - recent[0][2]
        if dt_window < self._stop_min_elapsed_sec:
            return False

        xs = np.array([p[0] for p in recent])
        ys = np.array([p[1] for p in recent])
        std = float(np.sqrt(np.var(xs) + np.var(ys)))
        return std < self._stop_std_threshold

    def get_stop_duration(self, tid: int) -> float:
        if not self.is_stopped(tid):
            return 0.0
        stop_since = self._stop_since.get(tid)
        if stop_since is None:
            return 0.0
        now = self._last_time.get(tid, stop_since)
        return round(max(0.0, now - stop_since), 2)

    def is_sudden_slowdown(self, tid: int) -> bool:
        spd = list(self._speed[tid])
        needed = SLOWDOWN_PAST_WIN + SLOWDOWN_NOW_WIN
        if len(spd) < needed:
            return False

        past_mean = float(np.mean(spd[-needed:-SLOWDOWN_NOW_WIN]))
        curr_mean = float(np.mean(spd[-SLOWDOWN_NOW_WIN:]))

        if past_mean < self._slowdown_min_speed:
            return False
        if self.is_stopped(tid):
            return False

        return curr_mean < past_mean * (1 - self._slowdown_ratio)

    def get_residence(self, tid: int, frame_id: int) -> Dict:
        entry_frame = self._entry_frame.get(tid, frame_id)
        now = self._last_time.get(tid, 0.0)
        entry_time = self._entry_time.get(tid, now)
        return {
            "entry_frame": entry_frame,
            "frames_in_roi": frame_id - entry_frame,
            "seconds_in_roi": round(max(0.0, now - entry_time), 2),
        }

    def cleanup(self, active_ids: set) -> None:
        for tid in list(self._pos.keys()):
            if tid not in active_ids:
                del self._pos[tid]
                self._speed.pop(tid, None)
                self._ema.pop(tid, None)
                self._bh_ema.pop(tid, None)
                self._entry_frame.pop(tid, None)
                self._entry_time.pop(tid, None)
                self._last_time.pop(tid, None)
                self._stop_since.pop(tid, None)
                self._class_votes.pop(tid, None)
