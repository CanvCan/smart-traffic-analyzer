"""
core/metrics.py

Per-vehicle and scene-level traffic metrics.

Design decisions:
- Speed    : EMA-smoothed, perspective-compensated px/s
- Direction: Linear regression over last N positions (noise-resistant)
- Stopped  : Positional std-dev over a time window (camera-shake resistant)
- Slowdown : Two-window EMA comparison with low-speed guard
- Occupancy: Merged-interval area to avoid double-counting overlapping boxes
- Status   : Weighted tri-factor (speed + occupancy + count)
"""

import time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from traffic_analyzer.utils.config_loader import LaneConfig

# ── TUNABLE CONSTANTS ────────────────────────────────────────────────────────

EMA_ALPHA = 0.25  # EMA smoothing factor for speed (0=heavy smooth, 1=raw)
POSITION_HISTORY = 30  # Max position history per track
SPEED_HISTORY = 30  # Max speed history per track
DIRECTION_WINDOW = 10  # Positions used for direction regression
STOP_WINDOW = 30  # Frames to analyse for stopped detection
STOP_STD_THRESHOLD = 3.0  # Max positional std-dev (px) to be "stopped"
STOP_MIN_FRAMES = 40  # Must be in stop window for this many frames
SLOWDOWN_PAST_WIN = 15  # Past window for slowdown comparison
SLOWDOWN_NOW_WIN = 5  # Current window for slowdown comparison
SLOWDOWN_RATIO = 0.50  # Speed must drop by this fraction to flag anomaly
SLOWDOWN_MIN_SPEED = 30.0  # px/s — below this, slowdown is expected (stop-go)

# Traffic status thresholds (px/s — calibrate per camera after deployment)
# Typical highway camera at 0.5 scale: free-flow ~120-200 px/s, slow ~40 px/s
SPEED_FREE = 120.0  # above this → no speed penalty
SPEED_FLOW = 70.0  # above this → mild penalty
SPEED_HEAVY = 25.0  # below this → heavy penalty

# Occupancy thresholds (fraction of lane area covered by vehicle boxes)
OCC_FREE = 0.25  # below → no density penalty
OCC_FLOW = 0.40  # below → mild penalty
OCC_HEAVY = 0.65  # above → heavy penalty

# Count thresholds (vehicles visible in lane at once)
COUNT_FREE = 7
COUNT_FLOW = 15
COUNT_HEAVY = 26

# Snapshot interval
SNAPSHOT_EVERY = 48  # frames


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
        """
        if not self.is_stopped(tid):
            return 0.0
        pos = list(self._pos[tid])
        if len(pos) < 2:
            return 0.0
        return round(pos[-1][2] - pos[-STOP_MIN_FRAMES][2], 2)

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
                del self._speed[tid]
                self._ema.pop(tid, None)
                self._entry_frame.pop(tid, None)
                self._entry_time.pop(tid, None)


# ── SCENE-LEVEL METRICS ──────────────────────────────────────────────────────

class TrafficMetrics:
    """
    Stateless scene-level metric calculations.
    All methods are static — no per-instance state needed.
    """

    @staticmethod
    def get_lane(cx: float, cy: float, lanes: List[LaneConfig]) -> Optional[str]:
        """
        Point-in-polygon test when polygon points are available,
        falls back to point-in-rectangle using roi bounding box.
        Returns the name of the matching lane or None.
        """
        import numpy as np
        for lane in lanes:
            if lane.points and len(lane.points) >= 3:
                # Ray-casting / cv2.pointPolygonTest
                import cv2 as _cv2
                pts = np.array(lane.points, dtype=np.float32)
                result = _cv2.pointPolygonTest(pts, (float(cx), float(cy)), False)
                if result >= 0:
                    return lane.name
            else:
                x1, y1, x2, y2 = lane.roi
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    return lane.name
        return None

    @staticmethod
    def occupancy_ratio(boxes: List[Tuple[int, int, int, int]],
                        roi_area: float) -> float:
        """
        True union area of bounding boxes via sweep-line algorithm.

        Avoids double-counting when vehicles overlap:
          1. Build Y-axis events (enter/leave) for each box
          2. Sweep top-to-bottom; at each Y slab, merge active X-intervals
          3. Accumulate slab_height * merged_x_length

        O(n^2) in worst case but n < 50 in practice — fast enough per frame.
        """
        if not boxes or roi_area <= 0:
            return 0.0

        # Collect unique y-boundaries and sort them
        ys = set()
        for x1, y1, x2, y2 in boxes:
            ys.add(y1);
            ys.add(y2)
        ys = sorted(ys)

        union_area = 0.0
        for i in range(len(ys) - 1):
            slab_y1 = ys[i]
            slab_y2 = ys[i + 1]
            slab_h = slab_y2 - slab_y1
            if slab_h <= 0:
                continue

            # Collect x-intervals of all boxes that cover this slab
            x_intervals = []
            for x1, y1, x2, y2 in boxes:
                if y1 <= slab_y1 and y2 >= slab_y2:
                    x_intervals.append((x1, x2))

            if not x_intervals:
                continue

            # Merge overlapping x-intervals
            x_intervals.sort()
            merged_x = 0
            cur_x1, cur_x2 = x_intervals[0]
            for nx1, nx2 in x_intervals[1:]:
                if nx1 <= cur_x2:
                    cur_x2 = max(cur_x2, nx2)
                else:
                    merged_x += cur_x2 - cur_x1
                    cur_x1, cur_x2 = nx1, nx2
            merged_x += cur_x2 - cur_x1

            union_area += slab_h * merged_x

        return round(min(union_area / roi_area, 1.0), 4)

    @staticmethod
    def traffic_status(avg_speed: float, vehicle_count: int,
                       occupancy: float) -> str:
        """
        Weighted tri-factor traffic classification.

        Scoring (0=good, 1=moderate, 2=bad):
          speed_score   — weight 3  (most reliable signal)
          density_score — weight 2  (true union area, no double-count)
          count_score   — weight 1  (least reliable, varies by lane length)

        Max score = 12 → bands calibrated so normal highway flow ≈ FREE/FLOW.

        Calibration guide:
          - Record 30s of free-flow → note avg_speed → set SPEED_FREE ~= that
          - Record congestion → note occupancy → adjust OCC_HEAVY
        """
        if vehicle_count == 0:
            return "FREE"

        # Speed score (weight 3)
        speed_score = (
            0 if avg_speed >= SPEED_FREE else
            1 if avg_speed >= SPEED_FLOW else
            2 if avg_speed >= SPEED_HEAVY else
            3
        )

        # Occupancy score (weight 2)
        density_score = (
            0 if occupancy < OCC_FREE else
            1 if occupancy < OCC_FLOW else
            2 if occupancy < OCC_HEAVY else
            3
        )

        # Count score (weight 1)
        count_score = (
            0 if vehicle_count <= COUNT_FREE else
            1 if vehicle_count <= COUNT_FLOW else
            2 if vehicle_count <= COUNT_HEAVY else
            3
        )

        total = speed_score * 3 + density_score * 2 + count_score * 1
        # Max possible = 3*3 + 3*2 + 3*1 = 18

        if total <= 2:
            return "FREE"
        elif total <= 6:
            return "FLOW"
        elif total <= 11:
            return "HEAVY"
        else:
            return "JAMMED"

    @staticmethod
    def roi_total_area(lanes: List[LaneConfig]) -> float:
        """
        Sum of lane areas. Uses shoelace formula for polygons,
        falls back to bounding-box area when no polygon points exist.
        """
        total = 0.0
        for lane in lanes:
            if lane.points and len(lane.points) >= 3:
                # Shoelace formula for polygon area
                pts = lane.points
                n = len(pts)
                area = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    area += pts[i][0] * pts[j][1]
                    area -= pts[j][0] * pts[i][1]
                total += abs(area) / 2.0
            else:
                x1, y1, x2, y2 = lane.roi
                total += (x2 - x1) * (y2 - y1)
        return total
