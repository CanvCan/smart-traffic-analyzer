"""
core/scene_metrics.py

Scene-level traffic metrics: lane assignment, occupancy, and status classification.

Design decisions:
- Occupancy: Merged-interval area to avoid double-counting overlapping boxes
- Status   : Weighted tri-factor (speed + occupancy + count)
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from traffic_analyzer.utils.config_loader import LaneConfig

# ── Traffic status thresholds (px/s — calibrate per camera after deployment) ──
# Typical highway camera at 0.5 scale: free-flow ~120-200 px/s, slow ~40 px/s
SPEED_FREE = 65.0  # above this → no speed penalty
SPEED_FLOW = 40.0  # above this → mild penalty
SPEED_HEAVY = 25.0  # below this → heavy penalty
SPEED_JAMMED = 15.0

# Occupancy thresholds (fraction of lane area covered by vehicle boxes)
OCC_FREE = 0.20  # below → no density penalty
OCC_FLOW = 0.35  # below → mild penalty
OCC_HEAVY = 0.55  # above → heavy penalty
OCC_JAMMED = 0.75

# Count thresholds (vehicles visible in lane at once)
COUNT_FREE = 10
COUNT_FLOW = 18
COUNT_HEAVY = 25
COUNT_JAMMED = 35

# Snapshot interval
SNAPSHOT_EVERY = 48  # frames


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
        for lane in lanes:
            if lane.points and len(lane.points) >= 3:
                pts = np.array(lane.points, dtype=np.float32)
                result = cv2.pointPolygonTest(pts, (float(cx), float(cy)), False)
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

        Scoring (0=best, 4=worst):
          speed_score   — weight 3  (most reliable signal)
          density_score — weight 2  (true union area, no double-count)
          count_score   — weight 1  (least reliable, varies by lane length)

        Max score = 24 → bands calibrated so normal highway flow ≈ FREE/FLOW.

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
            3 if avg_speed >= SPEED_JAMMED else
            4
        )

        # Occupancy score (weight 2)
        density_score = (
            0 if occupancy < OCC_FREE else
            1 if occupancy < OCC_FLOW else
            2 if occupancy < OCC_HEAVY else
            3 if occupancy < OCC_JAMMED else
            4
        )

        # Count score (weight 1)
        count_score = (
            0 if vehicle_count <= COUNT_FREE else
            1 if vehicle_count <= COUNT_FLOW else
            2 if vehicle_count <= COUNT_HEAVY else
            3 if vehicle_count <= COUNT_JAMMED else
            4
        )

        total = speed_score * 3 + density_score * 2 + count_score * 1
        # Max possible = 4*3 + 4*2 + 4*1 = 24  →  FREE ≤ 4, FLOW ≤ 9, HEAVY ≤ 16, else JAMMED

        if total <= 4:
            return "FREE"
        elif total <= 9:
            return "FLOW"
        elif total <= 16:
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