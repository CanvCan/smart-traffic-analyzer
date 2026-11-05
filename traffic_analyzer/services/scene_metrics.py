"""
services/scene_metrics.py

Scene-level traffic metrics: lane assignment, occupancy, and status classification.
Supports dynamic scene thresholds injected from `config.json` (`scene_thresholds`).
"""

import cv2
import numpy as np
from typing import Any, List, Optional, Sequence, Tuple

from traffic_analyzer.domain.models import LanePolygon

SPEED_FREE = 65.0
SPEED_FLOW = 40.0
SPEED_HEAVY = 25.0
SPEED_JAMMED = 15.0

OCC_FREE = 0.20
OCC_FLOW = 0.35
OCC_HEAVY = 0.55
OCC_JAMMED = 0.75

COUNT_FREE = 10
COUNT_FLOW = 18
COUNT_HEAVY = 25
COUNT_JAMMED = 35

SNAPSHOT_EVERY = 48


class TrafficMetrics:
    """Stateless scene-level metric calculations."""

    @staticmethod
    def get_lane(cx: float, cy: float, lanes: Sequence[LanePolygon]) -> Optional[str]:
        for lane in lanes:
            if not getattr(lane, "visible", True):
                continue
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
        if not boxes or roi_area <= 0:
            return 0.0

        ys = set()
        for x1, y1, x2, y2 in boxes:
            ys.add(y1)
            ys.add(y2)
        ys = sorted(ys)

        union_area = 0.0
        for i in range(len(ys) - 1):
            slab_y1 = ys[i]
            slab_y2 = ys[i + 1]
            slab_h = slab_y2 - slab_y1
            if slab_h <= 0:
                continue

            x_intervals = []
            for x1, y1, x2, y2 in boxes:
                if y1 <= slab_y1 and y2 >= slab_y2:
                    x_intervals.append((x1, x2))

            if not x_intervals:
                continue

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
    def traffic_status(
        avg_speed: float,
        vehicle_count: int,
        occupancy: float,
        thresholds: Optional[Any] = None,
    ) -> str:
        if vehicle_count == 0:
            return "FREE"

        spd_free = float(getattr(thresholds, "speed_free", SPEED_FREE)) if thresholds else SPEED_FREE
        spd_flow = float(getattr(thresholds, "speed_flow", SPEED_FLOW)) if thresholds else SPEED_FLOW
        spd_heavy = float(getattr(thresholds, "speed_heavy", SPEED_HEAVY)) if thresholds else SPEED_HEAVY
        spd_jammed = float(getattr(thresholds, "speed_jammed", SPEED_JAMMED)) if thresholds else SPEED_JAMMED

        occ_free = float(getattr(thresholds, "occ_free", OCC_FREE)) if thresholds else OCC_FREE
        occ_jammed = float(getattr(thresholds, "occ_jammed", OCC_JAMMED)) if thresholds else OCC_JAMMED
        occ_span = max(0.05, occ_jammed - occ_free)
        occ_flow = float(getattr(thresholds, "occ_flow", occ_free + occ_span * 0.27)) if thresholds else OCC_FLOW
        occ_heavy = float(getattr(thresholds, "occ_heavy", occ_free + occ_span * 0.64)) if thresholds else OCC_HEAVY

        speed_score = (
            0 if avg_speed >= spd_free else
            1 if avg_speed >= spd_flow else
            2 if avg_speed >= spd_heavy else
            3 if avg_speed >= spd_jammed else
            4
        )

        density_score = (
            0 if occupancy < occ_free else
            1 if occupancy < occ_flow else
            2 if occupancy < occ_heavy else
            3 if occupancy < occ_jammed else
            4
        )

        count_score = (
            0 if vehicle_count <= COUNT_FREE else
            1 if vehicle_count <= COUNT_FLOW else
            2 if vehicle_count <= COUNT_HEAVY else
            3 if vehicle_count <= COUNT_JAMMED else
            4
        )

        total = speed_score * 3 + density_score * 2 + count_score * 1

        if total <= 4:
            return "FREE"
        elif total <= 9:
            return "FLOW"
        elif total <= 16:
            return "HEAVY"
        else:
            return "JAMMED"

    @staticmethod
    def roi_total_area(lanes: Sequence[LanePolygon]) -> float:
        total = 0.0
        for lane in lanes:
            if not getattr(lane, "visible", True):
                continue
            if lane.points and len(lane.points) >= 3:
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
