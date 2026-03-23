"""
core/anomaly_detector.py

Detects per-vehicle anomalies (stopped vehicle, wrong way, sudden slowdown).
Delegates all kinematic checks to VehicleMetrics.

Priority: stopped_vehicle > wrong_way > sudden_slowdown
"""

from dataclasses import dataclass
from typing import Optional

from traffic_analyzer.core.vehicle_metrics import VehicleMetrics

# Maps each direction to its exact opposite.
# Wrong-way is only flagged when the vehicle moves in the opposite direction —
# lateral movement (e.g. left/right when expected is up) is not penalised.
OPPOSITE_DIRECTION = {
    "bottom_to_top": "top_to_bottom",
    "top_to_bottom": "bottom_to_top",
    "left_to_right": "right_to_left",
    "right_to_left": "left_to_right",
}


@dataclass
class AnomalyResult:
    """Result of a single-vehicle anomaly check."""
    is_anomaly: bool
    type: Optional[str]   # "stopped_vehicle" | "wrong_way" | "sudden_slowdown" | None
    stop_seconds: float


class AnomalyDetector:
    """
    Evaluates anomaly conditions for a given track.
    Only one anomaly type is reported at a time — priority order above.
    """

    def __init__(self, vm: VehicleMetrics):
        self._vm = vm

    def detect(self, tid: int, direction: str = "",
               expected_direction: str = "") -> AnomalyResult:
        """
        Return an AnomalyResult for track `tid`.

        direction          : current movement direction from VehicleMetrics
        expected_direction : lane's configured expected direction (may be empty)
        """
        if self._vm.is_stopped(tid):
            return AnomalyResult(
                is_anomaly=True,
                type="stopped_vehicle",
                stop_seconds=self._vm.get_stop_duration(tid),
            )

        if (expected_direction
                and direction not in ("stopped", "unknown", "")
                and direction == OPPOSITE_DIRECTION.get(expected_direction)):
            return AnomalyResult(
                is_anomaly=True,
                type="wrong_way",
                stop_seconds=0.0,
            )

        if self._vm.is_sudden_slowdown(tid):
            return AnomalyResult(
                is_anomaly=True,
                type="sudden_slowdown",
                stop_seconds=0.0,
            )

        return AnomalyResult(is_anomaly=False, type=None, stop_seconds=0.0)