"""
core/anomaly_detector.py

Evaluates anomaly conditions for individual vehicle tracks.

Detects three anomaly types in priority order:
  1. stopped_vehicle  — vehicle stationary beyond the configured threshold
  2. wrong_way        — vehicle moving in the direction opposite to lane expectation
  3. sudden_slowdown  — vehicle speed drops sharply within a short window

Kinematic state is delegated entirely to VehicleMetrics; this class only
applies the decision logic and returns a typed AnomalyResult value object.
"""

from traffic_analyzer.core.vehicle_metrics import VehicleMetrics
from traffic_analyzer.domain.models import AnomalyResult, AnomalyType

# Maps each direction string to its exact opposite.
# Wrong-way is only flagged when the vehicle moves in the exact opposite
# direction.  Lateral movement (e.g. left/right on an up/down lane) is
# not penalised.
OPPOSITE_DIRECTION = {
    "bottom_to_top": "top_to_bottom",
    "top_to_bottom": "bottom_to_top",
    "left_to_right": "right_to_left",
    "right_to_left": "left_to_right",
}


class AnomalyDetector:
    """
    Evaluates anomaly conditions for a single track.
    Only one anomaly type is reported per call — see priority order above.
    """

    def __init__(self, vm: VehicleMetrics):
        self._vm = vm

    def detect(self, tid: int, direction: str = "",
               expected_direction: str = "") -> AnomalyResult:
        """
        Return an AnomalyResult for track `tid`.

        Args:
            tid:                Track identifier.
            direction:          Current movement direction string from VehicleMetrics.
            expected_direction: Lane's configured expected direction (may be empty).
        """
        if self._vm.is_stopped(tid):
            return AnomalyResult(
                is_anomaly=True,
                type=AnomalyType.STOPPED_VEHICLE,
                stop_seconds=self._vm.get_stop_duration(tid),
            )

        if (expected_direction
                and direction not in ("stopped", "unknown", "")
                and direction == OPPOSITE_DIRECTION.get(expected_direction)):
            return AnomalyResult(
                is_anomaly=True,
                type=AnomalyType.WRONG_WAY,
                stop_seconds=0.0,
            )

        if self._vm.is_sudden_slowdown(tid):
            return AnomalyResult(
                is_anomaly=True,
                type=AnomalyType.SUDDEN_SLOWDOWN,
                stop_seconds=0.0,
            )

        return AnomalyResult.none()
