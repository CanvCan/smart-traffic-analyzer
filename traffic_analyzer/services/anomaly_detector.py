"""
services/anomaly_detector.py

Evaluates anomaly conditions for individual vehicle tracks.
"""

from traffic_analyzer.services.vehicle_metrics import VehicleMetrics
from traffic_analyzer.domain.models import AnomalyResult, AnomalyType

OPPOSITE_DIRECTION = {
    "bottom_to_top": "top_to_bottom",
    "top_to_bottom": "bottom_to_top",
    "left_to_right": "right_to_left",
    "right_to_left": "left_to_right",
}


class AnomalyDetector:
    """Evaluates anomaly conditions for a single track."""

    def __init__(self, vm: VehicleMetrics):
        self._vm = vm

    def detect(self, tid: int, direction: str = "",
               expected_direction: str = "") -> AnomalyResult:
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
