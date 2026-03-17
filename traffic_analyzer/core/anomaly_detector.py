"""
core/anomaly_detector.py

Detects per-vehicle anomalies (stopped vehicle, sudden slowdown).
Delegates all kinematic checks to VehicleMetrics.
"""

from dataclasses import dataclass
from typing import Optional

from traffic_analyzer.core.vehicle_metrics import VehicleMetrics


@dataclass
class AnomalyResult:
    """Result of a single-vehicle anomaly check."""
    is_anomaly: bool
    type: Optional[str]   # "stopped_vehicle" | "sudden_slowdown" | None
    stop_seconds: float


class AnomalyDetector:
    """
    Evaluates anomaly conditions for a given track.
    Stopped vehicle takes priority over sudden slowdown.
    """

    def __init__(self, vm: VehicleMetrics):
        self._vm = vm

    def detect(self, tid: int) -> AnomalyResult:
        """
        Return an AnomalyResult for track `tid`.
        Only one anomaly type is reported at a time — stopped takes priority.
        """
        stopped = self._vm.is_stopped(tid)
        if stopped:
            return AnomalyResult(
                is_anomaly=True,
                type="stopped_vehicle",
                stop_seconds=self._vm.get_stop_duration(tid),
            )

        if self._vm.is_sudden_slowdown(tid):
            return AnomalyResult(
                is_anomaly=True,
                type="sudden_slowdown",
                stop_seconds=0.0,
            )

        return AnomalyResult(is_anomaly=False, type=None, stop_seconds=0.0)