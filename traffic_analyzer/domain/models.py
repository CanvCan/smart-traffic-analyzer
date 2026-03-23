"""
domain/models.py

Core domain value objects and enumerations.

This module has NO external dependencies — pure Python only.
All other layers may import from here; this module imports from nowhere.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class VehicleClass(Enum):
    """Domain representation of detectable vehicle types."""

    CAR        = 2
    MOTORCYCLE = 3
    BUS        = 5
    TRUCK      = 7
    UNKNOWN    = -1

    @property
    def label(self) -> str:
        """Human-readable display name used in events and UI."""
        return {
            VehicleClass.CAR:        "Car",
            VehicleClass.MOTORCYCLE: "Motorcycle",
            VehicleClass.BUS:        "Bus",
            VehicleClass.TRUCK:      "Truck",
            VehicleClass.UNKNOWN:    "Vehicle",
        }[self]

    @property
    def is_heavy(self) -> bool:
        """Bus and Truck are classified as heavy vehicles."""
        return self in (VehicleClass.BUS, VehicleClass.TRUCK)

    @classmethod
    def from_id(cls, class_id: int) -> "VehicleClass":
        """Convert a YOLO class integer to VehicleClass. Returns UNKNOWN on invalid id."""
        try:
            return cls(class_id)
        except ValueError:
            return cls.UNKNOWN


class Direction(str, Enum):
    """Vehicle movement direction as reported by VehicleMetrics."""

    BOTTOM_TO_TOP = "bottom_to_top"
    TOP_TO_BOTTOM = "top_to_bottom"
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    STOPPED       = "stopped"
    UNKNOWN       = "unknown"

    def opposite(self) -> Optional["Direction"]:
        """Return the exact opposite direction, or None for non-directional states."""
        _opposites = {
            Direction.BOTTOM_TO_TOP: Direction.TOP_TO_BOTTOM,
            Direction.TOP_TO_BOTTOM: Direction.BOTTOM_TO_TOP,
            Direction.LEFT_TO_RIGHT: Direction.RIGHT_TO_LEFT,
            Direction.RIGHT_TO_LEFT: Direction.LEFT_TO_RIGHT,
        }
        return _opposites.get(self)


class TrafficStatus(str, Enum):
    """Scene-level traffic congestion classification."""

    FREE   = "FREE"
    FLOW   = "FLOW"
    HEAVY  = "HEAVY"
    JAMMED = "JAMMED"


class AnomalyType(str, Enum):
    """
    Types of detectable per-vehicle anomalies.

    Inheriting from str ensures correct JSON serialisation without a custom encoder:
        json.dumps(AnomalyType.WRONG_WAY)  ->  '"wrong_way"'
    Note: json.dumps uses the string *value*, not str(), which would give
    'AnomalyType.WRONG_WAY'.  The str-subclass path in the JSON encoder
    serialises directly as the underlying string value.
    """

    STOPPED_VEHICLE = "stopped_vehicle"
    WRONG_WAY       = "wrong_way"
    SUDDEN_SLOWDOWN = "sudden_slowdown"


@dataclass(frozen=True)
class BoundingBox:
    """
    Immutable bounding box — Value Object.
    Wraps (x1, y1, x2, y2) pixel coordinates.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class AnomalyResult:
    """
    Immutable result of a single-vehicle anomaly check — Value Object.

    - type is None when is_anomaly is False.
    - stop_seconds is non-zero only for STOPPED_VEHICLE type.
    - AnomalyType is a str subclass, so instances serialise cleanly to JSON.
    """

    is_anomaly:   bool
    type:         Optional[AnomalyType]
    stop_seconds: float

    @classmethod
    def none(cls) -> "AnomalyResult":
        """Factory for a no-anomaly result."""
        return cls(is_anomaly=False, type=None, stop_seconds=0.0)
