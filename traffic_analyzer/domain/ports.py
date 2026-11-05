"""
domain/ports.py

Outbound port interfaces (Driven Ports in Hexagonal Architecture terminology).

Rule: this file may only import from the Python standard library.
"""

from abc import ABC, abstractmethod
from typing import Any, List


class IDetector(ABC):
    """Contract for object detection models."""

    @abstractmethod
    def detect(self, frame: Any) -> Any:
        ...


class ITracker(ABC):
    """Contract for multi-object tracking algorithms."""

    @abstractmethod
    def update(self, detections: Any, frame: Any = None) -> Any:
        ...


class IEventPublisher(ABC):
    """Contract for publishing traffic domain events."""

    @abstractmethod
    def send(self, event: dict) -> None:
        """Route the event to the appropriate destination."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Flush in-flight messages and release resources."""
        ...


class CompositePublisher(IEventPublisher):
    """Fan-out publisher: forwards every event to multiple IEventPublisher implementations."""

    def __init__(self, *publishers: IEventPublisher):
        self._publishers: List[IEventPublisher] = list(publishers)

    def send(self, event: dict) -> None:
        for pub in self._publishers:
            pub.send(event)

    def close(self) -> None:
        for pub in self._publishers:
            pub.close()


class NullPublisher(IEventPublisher):
    """No-op publisher used when a camera has no ROI defined."""

    def send(self, event: dict) -> None:
        pass

    def close(self) -> None:
        pass
