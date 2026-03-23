"""
domain/ports.py

Outbound port interfaces (Driven Ports in Hexagonal Architecture terminology).

These interfaces allow the Application/Domain layer to communicate with
the outside world (Kafka, databases, etc.) without knowing the concrete
implementation.  Swap implementations by injecting a different class that
satisfies the same contract — the domain code never changes.

Rule: this file may only import from the Python standard library.
"""

from abc import ABC, abstractmethod


class IEventPublisher(ABC):
    """
    Contract for publishing traffic domain events.

    Analyzer and FrameProcessor depend on this interface, not on
    TrafficProducer or any other concrete transport.  To replace Kafka
    with RabbitMQ, write a new class that implements IEventPublisher and
    inject it in main.py — nothing else changes.
    """

    @abstractmethod
    def send(self, event: dict) -> None:
        """Route the event to the appropriate destination."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Flush in-flight messages and release resources."""
        ...
