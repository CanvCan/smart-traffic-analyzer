"""
kafka_layer/kafka_producer.py

Thin Kafka producer wrapper that implements IEventPublisher.

The Analyzer knows nothing about Kafka internals — it only calls
publisher.send(event) through the IEventPublisher interface.

Graceful fallback: if Kafka is unavailable the producer falls back to
console output without crashing the analyzer.
"""

import json

from traffic_analyzer.domain.ports import IEventPublisher

# ── Constants ────────────────────────────────────────────────────────────────
BOOTSTRAP_SERVERS = 'localhost:9092'
TOPIC_VEHICLES    = 'traffic.vehicles'
TOPIC_SNAPSHOTS   = 'traffic.snapshots'


class TrafficProducer(IEventPublisher):
    """
    Wraps KafkaProducer and routes traffic events to the correct topic.
    Implements IEventPublisher so it can be injected into Analyzer.

    Usage (in main.py Composition Root):
        publisher = TrafficProducer()
        analyzer  = Analyzer(video_loop, frame_processor, publisher)
    """

    def __init__(self):
        self._producer = self._connect()

    # ── IEventPublisher ───────────────────────────────────────────────────────

    def send(self, event: dict) -> None:
        """
        Route the event to the correct Kafka topic.
        Falls back to console output if Kafka is unavailable.

          vehicle_detected  -> traffic.vehicles   (keyed by vehicle ID)
          traffic_snapshot  -> traffic.snapshots  (unkeyed)
        """
        event_type = event.get("event_type")
        json_str   = self._serialize(event)

        if self._producer is not None:
            self._send_kafka(event_type, event, json_str)
        else:
            self._send_console(event_type, json_str)

    def close(self) -> None:
        """Flush in-flight messages and close the producer cleanly."""
        if self._producer is not None:
            try:
                self._producer.flush(timeout=5)
                self._producer.close()
                print("[Kafka] Producer flushed and closed.")
            except Exception as e:
                print(f"[Kafka] Error during shutdown: {e}")

    # ── Private ───────────────────────────────────────────────────────────────

    def _send_kafka(self, event_type: str, event: dict, json_str: str) -> None:
        try:
            if event_type == "vehicle_detected":
                key = str(event["vehicle"]["id"]).encode()
                self._producer.send(TOPIC_VEHICLES, key=key,
                                    value=json_str.encode())
            elif event_type == "traffic_snapshot":
                self._producer.send(TOPIC_SNAPSHOTS,
                                    value=json_str.encode())
        except Exception as e:
            print(f"[Kafka] Send error ({event_type}): {e}")
            self._send_console(event_type, json_str)

    @staticmethod
    def _send_console(event_type: str, json_str: str) -> None:
        """Only snapshots are printed — vehicle events would spam the console."""
        if event_type == "traffic_snapshot":
            print(f"[SNAPSHOT] {json_str}")

    @staticmethod
    def _serialize(event: dict) -> str:
        # AnomalyType inherits from str, so default=str is a safety net only.
        return json.dumps(event, ensure_ascii=False, default=str)

    @staticmethod
    def _connect():
        """
        Try to connect to Kafka.  Returns None on failure so the caller
        receives console-only fallback automatically without crashing.
        """
        try:
            from kafka import KafkaProducer
            from kafka.errors import NoBrokersAvailable
            try:
                producer = KafkaProducer(
                    bootstrap_servers=BOOTSTRAP_SERVERS,
                    acks=1,
                    compression_type='lz4',
                    linger_ms=10,
                    retries=3,
                    retry_backoff_ms=200,
                    buffer_memory=67108864,
                    max_block_ms=2000,
                )
                print(f"[Kafka] Connected -> {BOOTSTRAP_SERVERS}")
                return producer
            except NoBrokersAvailable:
                print(f"[Kafka] No broker at {BOOTSTRAP_SERVERS} "
                      f"— running in console-only mode.")
                return None
            except Exception as e:
                print(f"[Kafka] Could not connect ({e}) — console-only mode.")
                return None
        except ImportError:
            print("[Kafka] kafka-python not installed — console-only mode.")
            return None
