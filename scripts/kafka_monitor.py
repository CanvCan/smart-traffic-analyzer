"""
kafka_monitor.py
Run this alongside main.py to verify Kafka messages are flowing.

Usage:
    python kafka_monitor.py            # monitors both topics
    python kafka_monitor.py vehicles   # only traffic.vehicles
    python kafka_monitor.py snapshots  # only traffic.snapshots
"""

import sys
import json
from kafka import KafkaConsumer

BOOTSTRAP = 'localhost:9092'
TOPIC_V = 'traffic.vehicles'
TOPIC_S = 'traffic.snapshots'


def monitor(topics: list):
    print(f"[Monitor] Listening on: {', '.join(topics)}")
    print("[Monitor] Press Ctrl+C to stop.\n")

    consumer = KafkaConsumer(
        *topics,
        bootstrap_servers=BOOTSTRAP,
        auto_offset_reset='latest',  # only new messages
        enable_auto_commit=True,
        value_deserializer=lambda b: json.loads(b.decode('utf-8')),
        consumer_timeout_ms=30_000,  # stop if no message for 30s
    )

    count = 0
    try:
        for msg in consumer:
            count += 1
            event = msg.value
            etype = event.get("event_type", "unknown")

            if etype == "vehicle_detected":
                v = event.get("vehicle", {})
                k = event.get("kinematics", {})
                print(
                    f"[VEHICLE] "
                    f"ID:{v.get('id'):>4}  "
                    f"{v.get('class', '?'):<12}  "
                    f"Lane: {str(v.get('lane', '?')):<14}  "
                    f"Speed: {k.get('speed_px_per_sec', 0):>7.1f} px/s  "
                    f"Stopped: {k.get('is_stopped', False)}"
                )

            elif etype == "traffic_snapshot":
                c = event.get("counts", {})
                d = event.get("density", {})
                s = event.get("speed", {})
                print(
                    f"\n[SNAPSHOT] Frame {event.get('frame_id')}  |  "
                    f"Total: {c.get('total', 0)}  "
                    f"Car:{c.get('car', 0)} Moto:{c.get('motorcycle', 0)} "
                    f"Bus:{c.get('bus', 0)} Truck:{c.get('truck', 0)}  |  "
                    f"Avg speed: {s.get('average_px', 0):.1f} px/s  |  "
                    f"Status: {d.get('status', '?')}\n"
                )

            else:
                print(f"[UNKNOWN] {event}")

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        print(f"\n[Monitor] Stopped. Total messages received: {count}")


if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    if arg == "vehicles":
        monitor([TOPIC_V])
    elif arg == "snapshots":
        monitor([TOPIC_S])
    else:
        monitor([TOPIC_V, TOPIC_S])
