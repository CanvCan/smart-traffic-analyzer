"""
spark_layer/influx_sink.py

InfluxDB write functions for each PySpark query (Q1-Q6).
Used via foreachBatch in stream_processor.py.

Design:
  - Each function receives a micro-batch DataFrame and a batch_id
  - Rows are converted to influxdb_client Point objects
  - Written to InfluxDB in a single batch_write call
  - Errors are caught per-batch — one bad batch never stops the stream

Install dependency:
    pip install influxdb-client
"""

import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.domain.write_precision import WritePrecision

# ── InfluxDB connection config ────────────────────────────────────────────────
load_dotenv()

INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUX_URL = "http://localhost:8086"
INFLUX_ORG = "myorg"
INFLUX_BUCKET = "traffic_metrics"


def _get_client():
    """Create a fresh InfluxDB client. Called once per batch."""
    return InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG,
    )


# ── Q1 — Lane-based vehicle count ────────────────────────────────────────────
def write_q1(batch_df, batch_id: int) -> None:
    """
    Measurement : lane_vehicle_count
    Tags        : lane, vehicle_class
    Fields      : vehicle_count, avg_speed_px, max_speed_px, min_speed_px
    """
    rows = batch_df.collect()
    if not rows:
        return

    from influxdb_client import Point
    points = []
    for row in rows:
        p = (
            Point("lane_vehicle_count")
            .tag("lane", row["lane"] or "unknown")
            .tag("vehicle_class", row["vehicle_class"] or "unknown")
            .field("vehicle_count", int(row["vehicle_count"]))
            .field("avg_speed_px", float(row["avg_speed_px"] or 0))
            .field("max_speed_px", float(row["max_speed_px"] or 0))
            .field("min_speed_px", float(row["min_speed_px"] or 0))
        )
        points.append(p)

    _batch_write(points, "Q1")


# ── Q2 — Average speed per lane + class ──────────────────────────────────────
def write_q2(batch_df, batch_id: int) -> None:
    """
    Measurement : speed_tracking
    Tags        : lane, vehicle_class
    Fields      : avg_speed_px, sample_count, stopped_pct, slow_pct
    """
    rows = batch_df.collect()
    if not rows:
        return

    from influxdb_client import Point
    points = []
    for row in rows:
        p = (
            Point("speed_tracking")
            .tag("lane", row["lane"] or "unknown")
            .tag("vehicle_class", row["vehicle_class"] or "unknown")
            .field("avg_speed_px", float(row["avg_speed_px"] or 0))
            .field("sample_count", int(row["sample_count"]))
            .field("stopped_pct", float(row["stopped_pct"] or 0))
            .field("slow_pct", float(row["slow_pct"] or 0))
        )
        points.append(p)

    _batch_write(points, "Q2")


# ── Q3 — Heavy vehicle ratio ──────────────────────────────────────────────────
def write_q3(batch_df, batch_id: int) -> None:
    """
    Measurement : heavy_vehicle_ratio
    Tags        : lane
    Fields      : total_vehicles, heavy_count, bus_count, truck_count, heavy_ratio_pct
    """
    rows = batch_df.collect()
    if not rows:
        return

    from influxdb_client import Point
    points = []
    for row in rows:
        p = (
            Point("heavy_vehicle_ratio")
            .tag("lane", row["lane"] or "unknown")
            .field("total_vehicles", int(row["total_vehicles"]))
            .field("heavy_count", int(row["heavy_count"] or 0))
            .field("bus_count", int(row["bus_count"] or 0))
            .field("truck_count", int(row["truck_count"] or 0))
            .field("heavy_ratio_pct", float(row["heavy_ratio_pct"] or 0))
        )
        points.append(p)

    _batch_write(points, "Q3")


# ── Q4 — Anomaly density ──────────────────────────────────────────────────────
def write_q4(batch_df, batch_id: int) -> None:
    """
    Measurement : anomaly_density
    Tags        : lane, anomaly_type
    Fields      : anomaly_count, avg_stop_seconds, max_stop_seconds
    """
    rows = batch_df.collect()
    if not rows:
        return

    from influxdb_client import Point
    points = []
    for row in rows:
        p = (
            Point("anomaly_density")
            .tag("lane", row["lane"] or "unknown")
            .tag("anomaly_type", row["anomaly_type"] or "unknown")
            .field("anomaly_count", int(row["anomaly_count"]))
            .field("avg_stop_seconds", float(row["avg_stop_seconds"] or 0))
            .field("max_stop_seconds", float(row["max_stop_seconds"] or 0))
        )
        points.append(p)

    _batch_write(points, "Q4")


# ── Q5 — Traffic status transitions ──────────────────────────────────────────
def write_q5(batch_df, batch_id: int) -> None:
    """
    Measurement : traffic_status_transitions
    Tags        : traffic_status
    Fields      : snapshot_count, avg_speed_px, avg_vehicle_count,
                  avg_occupancy, avg_heavy_pct
    """
    rows = batch_df.collect()
    if not rows:
        return

    from influxdb_client import Point
    points = []
    for row in rows:
        p = (
            Point("traffic_status_transitions")
            .tag("traffic_status", row["traffic_status"] or "unknown")
            .field("snapshot_count", int(row["snapshot_count"]))
            .field("avg_speed_px", float(row["avg_speed_px"] or 0))
            .field("avg_min_speed_px", float(row["avg_min_speed_px"] or 0))
            .field("avg_max_speed_px", float(row["avg_max_speed_px"] or 0))
            .field("avg_vehicle_count", float(row["avg_vehicle_count"] or 0))
            .field("avg_occupancy", float(row["avg_occupancy"] or 0))
            .field("avg_heavy_pct", float(row["avg_heavy_pct"] or 0))
        )
        points.append(p)

    _batch_write(points, "Q5")


# ── Q6 — Lane-based traffic status ───────────────────────────────────────────
def write_q6(batch_df, batch_id: int) -> None:
    """
    Measurement : lane_traffic_status
    Tags        : lane, traffic_status
    Fields      : avg_speed_px, unique_vehicles
    """
    rows = batch_df.collect()
    if not rows:
        return

    from influxdb_client import Point
    points = []
    for row in rows:
        p = (
            Point("lane_traffic_status")
            .tag("lane", row["lane"] or "unknown")
            .tag("traffic_status", row["traffic_status"] or "unknown")
            .field("avg_speed_px", float(row["avg_speed_px"] or 0))
            .field("unique_vehicles", int(row["unique_vehicles"]))
        )
        points.append(p)

    _batch_write(points, "Q6")


# ── Shared write helper ───────────────────────────────────────────────────────
def _batch_write(points: list, query_name: str) -> None:
    """Write a list of Points to InfluxDB. Silently logs errors."""
    try:
        client = _get_client()
        write_api = client.write_api(write_options=SYNCHRONOUS)
        write_api.write(
            bucket=INFLUX_BUCKET,
            org=INFLUX_ORG,
            record=points,
        )
        client.close()
        print(f"[InfluxDB] {query_name} → {len(points)} points written")
    except Exception as e:
        print(f"[InfluxDB] {query_name} write error: {e}")
