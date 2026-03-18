"""
spark_layer/influx_sink.py

InfluxDB write functions for each PySpark query (Q1-Q11).
Used via foreachBatch in stream_processor.py.

Design:
  - Each writer class extends BaseInfluxWriter and implements build_points()
  - Writers are called as callables: write_q1(batch_df, batch_id)
  - Rows are converted to influxdb_client Point objects
  - Written to InfluxDB in a single batch_write call
  - Errors are caught per-batch — one bad batch never stops the stream
  - InfluxDB client is a module-level singleton (one connection, reused)
  - camera_id is a tag on every measurement for multi-camera support
  - Point timestamps are set by the InfluxDB server at write time

Install dependency:
    pip install influxdb-client
"""

import os
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# ── InfluxDB connection config ────────────────────────────────────────────────
load_dotenv()

INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUX_URL = "http://localhost:8086"
INFLUX_ORG = "myorg"
INFLUX_BUCKET = "traffic_metrics"

# ── Singleton client — one connection shared across all batches ───────────────
_client: InfluxDBClient = None


def _get_client() -> InfluxDBClient:
    global _client
    if _client is None:
        _client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    return _client


def _cam(row) -> str:
    return row["camera_id"] or "unknown"


# ── Shared write helper ───────────────────────────────────────────────────────
def _batch_write(points: list, query_name: str) -> None:
    try:
        write_api = _get_client().write_api(write_options=SYNCHRONOUS)
        write_api.write(
            bucket=INFLUX_BUCKET,
            org=INFLUX_ORG,
            record=points,
        )
        print(f"[InfluxDB] {query_name} → {len(points)} points written")
    except Exception as e:
        print(f"[InfluxDB] {query_name} write error: {e}")
        traceback.print_exc()


# ── Base writer ───────────────────────────────────────────────────────────────

class BaseInfluxWriter(ABC):
    """Base class for all InfluxDB batch writers."""

    @abstractmethod
    def build_points(self, rows) -> list:
        """Convert collected rows into a list of InfluxDB Point objects."""
        ...

    def __call__(self, batch_df, batch_id: int) -> None:
        rows = batch_df.collect()
        if not rows:
            return
        points = self.build_points(rows)
        if points:
            _batch_write(points, self.__class__.__name__)


# ── Q1 — Lane-based vehicle count ────────────────────────────────────────────

class Q1Writer(BaseInfluxWriter):
    """
    Measurement : lane_vehicle_count
    Tags        : camera_id, lane, vehicle_class
    Fields      : vehicle_count, avg_speed_px, max_speed_px, min_speed_px
    """

    def build_points(self, rows) -> list:
        return [
            Point("lane_vehicle_count")
            .tag("camera_id", _cam(row))
            .tag("lane", row["lane"] or "unknown")
            .tag("vehicle_class", row["vehicle_class"] or "unknown")
            .field("vehicle_count", int(row["vehicle_count"]))
            .field("avg_speed_px", float(row["avg_speed_px"] or 0))
            .field("max_speed_px", float(row["max_speed_px"] or 0))
            .field("min_speed_px", float(row["min_speed_px"] or 0))
            for row in rows
        ]


# ── Q2 — Speed tracking per lane + class ─────────────────────────────────────

class Q2Writer(BaseInfluxWriter):
    """
    Measurement : speed_tracking
    Tags        : camera_id, lane, vehicle_class
    Fields      : avg_speed_px, speed_stddev_px, stopped_pct, slow_pct
    """

    def build_points(self, rows) -> list:
        return [
            Point("speed_tracking")
            .tag("camera_id", _cam(row))
            .tag("lane", row["lane"] or "unknown")
            .tag("vehicle_class", row["vehicle_class"] or "unknown")
            .field("avg_speed_px", float(row["avg_speed_px"] or 0))
            .field("speed_stddev_px", float(row["speed_stddev_px"] or 0))
            .field("stopped_pct", float(row["stopped_pct"] or 0))
            .field("slow_pct", float(row["slow_pct"] or 0))
            for row in rows
        ]


# ── Q3 — Heavy vehicle ratio ──────────────────────────────────────────────────

class Q3Writer(BaseInfluxWriter):
    """
    Measurement : heavy_vehicle_ratio
    Tags        : camera_id, lane
    Fields      : total_vehicles, heavy_count, bus_count, truck_count, heavy_ratio_pct
    """

    def build_points(self, rows) -> list:
        return [
            Point("heavy_vehicle_ratio")
            .tag("camera_id", _cam(row))
            .tag("lane", row["lane"] or "unknown")
            .field("total_vehicles", int(row["total_vehicles"]))
            .field("heavy_count", int(row["heavy_count"] or 0))
            .field("bus_count", int(row["bus_count"] or 0))
            .field("truck_count", int(row["truck_count"] or 0))
            .field("heavy_ratio_pct", float(row["heavy_ratio_pct"] or 0))
            for row in rows
        ]


# ── Q4 — Anomaly density ──────────────────────────────────────────────────────

class Q4Writer(BaseInfluxWriter):
    """
    Measurement : anomaly_density
    Tags        : camera_id, lane, anomaly_type
    Fields      : anomaly_count, avg_stop_seconds, max_stop_seconds
    """

    def build_points(self, rows) -> list:
        return [
            Point("anomaly_density")
            .tag("camera_id", _cam(row))
            .tag("lane", row["lane"] or "unknown")
            .tag("anomaly_type", row["anomaly_type"] or "unknown")
            .field("anomaly_count", int(row["anomaly_count"]))
            .field("avg_stop_seconds", float(row["avg_stop_seconds"] or 0))
            .field("max_stop_seconds", float(row["max_stop_seconds"] or 0))
            for row in rows
        ]


# ── Q5 — Traffic status transitions ──────────────────────────────────────────

class Q5Writer(BaseInfluxWriter):
    """
    Measurement : traffic_status_transitions
    Tags        : camera_id, traffic_status
    Fields      : snapshot_count, avg_speed_px, avg_min_speed_px, avg_max_speed_px,
                  avg_vehicle_count, avg_occupancy, avg_heavy_pct
    """

    def build_points(self, rows) -> list:
        return [
            Point("traffic_status_transitions")
            .tag("camera_id", _cam(row))
            .tag("traffic_status", row["traffic_status"] or "unknown")
            .field("snapshot_count", int(row["snapshot_count"]))
            .field("avg_speed_px", float(row["avg_speed_px"] or 0))
            .field("avg_min_speed_px", float(row["avg_min_speed_px"] or 0))
            .field("avg_max_speed_px", float(row["avg_max_speed_px"] or 0))
            .field("avg_vehicle_count", float(row["avg_vehicle_count"] or 0))
            .field("avg_occupancy", float(row["avg_occupancy"] or 0))
            .field("avg_heavy_pct", float(row["avg_heavy_pct"] or 0))
            for row in rows
        ]


# ── Q6 — Lane speed metrics ───────────────────────────────────────────────────

class Q6Writer(BaseInfluxWriter):
    """
    Measurement : lane_speed_metrics
    Tags        : camera_id, lane
    Fields      : avg_speed_px, speed_stddev_px, unique_vehicles, stopped_pct
    """

    def build_points(self, rows) -> list:
        return [
            Point("lane_speed_metrics")
            .tag("camera_id", _cam(row))
            .tag("lane", row["lane"] or "unknown")
            .field("avg_speed_px", float(row["avg_speed_px"] or 0))
            .field("speed_stddev_px", float(row["speed_stddev_px"] or 0))
            .field("unique_vehicles", int(row["unique_vehicles"]))
            .field("stopped_pct", float(row["stopped_pct"] or 0))
            for row in rows
        ]


# ── Q7 — Flow rate per lane ───────────────────────────────────────────────────

class Q7Writer(BaseInfluxWriter):
    """
    Measurement : lane_flow_rate
    Tags        : camera_id, lane
    Fields      : vehicle_count, avg_speed_px
    """

    def build_points(self, rows) -> list:
        return [
            Point("lane_flow_rate")
            .tag("camera_id", _cam(row))
            .tag("lane", row["lane"] or "unknown")
            .field("vehicle_count", int(row["vehicle_count"]))
            .field("avg_speed_px", float(row["avg_speed_px"] or 0))
            for row in rows
        ]


# ── Q8 — Dwell time per lane ──────────────────────────────────────────────────

class Q8Writer(BaseInfluxWriter):
    """
    Measurement : dwell_time
    Tags        : camera_id, lane
    Fields      : avg_dwell_sec, max_dwell_sec, vehicle_count

    batch_df has one row per (window, camera_id, vehicle_id, lane).
    Second aggregation in Python produces per-(camera_id, lane) stats.
    """

    def build_points(self, rows) -> list:
        # Group by (camera_id, lane) — server time used as timestamp
        groups = defaultdict(list)
        for row in rows:
            if row["dwell_seconds"] is not None:
                key = (_cam(row), row["lane"] or "unknown")
                groups[key].append(float(row["dwell_seconds"]))

        if not groups:
            return []

        points = []
        for (camera_id, lane), dwells in groups.items():
            p = (
                Point("dwell_time")
                .tag("camera_id", camera_id)
                .tag("lane", lane)
                .field("avg_dwell_sec", round(sum(dwells) / len(dwells), 1))
                .field("max_dwell_sec", round(max(dwells), 1))
                .field("vehicle_count", len(dwells))
            )
            points.append(p)
        return points


# ── Q9 — Lane occupancy ───────────────────────────────────────────────────────

class Q9Writer(BaseInfluxWriter):
    """
    Measurement : lane_occupancy
    Tags        : camera_id, lane
    Fields      : avg_vehicle_count, max_vehicle_count

    Pivots wide row (one per window) to one InfluxDB point per lane.
    """

    def build_points(self, rows) -> list:
        points = []
        for row in rows:
            camera_id = _cam(row)
            for lane, avg_col, max_col in [
                ("Right_Lane", "right_lane_avg", "right_lane_max"),
                ("Middle_Lane", "middle_lane_avg", "middle_lane_max"),
                ("Left_Lane", "left_lane_avg", "left_lane_max"),
            ]:
                avg_val = row[avg_col]
                max_val = row[max_col]
                if avg_val is None and max_val is None:
                    continue
                p = (
                    Point("lane_occupancy")
                    .tag("camera_id", camera_id)
                    .tag("lane", lane)
                    .field("avg_vehicle_count", float(avg_val or 0))
                    .field("max_vehicle_count", int(max_val or 0))
                )
                points.append(p)
        return points


# ── Q10 — Direction analysis ──────────────────────────────────────────────────

class Q10Writer(BaseInfluxWriter):
    """
    Measurement : direction_analysis
    Tags        : camera_id, lane, direction
    Fields      : vehicle_count, avg_speed_px
    """

    def build_points(self, rows) -> list:
        return [
            Point("direction_analysis")
            .tag("camera_id", _cam(row))
            .tag("lane", row["lane"] or "unknown")
            .tag("direction", row["direction"] or "unknown")
            .field("vehicle_count", int(row["vehicle_count"]))
            .field("avg_speed_px", float(row["avg_speed_px"] or 0))
            for row in rows
        ]


# ── Q11 — Vehicle class breakdown ────────────────────────────────────────────

class Q11Writer(BaseInfluxWriter):
    """
    Measurement : vehicle_class_breakdown
    Tags        : camera_id
    Fields      : avg_car, avg_motorcycle, avg_bus, avg_truck, avg_total
    """

    def build_points(self, rows) -> list:
        return [
            Point("vehicle_class_breakdown")
            .tag("camera_id", _cam(row))
            .field("avg_car", float(row["avg_car"] or 0))
            .field("avg_motorcycle", float(row["avg_motorcycle"] or 0))
            .field("avg_bus", float(row["avg_bus"] or 0))
            .field("avg_truck", float(row["avg_truck"] or 0))
            .field("avg_total", float(row["avg_total"] or 0))
            for row in rows
        ]


# ── Module-level instances for backward compatibility with stream_processor.py ─
write_q1 = Q1Writer()
write_q2 = Q2Writer()
write_q3 = Q3Writer()
write_q4 = Q4Writer()
write_q5 = Q5Writer()
write_q6 = Q6Writer()
write_q7 = Q7Writer()
write_q8 = Q8Writer()
write_q9 = Q9Writer()
write_q10 = Q10Writer()
write_q11 = Q11Writer()
