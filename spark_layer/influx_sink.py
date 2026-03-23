"""
spark_layer/influx_sink.py

InfluxDB sink writers for each PySpark streaming query (Q1–Q11).
Plugged into Spark Structured Streaming via foreachBatch.

Architecture:
  - InfluxConfig carries all connection parameters; values are read from
    environment variables at startup and can be overridden per-instance for testing.
  - BaseInfluxWriter provides a shared __call__ / _write contract.
  - Each QnWriter subclass implements build_points() to map Spark rows to
    influxdb_client Point objects for its specific measurement schema.
  - The InfluxDB client is lazily initialised per writer instance — no shared
    global state, safe for concurrent Spark executors.
  - Errors are isolated per batch: a single write failure never stops the stream.
  - camera_id is a tag on every measurement, enabling multi-camera dashboards.
  - Point timestamps are assigned by the InfluxDB server at write time.

Install dependency:
    pip install influxdb-client
"""

import math
import os
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

load_dotenv()


# ── InfluxDB connection configuration ────────────────────────────────────────

@dataclass
class InfluxConfig:
    """
    Value object carrying all InfluxDB connection parameters.

    Used instead of hardcoded constants; a different config can be
    injected in tests or alternative environments.
    """
    url:    str
    token:  str
    org:    str
    bucket: str

    @classmethod
    def from_env(cls) -> "InfluxConfig":
        """Build configuration from environment variables."""
        return cls(
            url    = os.getenv("INFLUXDB_URL",    "http://localhost:8086"),
            token  = os.getenv("INFLUXDB_TOKEN",  ""),
            org    = os.getenv("INFLUXDB_ORG",    "myorg"),
            bucket = os.getenv("INFLUXDB_BUCKET", "traffic_metrics"),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cam(row) -> str:
    return row["camera_id"] or "unknown"


def _stdev(values: list) -> float:
    """Sample standard deviation. Returns 0.0 for fewer than 2 values."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))


# ── Base writer ───────────────────────────────────────────────────────────────

class BaseInfluxWriter(ABC):
    """
    Base class for all InfluxDB batch writers.

    Each writer receives its own InfluxConfig via the constructor.
    No global state — multiple instances run safely in concurrent contexts.
    """

    def __init__(self, config: InfluxConfig):
        self._config = config
        self._client: InfluxDBClient = None  # lazy init

    def _get_client(self) -> InfluxDBClient:
        if self._client is None:
            self._client = InfluxDBClient(
                url=self._config.url,
                token=self._config.token,
                org=self._config.org,
            )
        return self._client

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
            self._write(points)

    def _write(self, points: list) -> None:
        try:
            write_api = self._get_client().write_api(write_options=SYNCHRONOUS)
            write_api.write(
                bucket=self._config.bucket,
                org=self._config.org,
                record=points,
            )
            print(f"[InfluxDB] {self.__class__.__name__} → {len(points)} points written")
        except Exception as e:
            print(f"[InfluxDB] {self.__class__.__name__} write error: {e}")
            traceback.print_exc()


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

    batch_df has one row per (window, camera_id, lane, vehicle_class, vehicle_id).
    Second aggregation in Python produces per-(camera_id, lane, vehicle_class) stats.
    stopped_pct / slow_pct are vehicle-based percentages (not frame-based).
    speed_stddev_px is the between-vehicle speed variation (stddev of per-vehicle avgs).
    """

    def build_points(self, rows) -> list:
        groups = defaultdict(list)
        for row in rows:
            key = (_cam(row), row["lane"] or "unknown", row["vehicle_class"] or "unknown")
            groups[key].append(row)

        points = []
        for (camera_id, lane, vehicle_class), vehicle_rows in groups.items():
            speeds = [float(r["avg_speed_px"]) for r in vehicle_rows if r["avg_speed_px"] is not None]
            if not speeds:
                continue
            n = len(speeds)
            p = (
                Point("speed_tracking")
                .tag("camera_id", camera_id)
                .tag("lane", lane)
                .tag("vehicle_class", vehicle_class)
                .field("avg_speed_px", round(sum(speeds) / n, 2))
                .field("speed_stddev_px", round(_stdev(speeds), 2))
                .field("stopped_pct", round(sum(int(r["was_stopped"] or 0) for r in vehicle_rows) / n * 100, 1))
                .field("slow_pct", round(sum(int(r["was_slow"] or 0) for r in vehicle_rows) / n * 100, 1))
            )
            points.append(p)
        return points


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

    batch_df has one row per (window, camera_id, lane, vehicle_id).
    Second aggregation in Python produces per-(camera_id, lane) stats.
    stopped_pct is vehicle-based (not frame-based).
    speed_stddev_px is the between-vehicle speed variation (stddev of per-vehicle avgs).
    """

    def build_points(self, rows) -> list:
        groups = defaultdict(list)
        for row in rows:
            key = (_cam(row), row["lane"] or "unknown")
            groups[key].append(row)

        points = []
        for (camera_id, lane), vehicle_rows in groups.items():
            speeds = [float(r["avg_speed_px"]) for r in vehicle_rows if r["avg_speed_px"] is not None]
            if not speeds:
                continue
            n = len(speeds)
            p = (
                Point("lane_speed_metrics")
                .tag("camera_id", camera_id)
                .tag("lane", lane)
                .field("avg_speed_px", round(sum(speeds) / n, 2))
                .field("speed_stddev_px", round(_stdev(speeds), 2))
                .field("unique_vehicles", n)
                .field("stopped_pct", round(sum(int(r["was_stopped"] or 0) for r in vehicle_rows) / n * 100, 1))
            )
            points.append(p)
        return points


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
                ("Right_Lane",  "right_lane_avg",  "right_lane_max"),
                ("Middle_Lane", "middle_lane_avg", "middle_lane_max"),
                ("Left_Lane",   "left_lane_avg",   "left_lane_max"),
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
            .field("avg_car",        float(row["avg_car"] or 0))
            .field("avg_motorcycle", float(row["avg_motorcycle"] or 0))
            .field("avg_bus",        float(row["avg_bus"] or 0))
            .field("avg_truck",      float(row["avg_truck"] or 0))
            .field("avg_total",      float(row["avg_total"] or 0))
            for row in rows
        ]


# ── Module-level writer instances ────────────────────────────────────────────
# Imported directly by stream_processor.py as foreachBatch sinks.
# Override by constructing writer instances with a custom InfluxConfig.

_default_config = InfluxConfig.from_env()

write_q1  = Q1Writer(_default_config)
write_q2  = Q2Writer(_default_config)
write_q3  = Q3Writer(_default_config)
write_q4  = Q4Writer(_default_config)
write_q5  = Q5Writer(_default_config)
write_q6  = Q6Writer(_default_config)
write_q7  = Q7Writer(_default_config)
write_q8  = Q8Writer(_default_config)
write_q9  = Q9Writer(_default_config)
write_q10 = Q10Writer(_default_config)
write_q11 = Q11Writer(_default_config)
