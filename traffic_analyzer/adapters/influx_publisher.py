"""
adapters/influx_publisher.py

InfluxDB v2 publisher that implements IEventPublisher.

Every measurement is tagged with `camera_id` so Grafana dashboards can
filter by camera using a dashboard variable ($camera_id).

Measurement schema
──────────────────
vehicle_detected
  tags   : camera_id, lane, vehicle_class
  fields : speed_px (float), is_stopped (int 0/1), is_wrong_way (int 0/1)

traffic_snapshot
  tags   : camera_id, traffic_status
  fields : total (int), car (int), motorcycle (int), bus (int), truck (int),
           dolmus (int), taxi (int), avg_speed_px (float), density (float)

Configuration
─────────────
Set these values in config.json under "influx_settings", or use the defaults
below for a local InfluxDB OSS instance (localhost:8086, no auth).

Falls back to console-only mode if InfluxDB is unreachable or not installed.
"""

from __future__ import annotations
from datetime import datetime, timezone

from traffic_analyzer.domain.ports import IEventPublisher  # noqa: domain port

# ── Defaults (override in config.json → influx_settings) ─────────────────────
DEFAULT_URL    = "http://localhost:8086"
DEFAULT_TOKEN  = ""          # empty = no auth (InfluxDB OSS without auth)
DEFAULT_ORG    = "traffic"
DEFAULT_BUCKET = "traffic_metrics"


class InfluxPublisher(IEventPublisher):
    """
    Writes traffic events to InfluxDB v2.
    Gracefully falls back to console when InfluxDB is unavailable.

    Args:
        camera_id:  Tag value written on every point (e.g. 'CAM-TR-IZM-K87').
        url:        InfluxDB base URL.
        token:      API token (empty string for no-auth OSS).
        org:        InfluxDB organisation name.
        bucket:     Target bucket name.
        batch_size: How many points to buffer before flushing.
    """

    def __init__(
        self,
        camera_id: str,
        url:    str  = DEFAULT_URL,
        token:  str  = DEFAULT_TOKEN,
        org:    str  = DEFAULT_ORG,
        bucket: str  = DEFAULT_BUCKET,
        batch_size: int = 50,
    ):
        self._camera_id   = camera_id
        self._bucket      = bucket
        self._write_api   = None
        self._error_logged = False   # suppress repeated write errors
        self._client      = self._connect(url, token, org, batch_size)

    # ── IEventPublisher ───────────────────────────────────────────────────────

    def send(self, event: dict) -> None:
        etype = event.get("event_type")
        if etype == "vehicle_detected":
            self._write_vehicle(event)
        elif etype == "traffic_snapshot":
            self._write_snapshot(event)

    def close(self) -> None:
        if self._client:
            try:
                self._write_api.close()
                self._client.close()
                print("[InfluxDB] Client closed.")
            except Exception as e:
                print(f"[InfluxDB] Error during shutdown: {e}")

    # ── Write helpers ─────────────────────────────────────────────────────────

    def _write_vehicle(self, event: dict) -> None:
        if not self._write_api:
            return
        from influxdb_client import Point
        v = event.get("vehicle", {})
        k = event.get("kinematics", {})
        a = event.get("anomaly", {})

        p = (
            Point("vehicle_detected")
            .tag("camera_id",     self._camera_id)
            .tag("lane",          str(v.get("lane", "unknown")))
            .tag("vehicle_class", str(v.get("class", "unknown")))
            .field("speed_px",    float(k.get("speed_px_per_sec", 0.0)))
            .field("is_stopped",  int(bool(k.get("is_stopped", False))))
            .field("is_wrong_way",int(bool(a.get("is_anomaly") and
                                          a.get("type") == "wrong_way")))
            .time(datetime.now(timezone.utc))
        )
        self._safe_write(p)

    def _write_snapshot(self, event: dict) -> None:
        if not self._write_api:
            return
        from influxdb_client import Point
        c = event.get("counts", {})
        s = event.get("speed",   {})
        d = event.get("density", {})

        p = (
            Point("traffic_snapshot")
            .tag("camera_id",      self._camera_id)
            .tag("traffic_status", str(d.get("status", "UNKNOWN")))
            .field("total",        int(c.get("total",      0)))
            .field("car",          int(c.get("car",        0)))
            .field("motorcycle",   int(c.get("motorcycle", 0)))
            .field("bus",          int(c.get("bus",        0)))
            .field("truck",        int(c.get("truck",      0)))
            .field("dolmus",       int(c.get("dolmus",     0)))
            .field("taxi",         int(c.get("taxi",       0)))
            .field("avg_speed_px", float(s.get("average_px", 0.0)))
            .field("density",      float(d.get("occupancy_ratio", 0.0)))
            .time(datetime.now(timezone.utc))
        )
        self._safe_write(p)

    def _safe_write(self, point) -> None:
        try:
            self._write_api.write(bucket=self._bucket, record=point)
            self._error_logged = False   # reset on success
        except Exception as e:
            if not self._error_logged:
                # Print only once until a write succeeds again
                code = getattr(e, 'status', None) or str(e)[:60]
                print(f"[InfluxDB] Write error ({code}) — check token in config.json")
                self._error_logged = True

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self, url: str, token: str, org: str, batch_size: int):
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()   # proje kökündeki .env dosyasını yükle

            # config.json'da token yoksa INFLUXDB_TOKEN env değişkenine bak
            resolved_token = token or os.environ.get("INFLUXDB_TOKEN", "")

            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import WriteOptions

            client = InfluxDBClient(url=url, token=resolved_token, org=org)
            # Quick health check
            client.ping()
            self._write_api = client.write_api(
                write_options=WriteOptions(batch_size=batch_size, flush_interval=1_000)
            )
            print(f"[InfluxDB] Connected to {url}  bucket={self._bucket}  camera={self._camera_id}")
            return client
        except Exception as e:
            print(f"[InfluxDB] Not available ({e}) — InfluxDB writes disabled.")
            return None
