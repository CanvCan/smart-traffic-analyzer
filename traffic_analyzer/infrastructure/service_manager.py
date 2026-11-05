from __future__ import annotations

import re
import socket
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Tuple

SERVICE_NAMES = ("Docker", "Kafka", "Spark", "InfluxDB", "Grafana")

DEFAULT_KAFKA_PORT = 9092
DEFAULT_INFLUX_PORT = 8086
DEFAULT_GRAFANA_PORT = 3000
SPARK_UI_PORT = 4040

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _parse_compose_ports(compose_path: Path) -> Tuple[int, int, int]:
    """Read host ports for kafka, influxdb, and grafana from docker-compose.yml."""
    kafka_port = DEFAULT_KAFKA_PORT
    influx_port = DEFAULT_INFLUX_PORT
    grafana_port = DEFAULT_GRAFANA_PORT
    if not compose_path.exists():
        return kafka_port, influx_port, grafana_port
    try:
        text = compose_path.read_text(encoding="utf-8")
        current_service = ""
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            if indent == 2 and stripped.endswith(":"):
                current_service = stripped[:-1].strip()
            elif current_service in ("kafka", "influxdb", "grafana") and ":" in stripped and stripped.startswith("-"):
                m = re.search(r"['\"]?(\d+):(\d+)['\"]?", stripped)
                if m:
                    host_p = int(m.group(1))
                    if current_service == "kafka":
                        kafka_port = host_p
                    elif current_service == "influxdb":
                        influx_port = host_p
                    elif current_service == "grafana":
                        grafana_port = host_p
    except Exception:
        pass
    return kafka_port, influx_port, grafana_port


KAFKA_HOST = "localhost"
KAFKA_PORT = DEFAULT_KAFKA_PORT
INFLUXDB_URL = f"http://localhost:{DEFAULT_INFLUX_PORT}"
GRAFANA_URL = f"http://localhost:{DEFAULT_GRAFANA_PORT}"


class ServiceManager:
    """Health checker and Docker Compose controller for local infrastructure services."""

    def __init__(self, project_root: str | Path) -> None:
        self._project_root = Path(project_root).resolve()
        compose_file = self._project_root / "docker-compose.yml"
        k_port, i_port, g_port = _parse_compose_ports(compose_file)
        self.kafka_port = k_port
        self.influxdb_url = f"http://localhost:{i_port}"
        self.grafana_url = f"http://localhost:{g_port}"

    @staticmethod
    def _check_tcp(host: str, port: int, timeout: float = 1.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    @staticmethod
    def _check_http(url: str, timeout: float = 1.2) -> bool:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return 200 <= resp.status < 500
        except urllib.error.HTTPError as exc:
            return 200 <= exc.code < 500
        except Exception:
            return False

    def check_docker(self) -> Tuple[bool, str]:
        """Run `docker info` with CREATE_NO_WINDOW and 2s timeout."""
        try:
            res = subprocess.run(
                ["docker", "info"],
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                timeout=2.0,
                creationflags=_NO_WINDOW,
            )
            if res.returncode == 0:
                return True, "Docker Engine is active and responding."
            err = (res.stderr or res.stdout or "Docker daemon is not responding.").strip()
            return False, err.splitlines()[-1] if err else "Docker daemon is offline."
        except FileNotFoundError:
            return False, "Docker CLI was not found on system PATH."
        except Exception as exc:
            return False, str(exc)

    def check_kafka(self) -> Tuple[bool, str]:
        """Check Kafka broker socket on localhost:9092."""
        ok = self._check_tcp(KAFKA_HOST, self.kafka_port, timeout=1.0)
        if ok:
            return True, f"Broker listening on {KAFKA_HOST}:{self.kafka_port}"
        return False, f"Port {self.kafka_port} unreachable."

    def check_influx(self) -> Tuple[bool, str]:
        """Check InfluxDB HTTP ping on http://localhost:8086/ping."""
        ok = self._check_http(f"{self.influxdb_url}/ping", timeout=1.2)
        if ok:
            return True, f"Time-series DB online at {self.influxdb_url}"
        return False, f"Unreachable at {self.influxdb_url}."

    def check_grafana(self) -> Tuple[bool, str]:
        """Check Grafana HTTP health on http://localhost:3000/api/health."""
        ok = self._check_http(f"{self.grafana_url}/api/health", timeout=1.2)
        if ok:
            return True, f"Dashboard server online at {self.grafana_url}"
        return False, f"Unreachable at {self.grafana_url}."

    def check_spark(self) -> Tuple[bool, str]:
        """Check Spark container via `docker inspect -f {{.State.Running}} spark_streaming`."""
        for cname in ("spark_streaming", "spark_processor"):
            try:
                res = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", cname],
                    cwd=str(self._project_root),
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    creationflags=_NO_WINDOW,
                )
                if res.returncode == 0 and res.stdout.strip().lower() == "true":
                    return True, f"Streaming processor active ({cname}: Running)."
            except Exception:
                pass
        if self._check_tcp("localhost", SPARK_UI_PORT, timeout=0.6):
            return True, f"Spark UI active on localhost:{SPARK_UI_PORT}."
        return False, "Spark container (spark_streaming) is not running."

    def start_infrastructure(self) -> Tuple[bool, str]:
        """Run `docker compose up -d` in background without opening a terminal window."""
        return self.start_services()

    def _inspect_docker_containers(self) -> Tuple[bool, Dict[str, str], str]:
        """Return (docker_running, container_status_map, docker_error_msg)."""
        try:
            res = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.Names}}|{{.Status}}"],
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                timeout=3.0,
                creationflags=_NO_WINDOW,
            )
            if res.returncode != 0:
                err = (res.stderr or res.stdout or "Docker daemon is not responding.").strip()
                return False, {}, err.splitlines()[-1] if err else "Docker daemon is offline."
            status_map: Dict[str, str] = {}
            for line in res.stdout.splitlines():
                if "|" in line:
                    name, st = line.split("|", 1)
                    status_map[name.strip()] = st.strip()
            return True, status_map, ""
        except FileNotFoundError:
            return False, {}, "Docker CLI was not found on system PATH."
        except Exception as exc:
            return False, {}, str(exc)

    def check_all_detailed(self) -> Tuple[Dict[str, bool], Dict[str, str], str]:
        """
        Return:
          - statuses: Dict[service_name, is_running]
          - details:  Dict[service_name, human_readable_detail_or_error]
          - summary:  Overall diagnostic message
        """
        docker_ok, container_map, docker_err = self._inspect_docker_containers()
        kafka_ok = self._check_tcp(KAFKA_HOST, self.kafka_port)
        influx_ok = self._check_http(f"{self.influxdb_url}/ping")
        grafana_ok = self._check_http(f"{self.grafana_url}/api/health")

        spark_container_status = (
            container_map.get("spark_streaming")
            or container_map.get("spark_processor")
            or ""
        )
        spark_ok = (
            spark_container_status.lower().startswith("up")
            or self._check_tcp("localhost", SPARK_UI_PORT, timeout=0.6)
        )

        statuses: Dict[str, bool] = {
            "Docker": docker_ok,
            "Kafka": kafka_ok,
            "Spark": spark_ok,
            "InfluxDB": influx_ok,
            "Grafana": grafana_ok,
        }

        details: Dict[str, str] = {}
        if docker_ok:
            details["Docker"] = "Docker Engine is active and responding."
        else:
            details["Docker"] = f"Cannot connect to Docker Desktop ({docker_err})."

        if kafka_ok:
            details["Kafka"] = f"Broker listening on localhost:{self.kafka_port}"
        else:
            st = container_map.get("kafka_layer", "Not created")
            details["Kafka"] = f"Port {self.kafka_port} unreachable (container: {st})."

        if spark_ok:
            details["Spark"] = f"Streaming processor active ({spark_container_status or 'Running'})."
        else:
            st = spark_container_status or "Container not started"
            details["Spark"] = f"Spark processor is not running ({st})."

        if influx_ok:
            details["InfluxDB"] = f"Time-series DB online at {self.influxdb_url}"
        else:
            st = container_map.get("influxdb", "Not created")
            details["InfluxDB"] = f"Unreachable at {self.influxdb_url} (container: {st})."

        if grafana_ok:
            details["Grafana"] = f"Dashboard server online at {self.grafana_url}"
        else:
            st = container_map.get("grafana", "Not created")
            details["Grafana"] = f"Unreachable at {self.grafana_url} (container: {st})."

        summary = self.format_status_message(statuses, docker_err)
        return statuses, details, summary

    def check_all(self) -> Tuple[Dict[str, bool], str]:
        statuses, _details, summary = self.check_all_detailed()
        return statuses, summary

    @staticmethod
    def format_status_message(statuses: Dict[str, bool], docker_err: str = "") -> str:
        if not statuses.get("Docker", False):
            reason = f" ({docker_err})" if docker_err else ""
            return f"Docker Engine is not running{reason}. Please start Docker Desktop and try again."
        offline = [name for name in SERVICE_NAMES if not statuses.get(name, False)]
        if not offline:
            return "All services (Docker, Kafka, Spark, InfluxDB, Grafana) are running smoothly."
        return (
            f"Stopped services: {', '.join(offline)}. "
            "Click 'Start All Services' below to launch them automatically."
        )

    def check_all_async(
        self,
        callback: Callable[[Dict[str, bool], Dict[str, str], str], None],
    ) -> None:
        def _worker() -> None:
            statuses, details, summary = self.check_all_detailed()
            callback(statuses, details, summary)

        threading.Thread(target=_worker, daemon=True).start()

    def start_services(self) -> Tuple[bool, str]:
        docker_ok, _, docker_err = self._inspect_docker_containers()
        if not docker_ok:
            return (
                False,
                f"Services could not be started because Docker Engine is offline ({docker_err}). "
                "Please open Docker Desktop and try again.",
            )

        commands = [
            ["docker", "compose", "up", "-d"],
            ["docker-compose", "up", "-d"],
        ]
        last_error = "Could not execute docker compose."
        for cmd in commands:
            try:
                res = subprocess.run(
                    cmd,
                    cwd=str(self._project_root),
                    capture_output=True,
                    text=True,
                    timeout=180,
                    creationflags=_NO_WINDOW,
                )
                if res.returncode == 0:
                    return True, "All Docker services started successfully."
                err_text = (res.stderr or res.stdout or "").strip()
                if err_text:
                    lines = [ln.strip() for ln in err_text.splitlines() if ln.strip() and "level=warning" not in ln]
                    last_error = lines[-1] if lines else err_text.splitlines()[-1]
            except FileNotFoundError:
                continue
            except Exception as exc:
                last_error = str(exc)
        return (
            False,
            f"Service startup failed: {last_error}. Please verify Docker Desktop and try again.",
        )

    def start_services_async(
        self,
        callback: Callable[[bool, str, Dict[str, bool], Dict[str, str]], None],
    ) -> None:
        def _worker() -> None:
            ok, msg = self.start_services()
            statuses, details, summary = self.check_all_detailed()
            final_msg = summary if ok else msg
            callback(ok, final_msg, statuses, details)

        threading.Thread(target=_worker, daemon=True).start()
