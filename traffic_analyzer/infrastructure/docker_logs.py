"""
traffic_analyzer/infrastructure/docker_logs.py

Streams Docker container logs asynchronously in a background thread.
"""

from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_DEFAULT_COMPOSE_PATH = Path(__file__).resolve().parent.parent.parent / "docker-compose.yml"


def resolve_compose_container_name(
    service_name: str,
    compose_path: Path = _DEFAULT_COMPOSE_PATH,
) -> Optional[str]:
    """Extract `container_name` for *service_name* from `docker-compose.yml` if present."""
    if not compose_path.exists():
        return None
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
            elif current_service == service_name and stripped.startswith("container_name:"):
                m = re.match(r"container_name:\s*['\"]?([^'\"\s#]+)['\"]?", stripped)
                if m:
                    return m.group(1).strip()
    except Exception:
        pass
    return None


class DockerLogStreamer:
    """
    Streams `docker logs -f --tail 100 <container>` in a background daemon thread
    into a thread-safe queue without blocking the UI loop.
    Merges stderr into stdout (`stderr=subprocess.STDOUT`) and reads raw unbuffered
    bytes line-by-line with `line.decode('utf-8', errors='replace')` so both container
    logs and Docker daemon/container errors appear immediately in the GUI console.
    """

    def __init__(
        self,
        container_candidates: List[str],
        service_name: Optional[str] = None,
        compose_path: Path = _DEFAULT_COMPOSE_PATH,
    ) -> None:
        ordered: List[str] = []
        if service_name:
            from_compose = resolve_compose_container_name(service_name, compose_path)
            if from_compose:
                ordered.append(from_compose)
            if service_name not in ordered:
                ordered.append(service_name)
        for cand in container_candidates:
            if cand and cand not in ordered:
                ordered.append(cand)
        self.candidates: List[str] = ordered or ["kafka"]
        self.active_container: str = self.candidates[0]
        self.queue: "queue.Queue[str]" = queue.Queue(maxsize=2000)
        self._stop_evt = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=f"LogStream-{self.candidates[0]}",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _resolve_active_container(self) -> str:
        # 1. Prefer a candidate container that is currently running
        existing_fallback: Optional[str] = None
        for name in self.candidates:
            try:
                res = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", name],
                    capture_output=True,
                    text=True,
                    timeout=2.5,
                    creationflags=CREATE_NO_WINDOW,
                )
                if res.returncode == 0:
                    if existing_fallback is None:
                        existing_fallback = name
                    if res.stdout.strip().lower() == "true":
                        self.active_container = name
                        return name
            except Exception:
                pass

        # 2. Check `docker ps --format {{.Names}}` for compose-prefixed container names
        try:
            ps_res = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=2.5,
                creationflags=CREATE_NO_WINDOW,
            )
            if ps_res.returncode == 0:
                running_names = [n.strip() for n in ps_res.stdout.splitlines() if n.strip()]
                for cand in self.candidates:
                    for actual in running_names:
                        if cand.lower() in actual.lower():
                            self.active_container = actual
                            return actual
        except Exception:
            pass

        resolved = existing_fallback or self.candidates[0]
        self.active_container = resolved
        return resolved

    def _enqueue(self, line: str) -> None:
        if not line:
            return
        try:
            self.queue.put_nowait(line)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(line)
            except Exception:
                pass

    def _run_loop(self) -> None:
        while not self._stop_evt.is_set():
            target = self._resolve_active_container()
            try:
                self._proc = subprocess.Popen(
                    ["docker", "logs", "-f", "--tail", "100", target],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    creationflags=CREATE_NO_WINDOW,
                )
                assert self._proc.stdout is not None
                for raw_bytes in iter(self._proc.stdout.readline, b""):
                    if self._stop_evt.is_set():
                        break
                    line = raw_bytes.decode("utf-8", errors="replace")
                    if line.strip():
                        self._enqueue(line if line.endswith("\n") else f"{line}\n")
                self._proc.wait(timeout=1.0)
            except Exception as exc:
                self._enqueue(f"[DockerLogStreamer] {target}: {exc}\n")
            finally:
                self._proc = None

            if not self._stop_evt.is_set():
                self._stop_evt.wait(3.0)
