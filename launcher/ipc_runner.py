"""
launcher/ipc_runner.py

Separate multiprocessing.Process runner and Queue IPC coordinator for the
analysis pipeline and OpenCV HighGUI window. Ensures Tkinter's mainloop()
and OpenCV's cv2.waitKey() never execute in the same process.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from traffic_analyzer.domain.ports import IEventPublisher


class GuiEventQueuePublisher(IEventPublisher):
    """Forwards emitted domain telemetry events across IPC to the Launcher Logs & Streams tab."""

    def __init__(self, event_queue: mp.Queue) -> None:
        self._event_queue = event_queue

    def send(self, event: dict) -> None:
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            try:
                self._event_queue.get_nowait()
                self._event_queue.put_nowait(event)
            except Exception:
                pass
        except Exception:
            pass

    def close(self) -> None:
        pass


def run_analysis_worker_process(
    cmd_queue: mp.Queue,
    status_queue: mp.Queue,
    config_path: str,
    cameras_dir: str,
    visual_only: bool = False,
    source_id: Optional[str] = None,
    video_path: Optional[str] = None,
    event_queue: Optional[mp.Queue] = None,
) -> None:
    """
    Entry point executed inside the child multiprocessing.Process.
    Wires RFDETRDetector, ByteTracker, EventBuilder, FrameProcessor,
    FrameRenderer, GhostTrackManager, InfluxPublisher + TrafficProducer
    (CompositePublisher / NullPublisher), VideoLoop, and Analyzer.
    """
    try:
        status_queue.put({"state": "STARTING", "detail": "Loading configuration and RF-DETR model..."})

        from traffic_analyzer.infrastructure.config_loader import (
            load_camera_lanes,
            load_config,
            resolve_local_video_source_id,
        )
        from traffic_analyzer.adapters.detector import RFDETRDetector
        from traffic_analyzer.adapters.tracker import ByteTracker
        from traffic_analyzer.application.event_builder import EventBuilder
        from traffic_analyzer.application.frame_processor import FrameProcessor
        from traffic_analyzer.infrastructure.video_loop import VideoLoop
        from traffic_analyzer.application.analyzer import Analyzer
        from traffic_analyzer.services.ghost_track_manager import GhostTrackManager
        from traffic_analyzer.visualization.frame_renderer import FrameRenderer
        from traffic_analyzer.domain.ports import CompositePublisher, NullPublisher

        cfg_file = Path(config_path).resolve()
        project_root = cfg_file.parent.parent if cfg_file.parent.name == "traffic_analyzer" else cfg_file.parent
        cam_dir_path = Path(cameras_dir).resolve()

        cfg = load_config(str(cfg_file))
        video_source = str(video_path or cfg.camera.video_path)
        is_live_url = video_source.lower().startswith(("http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://"))
        if not is_live_url and not Path(video_source).is_absolute():
            cand_v1 = (cfg_file.parent / video_source).resolve()
            cand_v2 = (project_root / video_source).resolve()
            video_source = str(cand_v1 if cand_v1.exists() else cand_v2)

        effective_source_id = str(source_id or cfg.camera.source_id).strip()
        if not is_live_url and (not effective_source_id or effective_source_id.lower() == "local"):
            effective_source_id = resolve_local_video_source_id(video_source, cam_dir_path)

        cfg.camera.source_id = effective_source_id
        cfg.camera.video_path = video_source
        cam_lanes = load_camera_lanes(effective_source_id, cam_dir_path)
        cfg.lanes = cam_lanes

        model_path = cfg.model.model_path
        if not Path(model_path).is_absolute():
            candidate = (project_root / model_path).resolve()
            if candidate.exists():
                model_path = str(candidate)

        detector = RFDETRDetector(
            model_path=model_path,
            device=cfg.model.device,
            threshold=cfg.model.threshold,
            resolution=cfg.model.resolution,
            batch_size=cfg.model.batch_size,
        )
        tracker = ByteTracker(
            lost_track_buffer=cfg.tracking.lost_track_buffer,
            min_matching_threshold=cfg.tracking.min_matching_threshold,
            min_consecutive_frames=cfg.tracking.min_consecutive_frames,
        )
        ghost_mgr = GhostTrackManager(
            ghost_frames=cfg.tracking.ghost_frames,
        )
        renderer = FrameRenderer()

        gui_pub: Optional[IEventPublisher] = (
            GuiEventQueuePublisher(event_queue) if event_queue is not None else None
        )

        if visual_only or not cfg.lanes:
            print("[Worker] No ROI defined — running in visual-only mode (NullPublisher).")
            publisher: IEventPublisher = gui_pub if gui_pub is not None else NullPublisher()
        else:
            from traffic_analyzer.adapters.influx_publisher import InfluxPublisher
            from traffic_analyzer.adapters.kafka_producer import TrafficProducer

            influx = InfluxPublisher(
                camera_id=cfg.camera.source_id,
                url=cfg.influx.url,
                token=cfg.influx.token,
                org=cfg.influx.org,
                bucket=cfg.influx.bucket,
            )
            kafka_pub = TrafficProducer(
                bootstrap_servers=cfg.kafka.bootstrap_servers,
            )
            pubs: List[IEventPublisher] = [kafka_pub, influx]
            if gui_pub is not None:
                pubs.append(gui_pub)
            publisher = CompositePublisher(*pubs)

        processor = FrameProcessor(
            detector=detector,
            tracker=tracker,
            event_builder=EventBuilder(cfg),
            ghost_manager=ghost_mgr,
            config=cfg,
        )
        video_loop = VideoLoop(
            video_path=video_source,
            display_width=cfg.camera.display_width,
            cmd_queue=cmd_queue,
            status_queue=status_queue,
            renderer=renderer,
            source_name=cfg.camera.source_id,
            lanes=cfg.lanes,
        )

        status_queue.put({"state": "RUNNING", "detail": f"Analyzing {cfg.camera.source_id}"})
        Analyzer(
            video_loop=video_loop,
            frame_processor=processor,
            publisher=publisher,
        ).run()

    except Exception as exc:
        traceback.print_exc()
        status_queue.put({
            "state": "ERROR",
            "reason": "exception",
            "detail": str(exc),
        })


class AnalysisProcessManager:
    """
    Manages the lifecycle of the child analysis process from the Tkinter
    Launcher process using non-blocking Queue polling.
    """

    def __init__(
        self,
        on_state_change: Callable[[str, Dict[str, Any]], None],
    ) -> None:
        self._on_state_change = on_state_change
        self._process: mp.Process | None = None
        self._cmd_queue: mp.Queue | None = None
        self._status_queue: mp.Queue | None = None
        self.event_queue: mp.Queue = mp.Queue(maxsize=2000)

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(
        self,
        config_path: Path,
        cameras_dir: Path,
        visual_only: bool = False,
        source_id: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> bool:
        if self.is_running:
            return False

        self._cmd_queue = mp.Queue()
        self._status_queue = mp.Queue()
        self._process = mp.Process(
            target=run_analysis_worker_process,
            args=(
                self._cmd_queue,
                self._status_queue,
                str(config_path),
                str(cameras_dir),
                bool(visual_only),
                source_id,
                video_path,
                self.event_queue,
            ),
            daemon=False,
            name="SmartTrafficAnalysisWorker",
        )
        self._process.start()
        return True

    def request_stop(self, timeout_sec: float = 2.0) -> None:
        if not self.is_running:
            self._cleanup_handles()
            return

        if self._cmd_queue is not None:
            try:
                self._cmd_queue.put_nowait({"cmd": "STOP"})
            except Exception:
                pass

        if self._process is not None:
            self._process.join(timeout=timeout_sec)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)

        self._cleanup_handles()
        self._on_state_change("STOPPED", {"reason": "user_stopped", "detail": "Stopped by user."})

    def poll(self) -> None:
        if self._status_queue is not None:
            while True:
                try:
                    msg = self._status_queue.get_nowait()
                except queue.Empty:
                    break
                except Exception:
                    break
                state = str(msg.get("state", "INFO"))
                self._on_state_change(state, msg)
                if state in ("STOPPED", "ERROR"):
                    if self._process is not None:
                        self._process.join(timeout=0.5)
                    self._cleanup_handles()
                    return

        if self._process is not None and not self._process.is_alive():
            exitcode = self._process.exitcode
            self._cleanup_handles()
            if exitcode not in (0, None):
                self._on_state_change(
                    "ERROR",
                    {"reason": "process_exit", "detail": f"Worker exited with code {exitcode}."},
                )
            else:
                self._on_state_change(
                    "STOPPED",
                    {"reason": "window_closed", "detail": "Returned to launcher."},
                )

    def _cleanup_handles(self) -> None:
        self._process = None
        self._cmd_queue = None
        self._status_queue = None


PipelineProcessController = AnalysisProcessManager
