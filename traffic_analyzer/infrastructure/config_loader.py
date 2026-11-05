"""
traffic_analyzer/infrastructure/config_loader.py

Fault-tolerant configuration loader and atomic persistence functions for Smart Traffic Analyzer.
Uses `inspect.signature` to filter unknown JSON keys safely and populates
domain configuration dataclasses.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from traffic_analyzer.domain.models import (
    DISPLAY_WIDTH_BOUNDS,
    RESOLUTION_BLOCK_SIZE,
    VALID_DIRECTIONS,
    AnomalyConfig,
    AppConfig,
    CameraConfig,
    InfluxConfig,
    KafkaConfig,
    KinematicsConfig,
    LaneConfig,
    ModelConfig,
    SceneThresholdsConfig,
    ServicesConfig,
    TrackingConfig,
)
from traffic_analyzer.domain.text import slugify_ascii

_FINGERPRINT_CHUNK_BYTES = 65536

_DIRECTION_UI_MAP: Dict[str, str] = {
    "DOWN": "top_to_bottom",
    "UP": "bottom_to_top",
    "LEFT": "right_to_left",
    "RIGHT": "left_to_right",
    "top_to_bottom": "top_to_bottom",
    "bottom_to_top": "bottom_to_top",
    "left_to_right": "left_to_right",
    "right_to_left": "right_to_left",
}

T = TypeVar("T")


def _instantiate_filtered(cls: Type[T], raw: Optional[Dict[str, Any]]) -> T:
    """
    Fault-tolerant dataclass initializer: inspects `cls.__init__` parameters
    and passes only recognized keys from `raw`.
    """
    if not raw or not isinstance(raw, dict):
        return cls()  # type: ignore[call-arg]
    sig = inspect.signature(cls.__init__)
    valid_keys = {
        k
        for k, param in sig.parameters.items()
        if k != "self"
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return cls(**filtered)


def atomic_write_json(payload: Dict[str, Any], path: str | Path) -> Path:
    """Write *payload* as formatted UTF-8 JSON to *path* atomically using tmp + fsync + replace."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp_path), str(target))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return target


def resolve_local_video_source_id(
    video_path: str | Path,
    cameras_dir: Optional[Path] = None,
) -> str:
    """
    Derive an isolated, deterministic `source_id` for a local video file from its stem.
    Every local video file maps to `local_<slugified_stem>` (e.g. `test_video4.mp4` -> `local_test_video4`,
    `C:/videos/test_video1.mp4` -> `local_test_video1`). Never returns the generic `'local'` string.
    """
    p = Path(str(video_path))
    raw_stem = slugify_ascii(p.stem or "video").lower()
    if not raw_stem or raw_stem == "local":
        raw_stem = "video"

    if raw_stem.startswith("local_"):
        base_stem = raw_stem[len("local_"):]
    else:
        base_stem = raw_stem

    return f"local_{base_stem}"


def compute_video_source_id(video_path: str | Path) -> str:
    """
    Generate a deterministic, collision-free ASCII source_id for a local video file.
    Combines the ASCII-slugified file stem with a 6-character SHA-256 hex digest
    derived from the file byte size, first 64 KB, and last 64 KB (e.g. test_video4_9f8a2b).
    Never falls back to 'local'.
    """
    p = Path(str(video_path))
    stem = slugify_ascii(p.stem or "video")
    if not p.exists() or not p.is_file():
        return stem
    try:
        size = p.stat().st_size
        hasher = hashlib.sha256()
        hasher.update(str(size).encode("ascii"))
        with open(p, "rb") as fh:
            head = fh.read(_FINGERPRINT_CHUNK_BYTES)
            hasher.update(head)
            if size > _FINGERPRINT_CHUNK_BYTES:
                fh.seek(max(0, size - _FINGERPRINT_CHUNK_BYTES))
                tail = fh.read(_FINGERPRINT_CHUNK_BYTES)
                hasher.update(tail)
        digest6 = hasher.hexdigest()[:6]
        return f"{stem}_{digest6}"
    except OSError:
        return stem


def _normalize_expected_direction(raw_dir: Any) -> str:
    if not raw_dir:
        return ""
    key = str(raw_dir).strip()
    mapped = _DIRECTION_UI_MAP.get(key) or _DIRECTION_UI_MAP.get(key.upper(), "")
    return mapped if mapped in VALID_DIRECTIONS else ""


def _parse_lanes(
    raw_lanes: Union[dict, list, None],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> List[LaneConfig]:
    if not raw_lanes:
        return []

    items: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(raw_lanes, dict):
        for name, data in raw_lanes.items():
            if isinstance(data, dict):
                items.append((str(name), data))
    elif isinstance(raw_lanes, list):
        for i, data in enumerate(raw_lanes):
            if isinstance(data, dict):
                name = str(data.get("name") or f"Lane_{i + 1}")
                items.append((name, data))

    parsed: List[LaneConfig] = []
    for idx, (name, data) in enumerate(items):
        pts = [
            [int(round(p[0] * scale_x)), int(round(p[1] * scale_y))]
            for p in data.get("points", [])
            if isinstance(p, (list, tuple)) and len(p) >= 2
        ]
        raw_roi = data.get("roi")
        if isinstance(raw_roi, (list, tuple)) and len(raw_roi) == 4:
            roi = [
                int(round(raw_roi[0] * scale_x)),
                int(round(raw_roi[1] * scale_y)),
                int(round(raw_roi[2] * scale_x)),
                int(round(raw_roi[3] * scale_y)),
            ]
        elif pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            roi = [min(xs), min(ys), max(xs), max(ys)]
        else:
            continue

        if roi[2] <= roi[0]:
            roi[2] = roi[0] + 1
        if roi[3] <= roi[1]:
            roi[3] = roi[1] + 1

        lbl = data.get("label_pt", [])
        if isinstance(lbl, (list, tuple)) and len(lbl) == 2:
            lbl = [int(round(lbl[0] * scale_x)), int(round(lbl[1] * scale_y))]
        elif pts:
            lbl = [pts[0][0], pts[0][1]]
        else:
            lbl = [roi[0], roi[1]]

        exp_dir = _normalize_expected_direction(data.get("expected_direction"))

        parsed.append(
            LaneConfig(
                name=slugify_ascii(name),
                roi=roi,
                points=pts,
                label_pt=lbl,
                expected_direction=exp_dir,
                display_name=str(data.get("display_name") or name),
                visible=bool(data.get("visible", True)),
                locked=bool(data.get("locked", False)),
                order=int(data.get("order", idx)),
            )
        )
    return parsed


def _canonical_camera_filename(camera_id: str) -> str:
    raw = str(camera_id or "").strip()
    if raw.lower().startswith("local_") or raw.islower():
        return slugify_ascii(raw).lower()
    return slugify_ascii(raw)


def resolve_camera_json_path(camera_id: str, cameras_dir: Path) -> Optional[Path]:
    """
    Resolve cameras/<camera_id>.json with support for `local_<stem>`, `<stem>`,
    and legacy Turkish-character filenames. Never falls back to `local.json`.
    """
    cameras_dir = Path(cameras_dir)
    if not cameras_dir.exists() or not camera_id:
        return None

    raw_id = str(camera_id).strip()
    ascii_upper = slugify_ascii(raw_id)
    ascii_lower = ascii_upper.lower()

    candidates: List[str] = []
    for name in (raw_id, ascii_lower, ascii_upper):
        if name and name not in candidates:
            candidates.append(name)

    if ascii_lower.startswith("local_"):
        stripped = ascii_lower[len("local_"):]
        for v in (stripped, stripped.upper()):
            if v and v not in candidates:
                candidates.append(v)
    else:
        prefixed = f"local_{ascii_lower}"
        if prefixed not in candidates:
            candidates.append(prefixed)

    for cand in candidates:
        p = cameras_dir / f"{cand}.json"
        if p.exists():
            return p

    # Check stem without 6-char hash suffix (e.g. test_video1_84561b -> test_video1 or local_test_video1)
    parts = ascii_upper.rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) == 6:
        for stem_cand in (parts[0], parts[0].lower(), f"local_{parts[0].lower()}"):
            p = cameras_dir / f"{stem_cand}.json"
            if p.exists():
                return p

    target_keys = {slugify_ascii(c).upper() for c in candidates}
    for candidate in cameras_dir.glob("*.json"):
        if candidate.stem.lower() == "local":
            continue
        if slugify_ascii(candidate.stem).upper() in target_keys:
            return candidate
    return None


def load_config(path: str = "traffic_analyzer/config.json") -> AppConfig:
    """
    Fault-tolerant configuration loader reading directly from disk using
    `inspect.signature` to filter dataclass fields safely.
    """
    cfg_path = Path(path)
    if not cfg_path.exists() and Path("traffic_analyzer/config.json").exists():
        cfg_path = Path("traffic_analyzer/config.json")
    elif not cfg_path.exists() and Path("config.json").exists():
        cfg_path = Path("config.json")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    services = _instantiate_filtered(ServicesConfig, raw.get("services_settings"))
    camera = _instantiate_filtered(CameraConfig, raw.get("camera_settings"))
    model = _instantiate_filtered(ModelConfig, raw.get("model_settings"))
    influx = _instantiate_filtered(InfluxConfig, raw.get("influx_settings"))
    kafka = _instantiate_filtered(KafkaConfig, raw.get("kafka_settings"))
    tracking = _instantiate_filtered(TrackingConfig, raw.get("tracking_settings"))
    kinematics = _instantiate_filtered(KinematicsConfig, raw.get("kinematics_settings"))
    anomaly = _instantiate_filtered(AnomalyConfig, raw.get("anomaly_settings"))
    scene_thresholds = _instantiate_filtered(SceneThresholdsConfig, raw.get("scene_thresholds"))

    lanes = _parse_lanes(raw.get("lanes", {}))
    presets = raw.get("presets", {})

    return AppConfig(
        camera=camera,
        model=model,
        services=services,
        influx=influx,
        tracking=tracking,
        kinematics=kinematics,
        anomaly=anomaly,
        scene_thresholds=scene_thresholds,
        kafka=kafka,
        lanes=lanes,
        presets=presets,
    )


def save_config(cfg: AppConfig, path: str | Path = "traffic_analyzer/config.json") -> Path:
    """Serialize AppConfig back to `traffic_analyzer/config.json` atomically on disk."""
    cfg_path = Path(path)
    existing: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    payload: Dict[str, Any] = {
        "services_settings": asdict(cfg.services),
        "influx_settings": asdict(cfg.influx),
        "kafka_settings": asdict(cfg.kafka),
        "camera_settings": asdict(cfg.camera),
        "model_settings": asdict(cfg.model),
        "tracking_settings": asdict(cfg.tracking),
        "kinematics_settings": asdict(cfg.kinematics),
        "anomaly_settings": asdict(cfg.anomaly),
        "scene_thresholds": asdict(cfg.scene_thresholds),
        "presets": cfg.presets or existing.get("presets", {}),
        "lanes": existing.get("lanes", {}),
    }
    return atomic_write_json(payload, cfg_path)


def load_camera_lanes(
    camera_id: str,
    cameras_dir: Path,
    target_resolution: Optional[Tuple[int, int]] = None,
) -> List[LaneConfig]:
    """
    Load lane config for a specific camera or local video from cameras/<camera_id>.json.
    Supports automatic coordinate scaling if target_resolution=(width, height) is given.
    """
    path = resolve_camera_json_path(camera_id, cameras_dir)
    if path is None or not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    scale_x, scale_y = 1.0, 1.0
    saved_res = raw.get("resolution")
    if (
        target_resolution is not None
        and isinstance(saved_res, list)
        and len(saved_res) == 2
        and saved_res[0] > 0
        and saved_res[1] > 0
    ):
        scale_x = float(target_resolution[0]) / float(saved_res[0])
        scale_y = float(target_resolution[1]) / float(saved_res[1])

    return _parse_lanes(raw.get("lanes", {}), scale_x=scale_x, scale_y=scale_y)


def save_camera_lanes(
    camera_id: str,
    cameras_dir: Path,
    lanes_data: Union[dict, list],
    resolution: Optional[List[int]] = None,
    display_name: str = "",
) -> Path:
    """
    Persist lane config for a specific camera or local video to cameras/<source_id>.json atomically.
    resolution stores the native source frame dimensions [orig_width, orig_height].
    """
    cameras_dir = Path(cameras_dir)
    cameras_dir.mkdir(parents=True, exist_ok=True)
    existing_path = resolve_camera_json_path(camera_id, cameras_dir)
    file_stem = existing_path.stem if existing_path is not None else _canonical_camera_filename(camera_id)
    path = cameras_dir / f"{file_stem}.json"
    payload: Dict[str, Any] = {
        "version": 2,
        "source_id": file_stem,
        "display_name": display_name or camera_id,
        "lanes": lanes_data,
    }
    if resolution and len(resolution) == 2:
        payload["resolution"] = [int(resolution[0]), int(resolution[1])]
    written = atomic_write_json(payload, path)
    print(f"[Config] Lanes saved -> {written}")
    return written
