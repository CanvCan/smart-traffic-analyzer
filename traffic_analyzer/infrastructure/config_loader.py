import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any

VALID_DIRECTIONS = {"bottom_to_top", "top_to_bottom", "left_to_right", "right_to_left"}


@dataclass
class InfluxConfig:
    url:    str = "http://localhost:8086"
    token:  str = ""
    org:    str = "myorg"
    bucket: str = "traffic_metrics"


@dataclass
class CameraConfig:
    source_id: str
    video_path: str
    display_width: int

    def __post_init__(self):
        if self.display_width <= 0:
            raise ValueError(f"display_width must be > 0, got {self.display_width}")


@dataclass
class ModelConfig:
    model_path: str
    threshold: float
    resolution: int

    def __post_init__(self):
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {self.threshold}")
        if self.resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {self.resolution}")


@dataclass
class LaneConfig:
    name: str
    roi: List[int]
    points: List[List[int]] = field(default_factory=list)  # polygon points (optional)
    label_pt: List[int] = field(default_factory=list)  # manual label position [x, y]
    expected_direction: str = ""  # e.g. "bottom_to_top" — used for wrong-way detection

    def __post_init__(self):
        if len(self.roi) != 4:
            raise ValueError(f"roi must have exactly 4 elements [x1,y1,x2,y2], got {self.roi}")
        if self.roi[0] >= self.roi[2] or self.roi[1] >= self.roi[3]:
            raise ValueError(f"roi must satisfy x1<x2 and y1<y2, got {self.roi}")
        if self.expected_direction and self.expected_direction not in VALID_DIRECTIONS:
            raise ValueError(f"expected_direction must be one of {VALID_DIRECTIONS}, got {self.expected_direction!r}")


@dataclass
class AppConfig:
    camera: CameraConfig
    model:  ModelConfig
    influx: InfluxConfig = field(default_factory=InfluxConfig)
    lanes:  List[LaneConfig] = field(default_factory=list)


def _parse_lanes(raw_lanes: dict) -> List[LaneConfig]:
    return [
        LaneConfig(
            name=name,
            roi=data["roi"],
            points=data.get("points", []),
            label_pt=data.get("label_pt", []),
            expected_direction=data.get("expected_direction", ""),
        )
        for name, data in raw_lanes.items()
    ]


def load_config(path: str = 'config.json') -> AppConfig:
    with open(path, 'r') as f:
        raw = json.load(f)

    camera = CameraConfig(**raw["camera_settings"])
    model  = ModelConfig(**raw["model_settings"])
    influx = InfluxConfig(**raw["influx_settings"]) if "influx_settings" in raw else InfluxConfig()
    lanes  = _parse_lanes(raw.get("lanes", {}))

    return AppConfig(camera=camera, model=model, influx=influx, lanes=lanes)


def load_camera_lanes(camera_id: str, cameras_dir: Path) -> List[LaneConfig]:
    """
    Load lane config for a specific camera from cameras/<camera_id>.json.
    Returns an empty list if the file does not exist yet (no ROI defined).
    """
    path = cameras_dir / f"{camera_id}.json"
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return _parse_lanes(raw.get("lanes", {}))


def save_camera_lanes(camera_id: str, cameras_dir: Path, lanes_data: dict) -> None:
    """
    Persist lane config for a specific camera to cameras/<camera_id>.json.
    lanes_data must be a dict in the same format as config.json "lanes" section.
    """
    cameras_dir.mkdir(parents=True, exist_ok=True)
    path = cameras_dir / f"{camera_id}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"lanes": lanes_data}, f, indent=4, ensure_ascii=False)
    print(f"[Config] Lanes saved → {path}")
