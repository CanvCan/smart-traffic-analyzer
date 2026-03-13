import json
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class CameraConfig:
    source_id: str
    video_path: str
    display_width: int
    k_factor: float


@dataclass
class ModelConfig:
    yolo_path: str
    conf_threshold: float
    imgsz: int
    half: bool
    iou: float
    agnostic_nms: bool


@dataclass
class LaneConfig:
    name: str
    roi: List[int]
    points: List[List[int]] = field(default_factory=list)  # polygon points (optional)
    label_pt: List[int] = field(default_factory=list)  # manual label position [x, y]


@dataclass
class AppConfig:
    camera: CameraConfig
    model: ModelConfig
    lanes: List[LaneConfig] = field(default_factory=list)


def load_config(path: str = 'config.json') -> AppConfig:
    with open(path, 'r') as f:
        raw = json.load(f)

    camera = CameraConfig(**raw["camera_settings"])
    model = ModelConfig(**raw["model_settings"])
    lanes = [
        LaneConfig(name=name, roi=data["roi"], points=data.get("points", []), label_pt=data.get("label_pt", []))
        for name, data in raw.get("lanes", {}).items()
    ]

    return AppConfig(camera=camera, model=model, lanes=lanes)
