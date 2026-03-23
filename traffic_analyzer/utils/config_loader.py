import json
from dataclasses import dataclass, field
from typing import Dict, List, Any

VALID_DIRECTIONS = {"bottom_to_top", "top_to_bottom", "left_to_right", "right_to_left"}


@dataclass
class CameraConfig:
    source_id: str
    video_path: str
    display_width: int
    k_factor: float

    def __post_init__(self):
        if self.display_width <= 0:
            raise ValueError(f"display_width must be > 0, got {self.display_width}")
        if self.k_factor <= 0:
            raise ValueError(f"k_factor must be > 0, got {self.k_factor}")


@dataclass
class ModelConfig:
    yolo_path: str
    conf_threshold: float
    imgsz: int
    half: bool
    iou: float
    agnostic_nms: bool

    def __post_init__(self):
        if not 0.0 < self.conf_threshold < 1.0:
            raise ValueError(f"conf_threshold must be in (0, 1), got {self.conf_threshold}")
        if self.imgsz <= 0:
            raise ValueError(f"imgsz must be > 0, got {self.imgsz}")
        if not 0.0 < self.iou < 1.0:
            raise ValueError(f"iou must be in (0, 1), got {self.iou}")


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
    model: ModelConfig
    lanes: List[LaneConfig] = field(default_factory=list)


def load_config(path: str = 'config.json') -> AppConfig:
    with open(path, 'r') as f:
        raw = json.load(f)

    camera = CameraConfig(**raw["camera_settings"])
    model = ModelConfig(**raw["model_settings"])
    lanes = [
        LaneConfig(name=name, roi=data["roi"], points=data.get("points", []), label_pt=data.get("label_pt", []), expected_direction=data.get("expected_direction", ""))
        for name, data in raw.get("lanes", {}).items()
    ]

    return AppConfig(camera=camera, model=model, lanes=lanes)
