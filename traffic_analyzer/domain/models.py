"""Domain value objects, configuration schemas, and immutable pipeline payloads for Smart Traffic Analyzer."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

DISPLAY_WIDTH_BOUNDS: Tuple[int, int] = (480, 3840)
RESOLUTION_BLOCK_SIZE: int = 40
VALID_DIRECTIONS = {"bottom_to_top", "top_to_bottom", "left_to_right", "right_to_left"}


class VehicleClass(Enum):
    BUS = (0, "Bus", True)
    CAR = (1, "Car", False)
    DOLMUS = (2, "Dolmus", False)
    MOTORCYCLE = (3, "Motorcycle", False)
    TAXI = (4, "Taxi", False)
    TRUCK = (5, "Truck", True)
    UNKNOWN = (-1, "Vehicle", False)

    def __init__(self, class_id: int, label: str, is_heavy: bool) -> None:
        self.class_id = class_id
        self.label = label
        self.is_heavy = is_heavy

    @classmethod
    def from_id(cls, cid: int) -> VehicleClass:
        for member in cls:
            if member.class_id == cid:
                return member
        return cls.UNKNOWN


class TrafficStatus(str, Enum):
    FREE = "FREE"
    FLOW = "FLOW"
    HEAVY = "HEAVY"
    JAMMED = "JAMMED"


class AnomalyType(str, Enum):
    STOPPED_VEHICLE = "stopped_vehicle"
    WRONG_WAY = "wrong_way"
    SUDDEN_SLOWDOWN = "sudden_slowdown"


@dataclass(frozen=True)
class LanePolygon:
    """Immutable geometric and directional specification of a traffic lane."""
    name: str
    roi: Tuple[int, int, int, int]
    points: Tuple[Tuple[int, int], ...] = ()
    label_pt: Tuple[int, int] = (0, 0)
    expected_direction: str = ""
    display_name: str = ""


@dataclass(frozen=True)
class AnomalyResult:
    is_anomaly: bool = False
    type: Optional[str] = None
    stop_seconds: float = 0.0

    @classmethod
    def none(cls) -> AnomalyResult:
        return cls(is_anomaly=False, type=None, stop_seconds=0.0)


@dataclass(frozen=True)
class FramePacket:
    """Atomic frame payload exchanged between CaptureThread and InferenceTrackWorker."""
    frame_id: int
    timestamp: float
    frame: Any
    is_live: bool = False


@dataclass(frozen=True)
class TrackOverlay:
    """Immutable per-vehicle rendering state for UI composition."""
    track_id: int
    class_id: int
    bbox: Tuple[int, int, int, int]
    speed_px: float
    anomaly_type: Optional[str] = None
    is_ghost: bool = False


@dataclass(frozen=True)
class OverlaySnapshot:
    """Complete scene overlay snapshot published atomically to RenderMainLoop."""
    frame_id: int
    timestamp: float
    tracks: Tuple[TrackOverlay, ...] = field(default_factory=tuple)


@dataclass
class ServicesConfig:
    kafka_broker: str = "localhost:9092"
    influxdb_url: str = "http://localhost:8086"
    grafana_url: str = "http://localhost:3000"
    spark_container_name: str = "spark_streaming"


@dataclass
class InfluxConfig:
    url: str = "http://localhost:8086"
    token: str = ""
    org: str = "myorg"
    bucket: str = "traffic_metrics"


@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"


@dataclass
class CameraConfig:
    source_id: str = "PEHLIVANOGLU_KAVSAGI"
    video_path: str = "test_video4.mp4"
    display_width: int = 960

    def __post_init__(self) -> None:
        lo, hi = DISPLAY_WIDTH_BOUNDS
        if not (lo <= int(self.display_width) <= hi):
            raise ValueError(
                f"display_width must be in [{lo}, {hi}], got {self.display_width}"
            )
        self.display_width = int(self.display_width)


@dataclass
class ModelConfig:
    model_path: str = "best.pt"
    threshold: float = 0.30
    resolution: int = 880
    batch_size: int = 1
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.threshold = float(self.threshold)
        self.resolution = int(self.resolution)
        self.batch_size = max(1, int(self.batch_size))
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {self.threshold}")
        if self.resolution <= 0 or self.resolution % RESOLUTION_BLOCK_SIZE != 0:
            raise ValueError(
                f"resolution must be a positive multiple of {RESOLUTION_BLOCK_SIZE}, got {self.resolution}"
            )


@dataclass
class TrackingConfig:
    lost_track_buffer: int = 150
    min_matching_threshold: float = 0.80
    min_consecutive_frames: int = 3
    ghost_frames: int = 30

    def __post_init__(self) -> None:
        self.lost_track_buffer = max(1, int(self.lost_track_buffer))
        self.min_matching_threshold = min(0.99, max(0.05, float(self.min_matching_threshold)))
        self.min_consecutive_frames = max(1, int(self.min_consecutive_frames))
        self.ghost_frames = max(0, int(self.ghost_frames))


@dataclass
class KinematicsConfig:
    ema_tau: float = 0.35
    position_history: int = 45
    direction_window: int = 10
    class_vote_window: int = 20
    min_frames_before_event: int = 5

    def __post_init__(self) -> None:
        self.ema_tau = max(0.05, float(self.ema_tau))
        self.position_history = max(5, int(self.position_history))
        self.direction_window = max(3, int(self.direction_window))
        self.class_vote_window = max(1, int(self.class_vote_window))
        self.min_frames_before_event = max(1, int(self.min_frames_before_event))


@dataclass
class AnomalyConfig:
    stop_std_threshold: float = 3.0
    stop_min_frames: int = 30
    stop_min_elapsed_sec: float = 1.0
    slowdown_ratio: float = 0.50
    slowdown_min_speed: float = 30.0

    def __post_init__(self) -> None:
        self.stop_std_threshold = max(0.1, float(self.stop_std_threshold))
        self.stop_min_frames = max(3, int(self.stop_min_frames))
        self.stop_min_elapsed_sec = max(0.1, float(self.stop_min_elapsed_sec))
        self.slowdown_ratio = min(0.95, max(0.05, float(self.slowdown_ratio)))
        self.slowdown_min_speed = max(1.0, float(self.slowdown_min_speed))


@dataclass
class SceneThresholdsConfig:
    speed_free: float = 65.0
    speed_flow: float = 40.0
    speed_heavy: float = 25.0
    speed_jammed: float = 15.0
    occ_free: float = 0.20
    occ_flow: float = 0.35
    occ_heavy: float = 0.55
    occ_jammed: float = 0.75
    snapshot_every: int = 48

    def __post_init__(self) -> None:
        self.speed_free = float(self.speed_free)
        self.speed_flow = float(self.speed_flow)
        self.speed_heavy = float(self.speed_heavy)
        self.speed_jammed = float(self.speed_jammed)
        self.occ_free = float(self.occ_free)
        self.occ_flow = float(self.occ_flow)
        self.occ_heavy = float(self.occ_heavy)
        self.occ_jammed = float(self.occ_jammed)
        self.snapshot_every = max(1, int(self.snapshot_every))


@dataclass
class LaneConfig:
    name: str
    roi: List[int]
    points: List[List[int]] = field(default_factory=list)
    label_pt: List[int] = field(default_factory=list)
    expected_direction: str = ""
    display_name: str = ""
    visible: bool = True
    locked: bool = False
    order: int = 0

    def __post_init__(self) -> None:
        if len(self.roi) != 4:
            raise ValueError(f"roi must have exactly 4 elements [x1,y1,x2,y2], got {self.roi}")
        if self.roi[0] >= self.roi[2] or self.roi[1] >= self.roi[3]:
            raise ValueError(f"roi must satisfy x1<x2 and y1<y2, got {self.roi}")
        if self.expected_direction and self.expected_direction not in VALID_DIRECTIONS:
            raise ValueError(f"expected_direction must be one of {VALID_DIRECTIONS}, got {self.expected_direction!r}")
        if not self.display_name:
            self.display_name = self.name


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    services: ServicesConfig = field(default_factory=ServicesConfig)
    influx: InfluxConfig = field(default_factory=InfluxConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    scene_thresholds: SceneThresholdsConfig = field(default_factory=SceneThresholdsConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    lanes: List[LaneConfig] = field(default_factory=list)
    presets: Dict[str, Any] = field(default_factory=dict)
