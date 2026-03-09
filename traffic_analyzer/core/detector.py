from abc import ABC, abstractmethod
from typing import Any
import warnings

warnings.filterwarnings("ignore")


class BaseDetector(ABC):
    """
    Strategy interface for object detectors.
    Swap YOLO for RT-DETR or any other model
    by implementing this interface.
    """

    @abstractmethod
    def detect(self, frame) -> Any:
        ...


class YOLODetector(BaseDetector):
    """YOLO detector backed by Ultralytics."""

    def __init__(self, model_path: str, device: str,
                 conf: float, imgsz: int, half: bool,
                 iou: float, agnostic_nms: bool,
                 target_classes: list):
        from ultralytics import YOLO
        self._model          = YOLO(model_path)
        self._device         = device
        self._conf           = conf
        self._imgsz          = imgsz
        self._half           = half
        self._iou            = iou
        self._agnostic_nms   = agnostic_nms
        self._target_classes = target_classes
        self._model.to(device)
        print(f"[Detector] {model_path} on {device.upper()} | imgsz={imgsz}")

    def detect(self, frame):
        return self._model(
            frame,
            verbose=False,
            classes=self._target_classes,
            conf=self._conf,
            imgsz=self._imgsz,
            half=self._half,
            iou=self._iou,
            agnostic_nms=self._agnostic_nms,
        )[0]

    @property
    def class_names(self) -> dict:
        return self._model.names
