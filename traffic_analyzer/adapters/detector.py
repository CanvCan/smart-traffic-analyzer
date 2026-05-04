from abc import ABC, abstractmethod
from typing import Any
import warnings

warnings.filterwarnings("ignore")


class BaseDetector(ABC):
    """
    Strategy interface for object detectors.
    Swap RF-DETR for any other model by implementing this interface.
    """

    @abstractmethod
    def detect(self, frame) -> Any:
        ...


class RFDETRDetector(BaseDetector):
    """RF-DETR detector backed by Roboflow rfdetr package.

    Loads best.pt (RF-DETR XXLarge, dinov2_windowed_base backbone, 6 custom
    Turkish traffic classes: bus/car/dolmus/motorcycle/taxi/truck).
    Returns sv.Detections directly — no post-conversion needed.
    """

    # Architecture params match the best.pt checkpoint exactly.
    _ARCH = dict(
        encoder              = "dinov2_windowed_base",
        hidden_dim           = 512,
        dec_layers           = 5,
        sa_nheads            = 16,
        ca_nheads            = 32,
        dec_n_points         = 4,
        num_windows          = 2,
        patch_size           = 20,
        out_feature_indexes  = [3, 6, 9, 12],
        num_queries          = 300,
        num_select           = 300,
        positional_encoding_size = 44,
        num_classes          = 6,
    )

    def __init__(self, model_path: str, device: str,
                 threshold: float, resolution: int):
        from rfdetr import RFDETRBase
        self._threshold = threshold
        self._model = RFDETRBase(
            **self._ARCH,
            resolution       = resolution,
            pretrain_weights = model_path,
            device           = device,
        )
        print(f"[Detector] RF-DETR {model_path} on {device.upper()} | resolution={resolution}")

    def detect(self, frame):
        import numpy as np
        # OpenCV frame is BGR; RF-DETR expects RGB.
        # .copy() ensures contiguous memory ([::-1] produces negative-stride view
        # which torchvision's to_tensor rejects).
        rgb = frame[:, :, ::-1].copy()
        det = self._model.predict(rgb, threshold=self._threshold)

        # RF-DETR model uses 1-based label indices (0=background, 1=bus, 2=car …).
        # Subtract 1 to get 0-based IDs that match our VehicleClass enum.
        # Invalid IDs (<=0 after subtraction) map to -1 (UNKNOWN).
        if len(det) > 0:
            corrected = det.class_id.astype(np.int64) - 1
            corrected[corrected < 0] = -1
            det.class_id = corrected
        return det

    @property
    def class_names(self) -> dict:
        names = self._model.class_names
        if names:
            return {i: n for i, n in enumerate(names)}
        return {0: "bus", 1: "car", 2: "dolmus", 3: "motorcycle", 4: "taxi", 5: "truck"}
