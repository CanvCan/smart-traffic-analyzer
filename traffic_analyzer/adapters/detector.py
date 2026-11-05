from typing import Any
import warnings

from traffic_analyzer.domain.ports import IDetector

warnings.filterwarnings("ignore")


def align_resolution_block40(resolution: int) -> int:
    """Snap resolution to the nearest positive multiple of 40 (patch_size=20 * num_windows=2)."""
    return max(40, int(round(int(resolution) / 40) * 40))


class BaseDetector(IDetector):
    """
    Strategy interface for object detectors.
    Swap RF-DETR for any other model by implementing this interface.
    """

    def detect(self, frame: Any) -> Any:
        raise NotImplementedError


class RFDETRDetector(BaseDetector):
    """RF-DETR detector backed by Roboflow rfdetr package.

    Loads best.pt (RF-DETR XXLarge, dinov2_windowed_base backbone, 6 custom
    Turkish traffic classes: bus/car/dolmus/motorcycle/taxi/truck).
    Includes Block-40 resolution alignment, PyTorch AMP (FP16) on CUDA,
    optimize_for_inference(), and startup CUDA warmup.
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
                 threshold: float, resolution: int, batch_size: int = 1):
        import numpy as np
        import torch
        from rfdetr import RFDETRBase
        if device == "auto" or (device == "cuda" and not torch.cuda.is_available()):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        aligned_res = max(40, int(round(int(resolution) / 40) * 40))
        self._device = device
        self._use_amp = (device == "cuda" and torch.cuda.is_available())
        self._threshold = threshold
        self._resolution = aligned_res
        self._batch_size = max(1, int(batch_size))
        self._model = RFDETRBase(
            **self._ARCH,
            resolution       = aligned_res,
            pretrain_weights = model_path,
            device           = device,
        )
        if hasattr(self._model, "optimize_for_inference"):
            try:
                self._model.optimize_for_inference()
            except Exception:
                pass
        try:
            dummy = np.zeros((aligned_res, aligned_res, 3), dtype=np.uint8)
            if self._use_amp:
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    self._model.predict(dummy, threshold=self._threshold)
            else:
                with torch.inference_mode():
                    self._model.predict(dummy, threshold=self._threshold)
        except Exception:
            pass
        print(
            f"[Detector] RF-DETR {model_path} on {device.upper()} | "
            f"resolution={aligned_res} | AMP={self._use_amp}"
        )

    def detect(self, frame):
        import numpy as np
        import torch
        rgb = frame[:, :, ::-1].copy()
        if self._use_amp:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                det = self._model.predict(rgb, threshold=self._threshold)
        else:
            with torch.inference_mode():
                det = self._model.predict(rgb, threshold=self._threshold)

        if len(det) > 0:
            det.class_id = det.class_id.astype(np.int64) - 1
            det.class_id[det.class_id < 0] = -1
        return det

    def close(self) -> None:
        """Release model weights and clear CUDA cache."""
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    @property
    def class_names(self) -> dict:
        names = self._model.class_names
        if names:
            return {i: n for i, n in enumerate(names)}
        return {0: "bus", 1: "car", 2: "dolmus", 3: "motorcycle", 4: "taxi", 5: "truck"}
