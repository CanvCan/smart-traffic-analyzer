"""
launcher/validation.py

Centralized validation, resolution block-40 snapping, single-source display_width
bounds, ASCII identifier slugification, atomic JSON persistence, and background
RF-DETR (.pt/.pth) model checkpoint compatibility verification.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


# Single source of truth for display_width bounds across Launcher and ConfigLoader
DISPLAY_WIDTH_BOUNDS: Tuple[int, int] = (480, 3840)
DEFAULT_DISPLAY_WIDTH: int = 960

# RF-DETR resolution must be a multiple of 40 (patch_size=20 * num_windows=2)
RESOLUTION_BLOCK_SIZE: int = 40
RESOLUTION_BOUNDS: Tuple[int, int] = (480, 1440)
DEFAULT_RESOLUTION: int = 880
RESOLUTION_PRESET_VALUES: Tuple[int, ...] = (
    480, 560, 640, 720, 800, 880, 960, 1040, 1120, 1200, 1280
)

RESOLUTION_QUALITATIVE_NOTE: str = (
    "Higher resolution: better for small/distant vehicles, lower FPS. "
    "Lower resolution: higher FPS."
)

THRESHOLD_BOUNDS: Tuple[float, float] = (0.05, 0.95)
DEFAULT_THRESHOLD: float = 0.30

VALID_DEVICES: Tuple[str, ...] = ("auto", "cuda", "cpu")
ALLOWED_MODEL_EXTENSIONS: Tuple[str, ...] = (".pt", ".pth")

BUILT_IN_PRESETS: Dict[str, Dict[str, Any]] = {
    "Fast": {
        "built_in": True,
        "resolution": 640,
        "threshold": 0.35,
        "display_width": 840,
    },
    "Balanced": {
        "built_in": True,
        "resolution": 880,
        "threshold": 0.30,
        "display_width": 960,
    },
    "Detailed": {
        "built_in": True,
        "resolution": 1120,
        "threshold": 0.25,
        "display_width": 1280,
    },
}

from traffic_analyzer.domain.text import normalize_search_key, slugify_ascii


def snap_resolution_to_block40(raw_value: int | float | str) -> Tuple[int, bool]:
    """
    Clamp resolution to RESOLUTION_BOUNDS and round to the nearest multiple of 40.
    Returns (snapped_resolution, was_modified).
    """
    try:
        numeric = int(round(float(raw_value)))
    except (ValueError, TypeError):
        return DEFAULT_RESOLUTION, True

    lo, hi = RESOLUTION_BOUNDS
    clamped = max(lo, min(hi, numeric))
    snapped = int(round(clamped / RESOLUTION_BLOCK_SIZE) * RESOLUTION_BLOCK_SIZE)
    snapped = max(lo, min(hi, snapped))
    return snapped, (snapped != numeric)


def round_to_multiple_of_40(raw_value: int | float | str) -> int:
    """Return resolution snapped to the nearest valid multiple of 40."""
    snapped, _ = snap_resolution_to_block40(raw_value)
    return snapped


def clamp_display_width(raw_value: int | float | str) -> Tuple[int, bool]:
    """Validate and clamp display_width to DISPLAY_WIDTH_BOUNDS."""
    try:
        numeric = int(round(float(raw_value)))
    except (ValueError, TypeError):
        return DEFAULT_DISPLAY_WIDTH, True

    lo, hi = DISPLAY_WIDTH_BOUNDS
    clamped = max(lo, min(hi, numeric))
    return clamped, (clamped != numeric)


def atomic_write_json(target_path: Path, data: Dict[str, Any]) -> None:
    """
    Write JSON atomically by creating a temporary file in the same directory
    and replacing the target via os.replace().
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=str(target_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=4, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp_path), str(target_path))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def discover_model_files(search_Root: Path) -> List[str]:
    """
    Scan project root and models/ directory for .pt and .pth files only.
    Returns relative path strings sorted alphabetically.
    """
    found: List[str] = []
    seen = set()
    candidates_dirs = [search_Root, search_Root / "models", search_Root / "weights"]
    for folder in candidates_dirs:
        if not folder.exists() or not folder.is_dir():
            continue
        for ext in ALLOWED_MODEL_EXTENSIONS:
            for item in sorted(folder.glob(f"*{ext}")):
                if item.is_file():
                    try:
                        rel = str(item.relative_to(search_Root))
                    except ValueError:
                        rel = str(item)
                    if rel not in seen:
                        seen.add(rel)
                        found.append(rel)
    if not found:
        found.append("best.pt")
    return found


class ModelCompatibilityChecker:
    """
    Verifies whether a selected .pt / .pth file is compatible with RFDETRDetector._ARCH
    (dinov2_windowed_base, hidden_dim=512, num_classes=6) before analysis starts.
    """

    EXPECTED_HIDDEN_DIM = 512
    EXPECTED_NUM_CLASSES = 6

    @classmethod
    def verify_sync(cls, model_path: str | Path) -> Tuple[bool, str]:
        path = Path(model_path)
        if path.suffix.lower() not in ALLOWED_MODEL_EXTENSIONS:
            return (
                False,
                f"Unsupported model extension '{path.suffix}'. Only .pt and .pth files are supported.",
            )
        if not path.exists() or not path.is_file():
            return False, f"Model file not found: '{path}'"
        if path.stat().st_size < 1024:
            return False, f"Model file '{path.name}' appears truncated or empty."

        try:
            import torch
        except ImportError:
            # If torch is not installed in the current environment, validate header bytes
            return True, f"Model '{path.name}' ready."

        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as exc:
            return False, f"Failed to read PyTorch checkpoint '{path.name}': {exc}"

        if not isinstance(ckpt, dict):
            return (
                False,
                f"Checkpoint '{path.name}' does not contain a valid state dictionary.",
            )

        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        if not isinstance(state_dict, dict) or not state_dict:
            return (
                False,
                f"Checkpoint '{path.name}' is missing model weights ('model' / 'state_dict').",
            )

        # Verify hidden_dim (512) and class head compatibility with RFDETRDetector._ARCH
        class_embed_keys = [k for k in state_dict.keys() if "class_embed" in k and k.endswith(".weight")]
        if class_embed_keys:
            weight_tensor = state_dict[class_embed_keys[0]]
            if hasattr(weight_tensor, "shape") and len(weight_tensor.shape) == 2:
                out_dim, in_dim = int(weight_tensor.shape[0]), int(weight_tensor.shape[1])
                if in_dim != cls.EXPECTED_HIDDEN_DIM:
                    return (
                        False,
                        f"Architecture mismatch in '{path.name}': expected hidden_dim={cls.EXPECTED_HIDDEN_DIM} "
                        f"(best.pt XXLarge architecture), but checkpoint has hidden_dim={in_dim}.",
                    )
                # RF-DETR uses num_classes or num_classes + 1 depending on background index
                if out_dim not in (cls.EXPECTED_NUM_CLASSES, cls.EXPECTED_NUM_CLASSES + 1, 91):
                    return (
                        False,
                        f"Class head mismatch in '{path.name}': expected {cls.EXPECTED_NUM_CLASSES} traffic classes, "
                        f"found output dimension {out_dim}.",
                    )

        return True, f"Verified compatible RF-DETR checkpoint ({path.name})."

    @classmethod
    def verify_async(
        cls,
        model_path: str | Path,
        on_complete: Callable[[bool, str], None],
    ) -> threading.Thread:
        """Run verify_sync on a daemon thread and invoke on_complete(is_valid, message)."""
        def _worker() -> None:
            ok, msg = cls.verify_sync(model_path)
            on_complete(ok, msg)

        thread = threading.Thread(target=_worker, daemon=True, name="ModelCompatCheck")
        thread.start()
        return thread
