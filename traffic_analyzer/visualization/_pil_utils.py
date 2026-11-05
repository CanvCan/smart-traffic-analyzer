"""
visualization/_pil_utils.py

Shared PIL font-loading and text-drawing helpers used by both the live
frame Renderer and the ROI canvas ROIRenderer.
Supports Windows and Linux Unicode TrueType fonts for Turkish glyphs.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from traffic_analyzer.domain.text import normalize_search_key, slugify_ascii

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}

_FONT_PATHS = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/segoeuil.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def pil_font(size: int) -> ImageFont.FreeTypeFont:
    """Return a cached PIL TrueType font at *size* points."""
    if size not in _FONT_CACHE:
        for path in _FONT_PATHS:
            try:
                _FONT_CACHE[size] = ImageFont.truetype(path, size)
                return _FONT_CACHE[size]
            except OSError:
                pass
        _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def pil_text(img: np.ndarray, text: str, x: int, y: int,
             font_size: int, color_bgr: tuple, shadow: bool = True) -> None:
    """Render *text* onto *img* (BGR, in-place) using PIL for Unicode support."""
    pil  = Image.fromarray(img[:, :, ::-1])
    draw = ImageDraw.Draw(pil)
    font = pil_font(font_size)
    cr, cg, cb = color_bgr[2], color_bgr[1], color_bgr[0]
    if shadow:
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(cr, cg, cb))
    img[:] = np.array(pil)[:, :, ::-1]


def pil_text_size(text: str, font_size: int) -> tuple[int, int]:
    """Return the (width, height) of *text* rendered at *font_size*."""
    d  = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bb = d.textbbox((0, 0), text, font=pil_font(font_size))
    return int(bb[2] - bb[0]), int(bb[3] - bb[1])
