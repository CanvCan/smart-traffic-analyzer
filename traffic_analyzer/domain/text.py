"""
traffic_analyzer/domain/text.py

Pure Python text normalization and ASCII slugification functions.
"""

from __future__ import annotations

import re

_TR_CHAR_MAP = str.maketrans({
    "ç": "c", "Ç": "C",
    "ğ": "g", "Ğ": "G",
    "ı": "i", "İ": "I",
    "ö": "o", "Ö": "O",
    "ş": "s", "Ş": "S",
    "ü": "u", "Ü": "U",
})


def slugify_ascii(text: str) -> str:
    """Convert a display name (potentially containing Turkish characters) into a strict ASCII ID."""
    if not text:
        return "UNNAMED"
    transliterated = text.strip().translate(_TR_CHAR_MAP)
    ascii_only = transliterated.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_only).strip("_")
    return slug or "UNNAMED"


def normalize_search_key(text: str) -> str:
    """Normalize text for case-insensitive search with Turkish I/ı and İ/i equivalence."""
    if not text:
        return ""
    return text.translate(_TR_CHAR_MAP).lower().strip()
