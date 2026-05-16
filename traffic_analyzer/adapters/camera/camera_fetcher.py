"""
adapters/camera/camera_fetcher.py

Fetches the live camera list from the IZUM (Izmir Metropolitan Municipality)
traffic camera API and resolves HLS stream URLs for each camera.

API endpoint: https://izum.izmir.bel.tr/v1/workspaces/cameras
"""

import requests
from dataclasses import dataclass
from typing import Optional

CAMERAS_API_URL = "https://izum.izmir.bel.tr/v1/workspaces/cameras"

# Common field name candidates for stream URL in the API response.
# We try them in order and use the first one found.
STREAM_URL_FIELDS = [
    "mjpegStreamUrl",                               # IZUM actual field
    "streamUrl", "stream_url", "hlsUrl", "hls_url",
    "liveUrl", "live_url", "url", "videoUrl", "video_url",
    "rtspUrl", "rtsp_url", "src",
]

# Common field name candidates for camera display name.
NAME_FIELDS = [
    "name", "cameraName", "camera_name", "title",
    "label", "description", "location", "address",
]

# Common field name candidates for camera ID.
ID_FIELDS = ["ufid", "id", "cameraId", "camera_id", "uid", "uuid", "code"]


@dataclass
class Camera:
    name: str
    camera_id: str
    stream_url: str
    raw: dict  # full original JSON object for debugging


def _pick(obj: dict, candidates: list[str], fallback: str = "") -> str:
    """Return the value of the first matching key found in obj.

    Normalises whitespace so non-breaking spaces, narrow spaces, or other
    Unicode whitespace variants from the API collapse to standard spaces.
    """
    for key in candidates:
        if key in obj and obj[key]:
            return " ".join(str(obj[key]).split())
    return fallback


def fetch_cameras(debug: bool = False) -> list[Camera]:
    """
    Call the IZUM camera API and return a list of Camera objects.

    Args:
        debug: If True, prints the raw JSON of the first camera so you can
               identify which field holds the stream URL.

    Returns:
        List of Camera dataclass instances.

    Raises:
        RuntimeError: If the API is unreachable or returns an unexpected format.
    """
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = requests.get(CAMERAS_API_URL, timeout=15, verify=False)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"[IZUM] Cannot reach camera API: {e}") from e

    data = resp.json()

    # The response may be a list directly or wrapped in a key like
    # {"cameras": [...]} or {"data": [...]} or {"features": [...]}.
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for wrapper_key in ("cameras", "data", "features", "results", "items", "content"):
            if wrapper_key in data and isinstance(data[wrapper_key], list):
                items = data[wrapper_key]
                break
        else:
            raise RuntimeError(
                f"[IZUM] Unexpected API response format. Top-level keys: {list(data.keys())}\n"
                "Please update camera_fetcher.py with the correct wrapper key."
            )
    else:
        raise RuntimeError(f"[IZUM] Unexpected API response type: {type(data)}")

    if not items:
        raise RuntimeError("[IZUM] API returned an empty camera list.")

    if debug:
        print("\n[IZUM DEBUG] First camera object from API:")
        import json
        print(json.dumps(items[0], indent=2, ensure_ascii=False))
        print()

    cameras: list[Camera] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        stream_url = _pick(item, STREAM_URL_FIELDS)
        name       = _pick(item, NAME_FIELDS, fallback=f"Camera-{len(cameras)+1}")
        camera_id  = _pick(item, ID_FIELDS,   fallback=str(len(cameras)))

        cameras.append(Camera(
            name=name,
            camera_id=camera_id,
            stream_url=stream_url,
            raw=item,
        ))

    return cameras
