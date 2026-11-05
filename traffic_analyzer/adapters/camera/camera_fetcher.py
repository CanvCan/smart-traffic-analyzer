"""
adapters/camera/camera_fetcher.py

Fetches the live camera list from the IZUM (Izmir Metropolitan Municipality)
traffic camera API and resolves stream URLs for each camera.
Separates display name (`name`, UTF-8) from ASCII identifier (`ascii_id`).
"""

import requests
from dataclasses import dataclass
from traffic_analyzer.domain.text import slugify_ascii

CAMERAS_API_URL = "https://izum.izmir.bel.tr/v1/workspaces/cameras"

STREAM_URL_FIELDS = [
    "mjpegStreamUrl",
    "streamUrl", "stream_url", "hlsUrl", "hls_url",
    "liveUrl", "live_url", "url", "videoUrl", "video_url",
    "rtspUrl", "rtsp_url", "src",
]

NAME_FIELDS = [
    "name", "cameraName", "camera_name", "title",
    "label", "description", "location", "address",
]

ID_FIELDS = ["ufid", "id", "cameraId", "camera_id", "uid", "uuid", "code"]


@dataclass
class Camera:
    name: str
    camera_id: str
    stream_url: str
    raw: dict

    @property
    def ascii_id(self) -> str:
        """Strict ASCII identifier derived from the camera display name."""
        return slugify_ascii(self.name)


def _pick(obj: dict, candidates: list[str], fallback: str = "") -> str:
    for key in candidates:
        if key in obj and obj[key]:
            return " ".join(str(obj[key]).split())
    return fallback


def fetch_cameras(debug: bool = False) -> list[Camera]:
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = requests.get(CAMERAS_API_URL, timeout=15, verify=False)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"[IZUM] Cannot reach camera API: {e}") from e

    data = resp.json()

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for wrapper_key in ("cameras", "data", "features", "results", "items", "content"):
            if wrapper_key in data and isinstance(data[wrapper_key], list):
                items = data[wrapper_key]
                break
        else:
            raise RuntimeError(
                f"[IZUM] Unexpected API response format. Top-level keys: {list(data.keys())}"
            )
    else:
        raise RuntimeError(f"[IZUM] Unexpected API response type: {type(data)}")

    if not items:
        raise RuntimeError("[IZUM] API returned an empty camera list.")

    if debug:
        import json
        print(json.dumps(items[0], indent=2, ensure_ascii=False))

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
