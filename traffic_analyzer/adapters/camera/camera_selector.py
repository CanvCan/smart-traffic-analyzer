"""
adapters/camera/camera_selector.py

Interactive terminal menu for selecting an IZUM live camera.
Uses questionary for arrow-key navigation (works on Windows Terminal).
"""

import sys
import questionary
from questionary import Style

from traffic_analyzer.adapters.camera.camera_fetcher import Camera, fetch_cameras

_MENU_STYLE = Style([
    ("qmark",        "fg:#00bcd4 bold"),
    ("question",     "bold"),
    ("answer",       "fg:#00bcd4 bold"),
    ("pointer",      "fg:#00bcd4 bold"),
    ("highlighted",  "fg:#00bcd4 bold"),
    ("selected",     "fg:#00bcd4"),
    ("separator",    "fg:#6c6c6c"),
    ("instruction",  "fg:#6c6c6c"),
])


def select_camera(debug: bool = False) -> tuple[str, str]:
    """
    Fetch the IZUM camera list, show an interactive menu, and return the
    (camera_name, stream_url) of the selected camera.

    Args:
        debug: Passed to fetch_cameras() to print the raw first-camera JSON.

    Returns:
        (name, stream_url) tuple for the chosen camera.
    """
    print("\n[IZUM] Fetching live camera list...")
    try:
        cameras = fetch_cameras(debug=debug)
    except RuntimeError as e:
        print(f"\n{e}")
        print("\nFalling back to manual URL entry.")
        return _manual_entry()

    cameras_with_stream = [c for c in cameras if c.stream_url]
    cameras_without     = [c for c in cameras if not c.stream_url]

    if cameras_without:
        print(f"[IZUM] Warning: {len(cameras_without)} camera(s) have no stream URL and will be skipped.")

    if not cameras_with_stream:
        print("[IZUM] No cameras with a known stream URL were found.")
        print("       The stream URL field name may differ from the expected candidates.")
        print("       Run with debug=True to inspect the raw API response.")
        return _manual_entry()

    print(f"[IZUM] {len(cameras_with_stream)} live camera(s) loaded.\n")

    choices = [
        questionary.Choice(title=f"{c.name}  [{c.camera_id}]", value=c)
        for c in cameras_with_stream
    ]
    choices.append(questionary.Choice(title="-- Enter URL manually --", value=None))

    selected: Camera | None = questionary.select(
        "Select a camera to analyse:",
        choices=choices,
        style=_MENU_STYLE,
        use_shortcuts=False,
    ).ask()

    if selected is None:
        return _manual_entry()

    return selected.name, selected.stream_url


def _manual_entry() -> tuple[str, str]:
    """Ask the user to type a stream URL directly."""
    url = questionary.text(
        "Enter the HLS stream URL (e.g. https://.../stream.m3u8):",
        style=_MENU_STYLE,
    ).ask()

    if not url:
        print("No URL provided. Exiting.")
        sys.exit(1)

    return "Manual Camera", url.strip()
