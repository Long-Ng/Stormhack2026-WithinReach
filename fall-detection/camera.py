"""Opening the video source: webcam, DroidCam phone camera, network stream or file.

DroidCam's Windows client installs a virtual webcam ("DroidCam Video") that is
always listed, even with no phone connected; it then sends a solid colour frame.
So "DroidCam is available" means the device exists AND sends a real picture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from config import Config


@dataclass
class Source:
    cap: cv2.VideoCapture
    is_file: bool   # True: use the file's own timestamps for reproducible replays
    label: str      # human-readable description for the console


def list_cameras() -> list[str]:
    """Camera names in OpenCV index order. Empty if names cannot be read (non-Windows)."""
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        return []
    try:
        return FilterGraph().get_input_devices()
    except Exception:
        return []


def find_droidcam(names: list[str]) -> int | None:
    for i, name in enumerate(names):
        if "droidcam" in name.lower():
            return i
    return None


def open_camera(index: int, cfg: Config) -> cv2.VideoCapture:
    """DirectShow first (fast startup); Media Foundation if that fails (DroidCam needs it)."""
    for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
            return cap
        cap.release()
    return cap


def is_live(cap: cv2.VideoCapture, cfg: Config) -> bool:
    """True once cfg.droidcam_live_frames distinct, non-flat frames arrive within
    cfg.droidcam_probe_s. With no phone, DroidCam sends one "Start DroidCam" card and
    then solid green, so a single textured frame is not enough; repeated identical
    frames do not count either, while a real camera's sensor noise makes each one differ."""
    deadline = time.perf_counter() + cfg.droidcam_probe_s
    prev = None
    good = 0
    while time.perf_counter() < deadline:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if frame_spread(frame) > cfg.placeholder_max_std and not (
                prev is not None and np.array_equal(frame, prev)):
            good += 1
            if good >= cfg.droidcam_live_frames:
                return True
        prev = frame
    return False


def frame_spread(frame) -> float:
    """Largest per-channel std. A solid colour frame is ~0 even when the colour is not grey."""
    return float(frame.reshape(-1, frame.shape[-1]).std(axis=0).max())


def open_source(source: str | None, cfg: Config) -> Source:
    if source is not None and "://" in source:
        # e.g. DroidCam over Wi-Fi: http://<phone-ip>:4747/video
        return Source(cv2.VideoCapture(source), False, f"stream {source}")
    if source is not None and not source.isdigit():
        return Source(cv2.VideoCapture(source), True, f"file {source}")
    if source is not None:
        index = int(source)
        return Source(open_camera(index, cfg), False, f"camera {index}")

    # Default: a connected DroidCam phone wins over the configured camera.
    if cfg.prefer_droidcam:
        names = list_cameras()
        index = find_droidcam(names)
        if index is not None:
            cap = open_camera(index, cfg)
            if cap.isOpened() and is_live(cap, cfg):
                return Source(cap, False, f"camera {index} ({names[index]})")
            cap.release()
            print(f"{names[index]} found but no phone is streaming; "
                  f"using camera {cfg.camera_index}")
    return Source(open_camera(cfg.camera_index, cfg), False, f"camera {cfg.camera_index}")
