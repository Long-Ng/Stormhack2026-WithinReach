"""Demo clip injection: press I or Space in the camera window to play a recorded video
through the whole system (detection, alerts, dashboard, phones) in place of the live
camera, so nobody has to fall on stage. Press again, or let the clip end, to go back.

    python main.py --inject demo_fall.mp4      # one clip
    python main.py --inject clips/             # a folder: each press plays the next one

Frames are paced at the clip's own frame rate for viewing, and their timestamps come from
the clip's own timeline (`elapsed_ms`), not the wall clock: if processing runs slower than
the clip, wall-clock time would stretch the fall and make it look too slow to detect.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
KEYS = (ord("i"), ord("I"), ord(" "))


def find_clips(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT)
    return [p] if p.is_file() else []


class ClipInjector:
    def __init__(self, path: str | Path, opener: Callable = cv2.VideoCapture,
                 clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep):
        self.clips = find_clips(path)
        if not self.clips:
            raise FileNotFoundError(f"no video found at {path}")
        self._open, self._clock, self._sleep = opener, clock, sleep
        self._next = 0
        self.cap = None
        self.name: str | None = None
        self._fps = 30.0
        self._t0 = 0.0
        self._n = 0

    @property
    def active(self) -> bool:
        return self.cap is not None

    def toggle(self) -> str:
        """Start the next clip, or stop the one playing. Returns a message for the console."""
        if self.active:
            self.stop()
            return "back to the live camera"
        clip = self.clips[self._next % len(self.clips)]
        self._next += 1
        cap = self._open(str(clip))
        if not cap.isOpened():
            return f"could not open {clip}"
        self.cap, self.name = cap, clip.name
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and 1 <= fps <= 120 else 30.0
        self._t0, self._n = self._clock(), 0
        return f"playing demo clip {clip.name}"

    @property
    def elapsed_ms(self) -> float:
        """Clip time of the last frame returned by read()."""
        return max(self._n - 1, 0) / self._fps * 1000.0

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap, self.name = None, None

    def read(self):
        """Next clip frame, paced to the clip's frame rate. (False, None) when it ends."""
        if self.cap is None:
            return False, None
        due = self._t0 + self._n / self._fps
        wait = due - self._clock()
        if wait > 0:
            self._sleep(wait)
        ok, frame = self.cap.read()
        if not ok:
            self.stop()
            return False, None
        self._n += 1
        return True, frame
