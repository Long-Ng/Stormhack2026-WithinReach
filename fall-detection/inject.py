"""Debug: splice a video file into the live camera feed.

    python main.py --inject fall.mp4 --inject-hold 10      # press i in the window to play it
    python main.py --inject a.mp4 --inject b.avi --inject-after 5 --no-display
    python main.py --inject "D:/footage/CAUCAFall/Subject.1"  # every video in the folder

While a clip plays, its frames replace the camera's and everything downstream
(detector, alerts, dashboard, clip recording) treats them as live. Frames are paced
at the clip's own frame rate and stamped on the live clock, so velocities match a
real fall; if processing is slower than the clip, frames are skipped like a slow
camera would. Most dataset clips end right after the fall, so `hold_s` keeps
showing the last frame (the person lying still) long enough to confirm.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def expand_videos(paths: list[str]) -> list[str]:
    """Files as given; a folder becomes the videos under it, sorted."""
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTS)
        else:
            out.append(str(p))
    return out


class Injector:
    def __init__(self, paths: list[str], hold_s: float = 0.0,
                 sleep=time.sleep):
        self.paths = [str(p) for p in paths]
        self.hold_s = hold_s
        self._sleep = sleep
        self._next = 0  # index into paths of the next clip to play
        self._cap: cv2.VideoCapture | None = None
        self._fps = 20.0
        self._t0 = 0.0  # live time (s) of the clip's first frame
        self._idx = -1  # index of the last frame returned
        self._last: np.ndarray | None = None
        self._end_t: float | None = None  # live time the clip ran out
        self.label = ""

    @property
    def active(self) -> bool:
        return self._cap is not None

    def start(self, now: float) -> str | None:
        """Start the next clip (cycling through the list) at live time `now`."""
        if not self.paths:
            return None
        self.stop()
        path = self.paths[self._next % len(self.paths)]
        self._next += 1
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[inject] could not open {path}")
            return None
        self._cap, self._fps = cap, cap.get(cv2.CAP_PROP_FPS) or 20.0
        self._t0, self._idx, self._last, self._end_t = now, -1, None, None
        self.label = f"INJECTED {Path(path).name}"
        return path

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self.label = ""

    def read(self, now: float) -> tuple[np.ndarray, float] | None:
        """Next clip frame and its live timestamp in seconds, waiting until it is due.
        None once the clip (and the hold after it) is over."""
        if self._cap is None:
            return None
        idx = self._idx + 1
        due = self._t0 + idx / self._fps
        if due > now:
            self._sleep(due - now)
        else:  # behind: skip to the frame due now
            idx = max(idx, int((now - self._t0) * self._fps))
            due = self._t0 + idx / self._fps
        if self._end_t is None:
            while self._idx < idx:
                ok, frame = self._cap.read()
                if not ok:
                    self._end_t = due
                    break
                self._idx, self._last = self._idx + 1, frame
        if self._end_t is not None:  # holding the last frame
            if self._last is None or due - self._end_t >= self.hold_s:
                self.stop()
                return None
            self._idx = idx
        return self._last.copy(), due  # copy: main.py draws the overlay in place
