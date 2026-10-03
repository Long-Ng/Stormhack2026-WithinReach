"""Cover-to-reset: hold a hand over the lens for cfg.cover_reset_s to reset detection.

"Covered" means the frame is dark and flat. Fires once per cover; the lens must be
uncovered before it can fire again.
"""

from __future__ import annotations

import numpy as np

from camera import frame_spread
from config import Config


class CoverReset:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.covered = False
        self.fired = False
        self.brightness = 0.0
        self.spread = 0.0
        self._start: float | None = None
        self._t = 0.0

    def update(self, frame: np.ndarray, t: float) -> bool:
        """Feed one frame (t in seconds). Returns True on the frame the reset fires."""
        cfg = self.cfg
        self._t = t
        self.brightness = float(frame.mean())
        self.spread = frame_spread(frame)
        self.covered = (self.brightness < cfg.cover_max_brightness
                        and self.spread < cfg.cover_max_std)
        if not self.covered:
            self._start = None
            self.fired = False
            return False
        if self._start is None:
            self._start = t
        if not self.fired and t - self._start >= cfg.cover_reset_s:
            self.fired = True
            return True
        return False

    @property
    def remaining(self) -> float | None:
        """Seconds until the reset fires, or None when not counting down."""
        if not self.covered or self.fired:
            return None
        return max(self.cfg.cover_reset_s - (self._t - self._start), 0.0)
