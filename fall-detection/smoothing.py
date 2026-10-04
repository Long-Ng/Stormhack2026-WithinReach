"""Display-only skeleton stabilizer: One Euro filter + visibility hysteresis + short hold.

The pose model wobbles a pixel or two per frame even on a still image, which makes the
drawn skeleton shimmer. Detection does not need this (features.py measures motion over a
1 s window), so this is only for what people see: the debug overlay and the
skeleton-only privacy view. Detection keeps using the raw landmarks.

One Euro filter (Casiez et al., CHI 2012): a low-pass filter whose cutoff rises with
speed. Still joints get heavy smoothing (no shimmer); fast joints get little (no lag
during a fall).
"""

from __future__ import annotations

import math

import numpy as np

from config import Config


def _alpha(cutoff, dt: float):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """Vectorized One Euro filter over an array of values (e.g. 33 x 2 pixel coordinates)."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0, shared: bool = False):
        """shared: x is (joints, 2); use one body-wide speed (median joint speed) for the
        cutoff. Per-joint wobble is random and cancels out across joints, while real
        movement moves them together, so jitter no longer loosens the filter."""
        self.min_cutoff, self.beta, self.d_cutoff, self.shared = min_cutoff, beta, d_cutoff, shared
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None
        self.t: float | None = None

    def reset(self) -> None:
        self.x = self.dx = self.t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None or self.t is None or t <= self.t:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x.copy()
        dt = t - self.t
        dx = (x - self.x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        if self.shared:
            cutoff = self.min_cutoff + self.beta * float(np.median(np.linalg.norm(self.dx, axis=1)))
        else:
            cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = _alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


class SkeletonStabilizer:
    """Feed raw landmarks every frame; get landmarks that are calmer to look at.

    - x/y go through a One Euro filter (pixels).
    - A joint is shown once its visibility rises above cfg.min_visibility and hidden only
      when it drops below cfg.skel_hide_visibility, so joints do not blink on and off.
    - When the pose drops out, the last skeleton is kept for cfg.skel_hold_s.
    Returned arrays use the same (33, 4) layout; hidden joints get visibility 0.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.filter = OneEuro(cfg.skel_min_cutoff, cfg.skel_beta, cfg.skel_d_cutoff,
                              shared=cfg.skel_body_speed)
        self.shown: np.ndarray | None = None  # per-joint visible flag (hysteresis state)
        self.last: np.ndarray | None = None
        self.last_t: float | None = None

    def update(self, lms: np.ndarray | None, t: float) -> np.ndarray | None:
        cfg = self.cfg
        if lms is None:
            if self.last is not None and self.last_t is not None and t - self.last_t <= cfg.skel_hold_s:
                return self.last.copy()
            self.filter.reset()
            self.shown = None
            self.last = None
            return None

        vis = lms[:, 3]
        if self.shown is None:
            self.shown = vis >= cfg.min_visibility
        else:
            self.shown = np.where(self.shown, vis >= cfg.skel_hide_visibility, vis >= cfg.min_visibility)

        out = lms.copy()
        out[:, :2] = self.filter(lms[:, :2].astype(np.float64), t)
        out[:, 3] = np.where(self.shown, np.maximum(vis, cfg.min_visibility), 0.0)
        self.last, self.last_t = out, t
        return out.copy()
