"""Per-frame feature extraction and smoothing.

Lengths and speeds are normalized by `torso_ref`, the median upright torso
length, so values do not depend on how far the person is from the camera.
"""

from __future__ import annotations

import csv
import math
from collections import deque
from dataclasses import asdict, dataclass, fields

import numpy as np

from config import Config

# Landmark indices (MediaPipe pose)
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
MOTION_POINTS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE)


@dataclass
class Features:
    t: float                    # seconds
    torso_angle: float          # degrees, smoothed. 0 = upright, 90 = horizontal, 180 = upside down
    torso_angle_raw: float
    hip_vel: float              # torso lengths/s, positive = down, smoothed
    hip_vel_peak: float         # max raw hip_vel over the last vel_window_s
    bbox_aspect: float          # width / height over visible landmarks
    hip_height: float | None    # (ankle_y - hip_y) / torso_ref; None if ankles not visible
    motion: float               # torso lengths/s, smoothed
    torso_len: float            # pixels
    torso_ref: float            # pixels
    shoulder_xy: tuple[float, float] | None = None  # pixels, for rest zones
    hip_xy: tuple[float, float] | None = None       # pixels


def _midpoint(lms: np.ndarray, visible: np.ndarray, a: int, b: int) -> np.ndarray | None:
    """Midpoint of a and b; falls back to whichever side is visible (side-on views hide one)."""
    if visible[a] and visible[b]:
        return (lms[a, :2] + lms[b, :2]) / 2.0
    if visible[a]:
        return lms[a, :2].copy()
    if visible[b]:
        return lms[b, :2].copy()
    return None


def torso_angle_deg(shoulder_mid: np.ndarray, hip_mid: np.ndarray) -> float:
    """0 = upright, 90 = horizontal, 180 = upside down (shoulders below hips).

    Keeping the sign matters for high cameras: someone lying head-toward a ceiling
    camera appears vertical but inverted, and must not read as standing.
    Left/right lean is not distinguished.
    """
    dx, dy = shoulder_mid - hip_mid  # image y points down, so upright has dy < 0
    return math.degrees(math.atan2(abs(dx), -dy))


def _ema(prev: float | None, x: float, alpha: float) -> float:
    return x if prev is None else alpha * x + (1.0 - alpha) * prev


class FeatureExtractor:
    """Per-frame features. `motion` is net displacement over cfg.motion_window_s, not
    frame-to-frame speed: pose-model wobble of ~1 px per frame in random directions adds
    up to a large per-frame speed even on a still image, but cancels out over a second,
    while real movement accumulates."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._ref_samples: deque[float] = deque(maxlen=cfg.torso_ref_frames)
        self._torso_ref: float | None = None
        self._vel_samples: deque[tuple[float, float]] = deque()  # (t, raw hip_vel)
        self._reset_tracking()

    def _reset_tracking(self) -> None:
        """Forget frame-to-frame state; used when the pose is lost."""
        self._prev_t: float | None = None
        self._prev_hip_y: float | None = None
        self._angle_s: float | None = None
        self._vel_s: float | None = None
        self._motion_s: float | None = None
        # (t, MOTION_POINTS xy, visible) over the last motion_window_s, for net displacement
        self._motion_hist: deque[tuple[float, np.ndarray, np.ndarray]] = deque()

    def _net_motion(self, lms: np.ndarray, visible: np.ndarray, t: float,
                    torso_ref: float) -> float | None:
        """Mean displacement of MOTION_POINTS since ~motion_window_s ago, torso lengths/s."""
        pts = lms[list(MOTION_POINTS), :2].copy()
        vis = visible[list(MOTION_POINTS)].copy()
        hist = self._motion_hist
        hist.append((t, pts, vis))
        # Keep the newest sample that is at least a window old as the reference.
        while len(hist) > 1 and t - hist[1][0] >= self.cfg.motion_window_s:
            hist.popleft()
        t0, pts0, vis0 = hist[0]
        if t - t0 <= 0:
            return None
        common = vis & vis0
        if not common.any():
            return None
        disp = np.linalg.norm(pts[common] - pts0[common], axis=1)
        return float(disp.mean()) / torso_ref / (t - t0)

    def update(self, lms: np.ndarray | None, t: float) -> Features | None:
        """Feed one frame (t in seconds). Returns None when the torso is not visible."""
        if lms is None:
            self._reset_tracking()
            return None

        cfg = self.cfg
        visible = lms[:, 3] >= cfg.min_visibility
        shoulder_mid = _midpoint(lms, visible, L_SHOULDER, R_SHOULDER)
        hip_mid = _midpoint(lms, visible, L_HIP, R_HIP)
        if shoulder_mid is None or hip_mid is None:
            self._reset_tracking()
            return None

        torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
        angle_raw = torso_angle_deg(shoulder_mid, hip_mid)

        # Learn the reference only while upright; frozen otherwise so foreshortening
        # during a fall does not inflate normalized speeds.
        if angle_raw < cfg.ref_upright_angle and torso_len > 1.0:
            self._ref_samples.append(torso_len)
            self._torso_ref = float(np.median(self._ref_samples))
        # Before the first upright frame, fall back to the live length.
        torso_ref = self._torso_ref if self._torso_ref is not None else max(torso_len, 1.0)

        dt = t - self._prev_t if self._prev_t is not None else 0.0
        hip_y = float(hip_mid[1])

        vel_raw = 0.0
        if dt > 0:
            vel_raw = (hip_y - self._prev_hip_y) / dt / torso_ref
            self._vel_samples.append((t, vel_raw))
        motion_raw = self._net_motion(lms, visible, t, torso_ref)
        while self._vel_samples and t - self._vel_samples[0][0] > cfg.vel_window_s:
            self._vel_samples.popleft()
        vel_peak = max((v for _, v in self._vel_samples), default=0.0)

        alpha = cfg.ema_alpha
        self._angle_s = _ema(self._angle_s, angle_raw, alpha)
        if dt > 0:
            self._vel_s = _ema(self._vel_s, vel_raw, alpha)
        if motion_raw is not None:
            self._motion_s = _ema(self._motion_s, motion_raw, alpha)

        pts = lms[visible, :2]
        bw = float(pts[:, 0].max() - pts[:, 0].min())
        bh = float(pts[:, 1].max() - pts[:, 1].min())
        bbox_aspect = bw / max(bh, 1.0)

        hip_height = None
        if visible[L_ANKLE] or visible[R_ANKLE]:
            ankle_mid = _midpoint(lms, visible, L_ANKLE, R_ANKLE)
            hip_height = (float(ankle_mid[1]) - hip_y) / torso_ref

        self._prev_t = t
        self._prev_hip_y = hip_y

        return Features(
            t=t,
            torso_angle=self._angle_s,
            torso_angle_raw=angle_raw,
            hip_vel=self._vel_s if self._vel_s is not None else 0.0,
            hip_vel_peak=vel_peak,
            bbox_aspect=bbox_aspect,
            hip_height=hip_height,
            motion=self._motion_s if self._motion_s is not None else 0.0,
            torso_len=torso_len,
            torso_ref=torso_ref,
            shoulder_xy=(float(shoulder_mid[0]), float(shoulder_mid[1])),
            hip_xy=(float(hip_mid[0]), float(hip_mid[1])),
        )


POSITIONS = {"shoulder_xy", "hip_xy"}  # pixel points, not features to tune: not logged


class FeatureLogger:
    """Writes one CSV row per frame. Frames without a pose get only `t` and pose=0."""

    COLUMNS = ["pose"] + [f.name for f in fields(Features) if f.name not in POSITIONS]

    def __init__(self, path: str):
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()

    def log(self, t: float, feats: Features | None) -> None:
        if feats is None:
            row = {"pose": 0, "t": f"{t:.3f}"}
        else:
            row = {"pose": 1}
            for k, v in asdict(feats).items():
                if k not in POSITIONS:
                    row[k] = "" if v is None else f"{v:.4f}"
        self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> FeatureLogger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
