"""Debug drawing. This overlay is the tuning tool, so keep numbers readable."""

from __future__ import annotations

import time

import cv2
import numpy as np

from cover import CoverReset
from detector import FallDetector, State
from features import Features
from imu import Wearable
from pose import Landmarks

# Body skeleton (face points other than the nose are skipped to reduce clutter).
SKELETON = [
    (11, 12),                                # shoulders
    (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),            # torso
    (23, 25), (25, 27), (24, 26), (26, 28),  # legs
    (27, 29), (29, 31), (27, 31),            # left foot
    (28, 30), (30, 32), (28, 32),            # right foot
]
JOINTS = sorted({0} | {i for edge in SKELETON for i in edge})

BONE_COLOR = (255, 255, 255)
JOINT_COLOR = (0, 200, 255)
TEXT_COLOR = (255, 255, 255)
WARN_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

STATE_COLORS = {  # BGR
    State.UPRIGHT: (0, 200, 0),
    State.FALLING: (0, 230, 255),
    State.ON_GROUND: (0, 140, 255),
    State.FALL_CONFIRMED: (0, 0, 255),
}


def draw_skeleton(frame: np.ndarray, lms: Landmarks, min_visibility: float) -> None:
    visible = lms[:, 3] >= min_visibility
    pts = lms[:, :2].astype(int)
    for a, b in SKELETON:
        if visible[a] and visible[b]:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), BONE_COLOR, 2, cv2.LINE_AA)
    for i in JOINTS:
        if visible[i]:
            cv2.circle(frame, tuple(pts[i]), 4, JOINT_COLOR, -1, cv2.LINE_AA)


def draw_text(frame: np.ndarray, text: str, org: tuple[int, int],
              color=TEXT_COLOR, scale: float = 0.6, thickness: int = 2) -> None:
    """Text with a dark outline so it reads on any background."""
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_features(frame: np.ndarray, feats: Features) -> None:
    hip_h = "  --" if feats.hip_height is None else f"{feats.hip_height:4.2f}"
    lines = [
        f"angle  {feats.torso_angle:5.1f} deg",
        f"hipvel {feats.hip_vel:+5.2f}  pk {feats.hip_vel_peak:+5.2f}",
        f"aspect {feats.bbox_aspect:5.2f}",
        f"hip_h  {hip_h}",
        f"motion {feats.motion:5.2f}",
        f"torso  {feats.torso_len:4.0f}/{feats.torso_ref:4.0f} px",
    ]
    for i, line in enumerate(lines):
        draw_text(frame, line, (10, 25 + 24 * i))


def draw_state(frame: np.ndarray, det: FallDetector) -> None:
    h = frame.shape[0]
    draw_text(frame, det.state.value, (10, h - 20), STATE_COLORS[det.state], scale=1.2, thickness=3)
    remaining = det.confirm_remaining
    if remaining is not None:
        draw_text(frame, f"still {remaining:3.1f}s", (10, h - 60), STATE_COLORS[det.state], scale=0.8)
    elif det.lying_time > 0:
        left = det.cfg.slow_fall_s - det.lying_time
        draw_text(frame, f"lying {left:4.1f}s", (10, h - 60), STATE_COLORS[State.ON_GROUND], scale=0.8)


def draw_centered(frame: np.ndarray, text: str, y: int, color, scale: float) -> None:
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, 3)
    draw_text(frame, text, ((frame.shape[1] - tw) // 2, y), color, scale=scale, thickness=3)


def draw_cover(frame: np.ndarray, cover: CoverReset) -> None:
    h, w = frame.shape[:2]
    # Live brightness / texture readout for tuning cover_max_brightness / cover_max_std.
    draw_text(frame, f"lum {cover.brightness:3.0f} tex {cover.spread:3.0f}", (w - 170, 50), scale=0.5)
    if cover.remaining is not None:
        draw_centered(frame, f"RESET IN {cover.remaining:3.1f}s", h // 2, (0, 230, 255), 1.4)
    elif cover.fired:
        draw_centered(frame, "RESET - uncover camera", h // 2, (0, 200, 0), 1.0)


def draw_wearable(frame: np.ndarray, wearable: Wearable) -> None:
    h, w = frame.shape[:2]
    if not wearable.connected:
        draw_text(frame, "phone OFFLINE", (w - 170, h - 20), WARN_COLOR, scale=0.55)
        return
    hit = wearable.last_impact
    if hit is not None and time.perf_counter() - hit.t < 3.0:
        draw_text(frame, f"IMPACT {hit.peak / 9.81:.1f} g", (w - 190, h - 20),
                  WARN_COLOR, scale=0.7)
    else:
        draw_text(frame, f"phone {wearable.accel:4.1f} m/s2", (w - 190, h - 20), scale=0.55)


def draw_overlay(frame: np.ndarray, lms: Landmarks | None, feats: Features | None,
                 det: FallDetector, fps: float, min_visibility: float,
                 cover: CoverReset | None = None, wearable: Wearable | None = None) -> None:
    if lms is not None:
        draw_skeleton(frame, lms, min_visibility)
    if feats is not None:
        draw_features(frame, feats)
    else:
        draw_text(frame, "NO POSE" if lms is None else "NO TORSO", (10, 30), WARN_COLOR, scale=0.9)

    draw_state(frame, det)
    if cover is not None:
        draw_cover(frame, cover)
    if wearable is not None:
        draw_wearable(frame, wearable)

    h, w = frame.shape[:2]
    draw_text(frame, f"FPS {fps:5.1f}", (w - 130, 25))
