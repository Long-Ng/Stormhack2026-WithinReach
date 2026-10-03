"""Debug drawing. This overlay is the tuning tool, so keep numbers readable."""

from __future__ import annotations

import cv2
import numpy as np

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


def draw_overlay(frame: np.ndarray, lms: Landmarks | None, fps: float,
                 min_visibility: float) -> None:
    if lms is not None:
        draw_skeleton(frame, lms, min_visibility)
    else:
        draw_text(frame, "NO POSE", (10, 60), WARN_COLOR, scale=0.9)

    h, w = frame.shape[:2]
    draw_text(frame, f"FPS {fps:5.1f}", (w - 130, 25))
