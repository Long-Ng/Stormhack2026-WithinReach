"""PoseEstimator: wraps the MediaPipe Tasks PoseLandmarker.

The rest of the code only sees `Landmarks`: a (33, 4) float array with columns
x_px, y_px, z, visibility. Coordinates are in pixels (image y axis points down).
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from config import Config

Landmarks = np.ndarray  # shape (33, 4): x_px, y_px, z, visibility

NUM_LANDMARKS = 33


class PoseEstimator:
    def __init__(self, cfg: Config):
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=cfg.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=cfg.min_pose_detection_confidence,
            min_pose_presence_confidence=cfg.min_pose_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: float) -> Landmarks | None:
        """Return pixel-space landmarks for the single person in frame, or None."""
        # detect_for_video requires strictly increasing integer timestamps.
        ts = int(timestamp_ms)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)

        if not result.pose_landmarks:
            return None

        lms = result.pose_landmarks[0]
        out = np.empty((NUM_LANDMARKS, 4), dtype=np.float32)
        for i, lm in enumerate(lms):
            vis = lm.visibility if lm.visibility is not None else 0.0
            out[i] = (lm.x * w, lm.y * h, lm.z, vis)
        return out

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
