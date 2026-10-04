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


GRAPH_S = 10.0  # seconds of phone acceleration shown in the graph
GRAPH_LINE = (120, 230, 120)
GRAPH_THRESH = (80, 160, 255)


def draw_phone_graph(frame: np.ndarray, wearable: Wearable | None) -> None:
    """The phone graph for the demo overlay; says so when no phone sensor is set up."""
    if wearable is not None:
        draw_wearable(frame, wearable)
        return
    h, w = frame.shape[:2]
    x0 = w - min(300, w // 2 - 20) - 10
    frame[h - 60:h - 28, x0:w - 10] = (frame[h - 60:h - 28, x0:w - 10] * 0.3).astype(np.uint8)
    draw_text(frame, "Phone sensor not connected", (x0 + 8, h - 38), WARN_COLOR, scale=0.5)


def draw_wearable(frame: np.ndarray, wearable: Wearable) -> None:
    """Scrolling graph of the phone's acceleration (Phyphox), bottom right: the last
    GRAPH_S seconds, the impact threshold as a dashed line, detected impacts in red."""
    h, w = frame.shape[:2]
    gw, gh = min(300, w // 2 - 20), 110
    x0, x1, y1 = w - gw - 10, w - 10, h - 34
    y0 = y1 - gh
    frame[y0 - 26:y1 + 6, x0:x1] = (frame[y0 - 26:y1 + 6, x0:x1] * 0.3).astype(np.uint8)  # dim panel
    now = time.perf_counter()
    if not wearable.connected:
        draw_text(frame, "Phone (Phyphox): OFFLINE", (x0 + 8, y0 - 8), WARN_COLOR, scale=0.5)
        return

    thr = wearable.cfg.imu_impact_ms2
    ymax = max(1.5 * thr, 1.0)

    def xy(t: float, a: float) -> tuple[int, int]:
        x = x0 + (1.0 - (now - t) / GRAPH_S) * (x1 - x0)
        return int(x), int(y1 - min(a, ymax) / ymax * gh)

    ty = xy(now, thr)[1]
    for x in range(x0, x1, 12):  # dashed threshold line
        cv2.line(frame, (x, ty), (min(x + 6, x1), ty), GRAPH_THRESH, 1, cv2.LINE_AA)
    draw_text(frame, f"{thr / 9.81:.1f} g", (x0 + 4, ty - 4), GRAPH_THRESH, scale=0.4, thickness=1)

    pts = [xy(t, a) for t, a in list(wearable.reader.samples) if now - t <= GRAPH_S]
    if len(pts) > 1:
        cv2.polylines(frame, [np.array(pts, np.int32)], False, GRAPH_LINE, 2, cv2.LINE_AA)
    for hit in list(wearable.impact_log):
        if now - hit.t <= GRAPH_S:
            cv2.circle(frame, xy(hit.t, hit.peak), 6, WARN_COLOR, -1, cv2.LINE_AA)

    hit = wearable.last_impact
    if hit is not None and now - hit.t <= wearable.cfg.imu_suspect_s:
        draw_text(frame, f"FALL SUSPECTED - phone shock {hit.peak / 9.81:.1f} g", (x0 + 8, y0 - 8),
                  WARN_COLOR, scale=0.5)
        draw_centered(frame, "FALL SUSPECTED", 80, WARN_COLOR, 1.1)  # big, top centre
    else:
        draw_text(frame, f"Phone (Phyphox) {wearable.accel:4.1f} m/s2", (x0 + 8, y0 - 8), scale=0.5)


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


PRIVACY_BG = (46, 36, 30)        # dark slate (BGR)
PRIVACY_LINE = (235, 225, 215)
PRIVACY_JOINT = (120, 200, 255)


def render_privacy_frame(shape: tuple, lms: Landmarks | None, min_visibility: float) -> np.ndarray:
    """A stick-figure picture with no camera pixels: plain background, the (stabilized)
    skeleton, and a round head. Used for the privacy view, its snapshots and clips."""
    h, w = shape[:2]
    frame = np.full((h, w, 3), PRIVACY_BG, np.uint8)
    thick = max(3, round(w / 160))
    if lms is not None:
        visible = lms[:, 3] >= min_visibility
        pts = lms[:, :2].astype(int)
        for a, b in SKELETON:
            if visible[a] and visible[b]:
                cv2.line(frame, tuple(pts[a]), tuple(pts[b]), PRIVACY_LINE, thick, cv2.LINE_AA)
        for i in JOINTS:
            if visible[i] and i != 0:
                cv2.circle(frame, tuple(pts[i]), thick + 2, PRIVACY_JOINT, -1, cv2.LINE_AA)
        if visible[0]:
            r = thick * 4
            if visible[11] and visible[12]:  # head size from shoulder width
                r = max(r, int(0.3 * np.linalg.norm(lms[11, :2] - lms[12, :2])))
            cv2.circle(frame, tuple(pts[0]), r, PRIVACY_LINE, thick, cv2.LINE_AA)
    else:
        draw_text(frame, "No one in view", (16, h - 20), (170, 170, 170), scale=0.7)
    draw_text(frame, "Privacy view", (w - 175, h - 20), (170, 170, 170), scale=0.6)
    return frame
