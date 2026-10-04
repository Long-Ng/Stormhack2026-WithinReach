import numpy as np
import pytest

from config import Config
from features import FeatureExtractor, torso_angle_deg

FPS = 30.0


def make_lms(shoulder, hip, ankle=None, knee=None):
    """Synthetic (33, 4) landmarks: both sides of each joint at the given (x, y), visible."""
    lms = np.zeros((33, 4), dtype=np.float32)
    for idx, pt in (((11, 12), shoulder), ((23, 24), hip), ((25, 26), knee), ((27, 28), ankle)):
        if pt is not None:
            for i in idx:
                lms[i] = (pt[0], pt[1], 0.0, 1.0)
    return lms


def standing(y_offset=0.0, torso=100.0):
    hip = (320.0, 200.0 + y_offset)
    return make_lms(shoulder=(320.0, hip[1] - torso), hip=hip,
                    knee=(320.0, hip[1] + 90), ankle=(320.0, hip[1] + 180))


def lying(torso=100.0):
    hip = (320.0, 400.0)
    return make_lms(shoulder=(320.0 - torso, 400.0), hip=hip,
                    knee=(410.0, 400.0), ankle=(500.0, 400.0))


def test_torso_angle_vertical_and_horizontal():
    assert torso_angle_deg(np.array([0.0, 0.0]), np.array([0.0, 100.0])) == pytest.approx(0.0)
    assert torso_angle_deg(np.array([100.0, 0.0]), np.array([0.0, 0.0])) == pytest.approx(90.0)
    # Upside down (shoulders below hips, e.g. lying head-toward a ceiling camera) is 180,
    # and the direction of a sideways lean does not matter.
    assert torso_angle_deg(np.array([0.0, 100.0]), np.array([0.0, 0.0])) == pytest.approx(180.0)
    assert torso_angle_deg(np.array([-50.0, 0.0]), np.array([0.0, 50.0])) == pytest.approx(45.0)
    assert torso_angle_deg(np.array([50.0, 50.0]), np.array([0.0, 0.0])) == pytest.approx(135.0)


def test_torso_ref_does_not_learn_from_upside_down():
    ex = FeatureExtractor(Config())
    t = 0.0
    for _ in range(30):
        f = ex.update(standing(torso=100.0), t)
        t += 1 / FPS
    inverted = make_lms(shoulder=(320.0, 300.0), hip=(320.0, 260.0))  # 40 px, foreshortened
    for _ in range(30):
        f = ex.update(inverted, t)
        t += 1 / FPS
    assert f.torso_angle > 170
    assert f.torso_ref == pytest.approx(100.0)


def test_extractor_reports_angle():
    ex = FeatureExtractor(Config(ema_alpha=1.0))
    assert ex.update(standing(), 0.0).torso_angle == pytest.approx(0.0)
    assert ex.update(lying(), 1 / FPS).torso_angle == pytest.approx(90.0)


def test_hip_vel_positive_moving_down_negative_moving_up():
    ex = FeatureExtractor(Config(ema_alpha=1.0))
    ex.update(standing(0.0), 0.0)
    f = ex.update(standing(10.0), 1 / FPS)  # 10 px down in one frame, torso 100 px
    assert f.hip_vel == pytest.approx(10 / 100 * FPS)
    assert f.hip_vel_peak == pytest.approx(f.hip_vel)
    f = ex.update(standing(0.0), 2 / FPS)
    assert f.hip_vel < 0


def test_hip_vel_peak_expires_after_window():
    ex = FeatureExtractor(Config(ema_alpha=1.0, vel_window_s=0.5))
    ex.update(standing(0.0), 0.0)
    ex.update(standing(20.0), 1 / FPS)
    t = 1 / FPS
    while t < 1.0:
        t += 1 / FPS
        f = ex.update(standing(20.0), t)
    assert f.hip_vel_peak == pytest.approx(0.0)


def test_torso_ref_frozen_while_horizontal():
    ex = FeatureExtractor(Config())
    t = 0.0
    for _ in range(30):
        f = ex.update(standing(torso=100.0), t)
        t += 1 / FPS
    ref = f.torso_ref
    assert ref == pytest.approx(100.0)
    # Lying with a foreshortened torso must not move the reference.
    for torso in (60.0, 40.0, 150.0):
        for _ in range(30):
            f = ex.update(lying(torso=torso), t)
            t += 1 / FPS
            assert f.torso_ref == ref


def test_hip_height_standing_vs_ground():
    ex = FeatureExtractor(Config())
    f = ex.update(standing(), 0.0)
    assert f.hip_height == pytest.approx(1.8)
    f = ex.update(lying(), 1 / FPS)
    assert f.hip_height == pytest.approx(0.0)
    no_ankles = make_lms(shoulder=(320.0, 100.0), hip=(320.0, 200.0))
    assert ex.update(no_ankles, 2 / FPS).hip_height is None


def test_bbox_aspect_and_motion():
    ex = FeatureExtractor(Config(ema_alpha=1.0))
    f = ex.update(standing(), 0.0)
    assert f.bbox_aspect < 0.5
    assert f.motion == 0.0
    f = ex.update(standing(), 1 / FPS)
    assert f.motion == pytest.approx(0.0)
    f = ex.update(lying(), 2 / FPS)
    assert f.bbox_aspect > 1.2
    assert f.motion > 1.0


def test_pose_lost_returns_none_and_resets_velocity():
    ex = FeatureExtractor(Config(ema_alpha=1.0))
    ex.update(standing(0.0), 0.0)
    assert ex.update(None, 1 / FPS) is None
    # No velocity spike from the jump across the gap.
    f = ex.update(standing(50.0), 2 / FPS)
    assert f.hip_vel == 0.0


def lying_at(dx=0.0, noise=None):
    """Lying pose shifted by dx px, optionally with per-point wobble."""
    lms = lying()
    lms[:, 0] += dx
    if noise is not None:
        lms[:, :2] += noise
    return lms


def test_pose_wobble_on_a_still_person_reads_as_still():
    # ~1.5 px random wobble per frame at 30 fps: frame-to-frame that is ~0.45 torso/s,
    # above still_motion; over a 1 s window it cancels out.
    rng = np.random.default_rng(1)
    ex = FeatureExtractor(Config(ema_alpha=1.0))
    motions = []
    for i in range(90):
        f = ex.update(lying_at(noise=rng.normal(0, 1.5, (33, 2))), i / FPS)
        if i >= 30:
            motions.append(f.motion)
    assert max(motions) < 0.15


def test_steady_movement_reads_its_real_speed():
    # Sliding 3 px per frame = 90 px/s = 0.9 torso lengths/s (torso 100 px).
    ex = FeatureExtractor(Config(ema_alpha=1.0))
    for i in range(60):
        f = ex.update(lying_at(dx=3.0 * i), i / FPS)
    assert f.motion == pytest.approx(0.9, rel=0.05)


def test_motion_window_resets_when_pose_is_lost():
    ex = FeatureExtractor(Config(ema_alpha=1.0))
    for i in range(30):
        ex.update(lying_at(dx=0.0), i / FPS)
    ex.update(None, 30 / FPS)
    f = ex.update(lying_at(dx=200.0), 31 / FPS)  # jump across the gap must not count
    assert f.motion == 0.0
