import numpy as np
import pytest

from config import Config
from smoothing import SkeletonStabilizer

FPS = 30.0


def pose(dx=0.0, vis=1.0):
    lms = np.zeros((33, 4))
    lms[:, 0] = 300 + 5 * np.arange(33) + dx
    lms[:, 1] = 200 + 3 * np.arange(33)
    lms[:, 3] = vis
    return lms


def frame_moves(seq):
    return np.mean([np.linalg.norm(b[:, :2] - a[:, :2], axis=1).mean() for a, b in zip(seq, seq[1:])])


def test_wobble_on_a_still_person_is_reduced():
    rng = np.random.default_rng(0)
    st = SkeletonStabilizer(Config())
    raw = [pose() + np.c_[rng.normal(0, 1.2, (33, 2)), np.zeros((33, 2))] for _ in range(90)]
    out = [st.update(r, i / FPS) for i, r in enumerate(raw)]
    assert frame_moves(out[30:]) < 0.4 * frame_moves(raw[30:])


def test_fast_movement_is_followed_closely():
    st = SkeletonStabilizer(Config())
    for i in range(30):
        st.update(pose(), i / FPS)
    for k in range(1, 16):  # 20 px/frame = 600 px/s, a fall-like speed
        out = st.update(pose(dx=20.0 * k), (30 + k) / FPS)
    lag = np.abs(out[:, 0] - pose(dx=20.0 * 15)[:, 0]).mean()
    assert lag < 25  # a little over one frame of movement


def test_short_dropout_holds_the_last_skeleton_then_clears():
    st = SkeletonStabilizer(Config(skel_hold_s=0.5))
    last = st.update(pose(), 0.0)
    assert np.allclose(st.update(None, 0.3), last)
    assert st.update(None, 0.7) is None


def test_joints_do_not_blink_around_the_threshold():
    st = SkeletonStabilizer(Config(min_visibility=0.5, skel_hide_visibility=0.3))
    seen = [st.update(pose(vis=v), i / FPS)[0, 3] > 0 for i, v in enumerate([0.6, 0.45, 0.4, 0.55, 0.35])]
    assert seen == [True] * 5  # stays shown while above 0.3
    assert st.update(pose(vis=0.2), 1.0)[0, 3] == 0  # hidden below 0.3
    assert st.update(pose(vis=0.45), 1.1)[0, 3] == 0  # and needs 0.5 to come back
    assert st.update(pose(vis=0.55), 1.2)[0, 3] > 0


def test_detection_inputs_are_not_modified():
    st = SkeletonStabilizer(Config())
    raw = pose()
    before = raw.copy()
    st.update(raw, 0.0); st.update(raw + 3, 0.1)
    assert np.array_equal(raw, before)
