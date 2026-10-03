"""Synthetic feature sequences at 30 fps through FallDetector."""

import pytest

from config import Config
from detector import FallDetector, State
from features import Features

FPS = 30.0

# Each phase is a dict of feature values held constant for its duration, or None (pose lost).
STAND = dict(angle=5.0, peak=0.0, aspect=0.4, hip_h=1.8, motion=0.1)
DROP = dict(angle=50.0, peak=3.0, aspect=0.9, hip_h=0.9, motion=3.0)
LYING = dict(angle=85.0, peak=0.0, aspect=2.0, hip_h=0.2, motion=0.05)
GET_UP = dict(angle=45.0, peak=0.0, aspect=0.9, hip_h=0.8, motion=1.5)


def feats(t, angle, peak, aspect, hip_h, motion):
    return Features(t=t, torso_angle=angle, torso_angle_raw=angle, hip_vel=peak,
                    hip_vel_peak=peak, bbox_aspect=aspect, hip_height=hip_h,
                    motion=motion, torso_len=100.0, torso_ref=100.0)


def run(phases, cfg=None):
    """phases: list of (seconds, values | None). Returns (detector, detections)."""
    det = FallDetector(cfg or Config())
    out = []
    frame = 0
    for duration, values in phases:
        for _ in range(round(duration * FPS)):
            t = frame / FPS
            d = det.update(None if values is None else feats(t, **values), t)
            if d is not None:
                out.append(d)
            frame += 1
    return det, out


def test_fall_then_still_emits_one_fall():
    det, events = run([(1, STAND), (0.3, DROP), (4, LYING)])
    assert [e.kind for e in events] == ["fall"]
    assert events[0].peak_hip_vel == pytest.approx(3.0)
    # Confirmed after ~confirm_s of stillness on the ground, not before.
    assert events[0].t == pytest.approx(1.3 + 3.0, abs=0.1)
    assert det.state is State.FALL_CONFIRMED


def test_sit_down_no_event():
    sit_down = dict(angle=20.0, peak=0.8, aspect=0.6, hip_h=1.0, motion=0.8)
    seated = dict(angle=10.0, peak=0.0, aspect=0.6, hip_h=0.8, motion=0.05)
    det, events = run([(1, STAND), (0.8, sit_down), (5, seated)])
    assert events == []
    assert det.state is State.UPRIGHT


def test_fast_crouch_and_stand_no_event():
    crouch = dict(angle=25.0, peak=2.5, aspect=0.8, hip_h=0.4, motion=2.0)
    det, events = run([(1, STAND), (0.5, crouch), (0.5, STAND | {"peak": 2.5}), (3, STAND)])
    assert events == []
    assert det.state is State.UPRIGHT


def test_slow_lie_down_no_fall_event():
    lowering = dict(angle=45.0, peak=0.4, aspect=0.9, hip_h=0.8, motion=0.4)
    det, events = run([(1, STAND), (2, lowering), (5, LYING)])
    assert events == []


def test_slow_lie_down_long_emits_prolonged_lying_once():
    lowering = dict(angle=45.0, peak=0.4, aspect=0.9, hip_h=0.8, motion=0.4)
    det, events = run([(1, STAND), (2, lowering), (20, LYING)])
    assert [e.kind for e in events] == ["prolonged_lying"]


def test_fall_then_get_up_no_event():
    det, events = run([(1, STAND), (0.3, DROP), (1.5, LYING), (0.5, GET_UP), (3, STAND)])
    assert events == []
    assert det.state is State.UPRIGHT


def test_pose_lost_on_ground_still_emits():
    det, events = run([(1, STAND), (0.3, DROP), (0.3, LYING), (5, None)])
    assert [e.kind for e in events] == ["fall"]


def test_pose_lost_briefly_does_not_count_as_still():
    # Lost for less than the grace period: the stillness timer holds, so no event yet.
    det, events = run([(1, STAND), (0.3, DROP), (1, LYING), (0.9, None), (1.5, LYING)])
    assert events == []
    assert det.state is State.ON_GROUND


def test_pose_lost_upright_stays_upright():
    det, events = run([(1, STAND), (10, None)])
    assert events == []
    assert det.state is State.UPRIGHT


def test_stays_down_30s_emits_once():
    det, events = run([(1, STAND), (0.3, DROP), (30, LYING)])
    assert len(events) == 1


def test_rearms_after_recovery():
    det, events = run([(1, STAND), (0.3, DROP), (4, LYING), (0.5, GET_UP), (2, STAND),
                       (0.3, DROP), (4, LYING)])
    assert [e.kind for e in events] == ["fall", "fall"]


def test_moving_on_ground_resets_stillness():
    writhing = LYING | {"motion": 1.0}
    det, events = run([(1, STAND), (0.3, DROP), (2, LYING), (1.0, writhing), (2, LYING)])
    assert events == []
    assert det.state is State.ON_GROUND
    assert det.confirm_remaining == pytest.approx(1.0, abs=0.1)


def test_brief_jitter_on_ground_does_not_reset_stillness():
    # One-frame twitches every second: still confirmed, only slightly later.
    twitch = LYING | {"motion": 1.0}
    phases = [(1, STAND), (0.3, DROP)]
    for _ in range(4):
        phases += [(29 / FPS, LYING), (1 / FPS, twitch)]
    det, events = run(phases)
    assert [e.kind for e in events] == ["fall"]


def test_jitter_longer_than_grace_resets_stillness():
    twitch = LYING | {"motion": 1.0}
    det, events = run([(1, STAND), (0.3, DROP), (2, LYING), (0.6, twitch), (2, LYING)],
                      Config(still_grace_s=0.5))
    assert events == []
