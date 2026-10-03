"""Camera + wearable: impacts and phone stillness alongside synthetic camera features."""

import pytest

from config import Config
from detector import FallDetector, State
from imu import Wearable
from tests.test_detector import DROP, FPS, LYING, STAND, feats

STILL_PHONE = "still"  # phone_still grows from the impact on


def run(phases, impact_at=None, cfg=None, phone=None):
    """phases: (seconds, values | None). impact_at: stream time of a wearable impact.
    phone: None (no wearable) or "still" (phone still since the impact)."""
    det = FallDetector(cfg or Config())
    out, frame = [], 0
    for duration, values in phases:
        for _ in range(round(duration * FPS)):
            t = frame / FPS
            if impact_at is not None and abs(t - impact_at) < 0.5 / FPS:
                det.add_impact(t)
            still = None
            if phone == STILL_PHONE:
                still = max(t - impact_at, 0.0) if impact_at is not None and t >= impact_at else 0.0
            d = det.update(None if values is None else feats(t, **values), t, still)
            if d is not None:
                out.append(d)
            frame += 1
    return det, out


def test_impact_at_landing_confirms_immediately():
    # Drop at 1.0-1.3 s, impact at landing 1.3 s: confirmed on the ground, no 3 s wait.
    det, events = run([(1, STAND), (0.3, DROP), (4, LYING)], impact_at=1.3)
    assert [(e.kind, e.sensor) for e in events] == [("fall", True)]
    assert events[0].t < 1.6


def test_no_impact_keeps_camera_timing():
    det, events = run([(1, STAND), (0.3, DROP), (4, LYING)])
    assert [(e.kind, e.sensor) for e in events] == [("fall", False)]
    assert events[0].t == pytest.approx(4.3, abs=0.1)


def test_unrelated_old_impact_does_not_confirm():
    det, events = run([(5, STAND), (0.3, DROP), (4, LYING)], impact_at=1.0)
    assert [(e.kind, e.sensor) for e in events] == [("fall", False)]


def test_out_of_frame_impact_then_still_is_sensor_fall():
    det, events = run([(1, STAND), (8, None)], impact_at=2.0, phone=STILL_PHONE)
    assert [e.kind for e in events] == ["sensor_fall"]
    assert events[0].t == pytest.approx(2.0 + 3.0, abs=0.1)
    assert det.state is State.FALL_CONFIRMED


def test_dropped_phone_with_person_upright_is_ignored():
    det, events = run([(1, STAND), (10, STAND)], impact_at=2.0, phone=STILL_PHONE)
    assert events == []


def test_impact_but_phone_keeps_moving_is_not_sensor_fall():
    det, events = run([(1, STAND), (8, None)], impact_at=2.0, phone=None)
    assert events == []


def test_one_event_per_fall_with_sensor():
    det, events = run([(1, STAND), (0.3, DROP), (30, LYING)], impact_at=1.3, phone=STILL_PHONE)
    assert len(events) == 1


class FakeReader:
    connected = True

    def __init__(self):
        self.samples = []


def test_wearable_impacts_and_stillness():
    cfg = Config(imu_impact_ms2=20.0, imu_still_ms2=1.5)
    r = FakeReader()
    w = Wearable(r, cfg)
    t0 = w._last_t
    r.samples += [(t0 + 0.02, 0.5), (t0 + 0.04, 25.0), (t0 + 0.06, 3.0), (t0 + 0.08, 0.4)]
    hits = w.poll()
    assert [h.peak for h in hits] == [25.0]
    assert w.still_for(t0 + 1.08) == pytest.approx(1.0)
    r.samples.append((t0 + 1.2, 5.0))
    assert w.poll() == []
    assert w.still_for(t0 + 2.0) == 0.0
