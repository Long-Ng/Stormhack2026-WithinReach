import threading

import numpy as np

from fake_gemini import FakeGeminiClient
from gemini import FallAnalyst, FrameHistory, report_dict


def history():
    h = FrameHistory()
    for i in range(120):
        h.add(np.zeros((48, 64, 3), np.uint8), i / 10)
    return h


def run_demo(ticks, speed_at=lambda t: 1.0):
    """Tick the analyst at the given stream times; return the reports in order."""
    got, done = [], threading.Semaphore(0)

    def on_report(inc_id, r, minutes, still):
        got.append(report_dict(r, minutes, still_minutes=still))
        done.release()
    a = FallAnalyst(FakeGeminiClient(latency_s=0), on_report, update_s=60.0, max_updates=60)
    h = history()
    a.start("inc1", h, 12.0)
    assert done.acquire(timeout=10)  # first look
    for t in ticks:
        a.speed = speed_at(t)
        before = len(got)
        a.tick(h, t)
        if len(a.watches) and a.watches["inc1"].updates > len(got) - 1:
            assert done.acquire(timeout=10)
        assert len(got) >= before
    return got


def test_the_story_escalates_to_call_911():
    # Normal speed: one check a minute for 10 minutes.
    reps = run_demo([12.0 + 60 * m for m in range(1, 11)])
    assert [round(r["minutes"]) for r in reps] == list(range(11))
    assert reps[0]["fall_type"] == "sideways" and reps[0]["urgency"] == "high"
    assert reps[1]["movement"] == "trying_to_get_up"
    assert [r["movement"] for r in reps[4:]] == ["still"] * 7
    assert reps[8]["urgency"] != "urgent" and reps[9]["urgency"] == "urgent"  # 5 still minutes
    assert "Call 911" in reps[9]["guidance"]


def test_fast_forward_gives_a_minute_every_two_seconds():
    reps = run_demo([12.0 + 2 * k for k in range(1, 11)], speed_at=lambda t: 30.0)
    assert [round(r["minutes"]) for r in reps] == list(range(11))
    assert reps[-1]["urgency"] == "urgent"  # the whole story in 20 s


def test_switching_speed_mid_way_does_not_jump_the_timeline():
    # 3 minutes at normal speed, then fast-forward: minutes carry on from 3.
    ticks = [72.0, 132.0, 192.0, 194.0, 196.0, 198.0]
    reps = run_demo(ticks, speed_at=lambda t: 1.0 if t <= 192.0 else 30.0)
    assert [round(r["minutes"]) for r in reps] == [0, 1, 2, 3, 4, 5, 6]


def test_turning_fast_forward_on_does_not_repeat_minute_zero():
    # Fall at 12 s; fast-forward from 17 s, so 17 s counts as 2.5 "minutes" in.
    minutes = [round(r["minutes"]) for r in run_demo([17.0, 19.0, 21.0], speed_at=lambda t: 30.0)]
    assert minutes[0] == 0 and minutes[1] >= 1        # first fast check is not minute 0 again
    assert len(set(minutes)) == len(minutes)          # no minute reported twice
