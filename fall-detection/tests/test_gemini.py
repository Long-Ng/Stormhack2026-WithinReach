import json
import threading

import numpy as np

from alerts import HELP, OK, AlertManager
from config import Config
from events import FallEvent
from gemini import (HIGH, LOW, MEDIUM, URGENT, FallAnalyst, FallReport, FrameHistory,
                    GeminiClient, guidance, report_dict)


def report(**kw):
    base = dict(is_fall=True, fall_type="backward", position="on_back", movement="small_movements",
                head_struck_likely="no", description="She fell backwards and is lying on her back.")
    return FallReport(**base | kw)


# --- guidance table ---------------------------------------------------------------------
def test_collapse_is_urgent():
    level, text = guidance(report(fall_type="collapse"), 0)
    assert level == URGENT and "Call 911 now" in text


def test_still_for_five_minutes_is_urgent():
    assert guidance(report(movement="still"), 2)[0] == MEDIUM
    level, text = guidance(report(movement="still"), 5)
    assert level == URGENT and "5+ minutes" in text


def test_head_strike_and_sideways_are_high_and_most_urgent_comes_first():
    level, text = guidance(report(fall_type="sideways", head_struck_likely="yes"), 0)
    assert level == HIGH and text.startswith("Their head may have hit")


def test_up_or_not_a_fall_is_low():
    assert guidance(report(movement="up_and_moving", position="standing"), 0)[0] == LOW
    up_after_head_strike = report(movement="up_and_moving", head_struck_likely="yes")
    assert guidance(up_after_head_strike, 0)[0] == HIGH  # a head strike still matters
    assert guidance(report(is_fall=False, fall_type="not_a_fall"), 0)[0] == LOW


def test_emergency_number_is_configurable():
    assert "Call 112 now" in guidance(report(fall_type="collapse"), 0, emergency="112")[1]


# --- parsing and the request ---------------------------------------------------------------
def test_unknown_values_fall_back_to_the_unknown_category():
    r = FallReport.from_json({"is_fall": True, "fall_type": "Cartwheel", "position": "ON_BACK",
                              "movement": "", "head_struck_likely": "maybe", "description": " x "})
    assert (r.fall_type, r.position, r.movement, r.head_struck_likely, r.description) == \
        ("unclear", "on_back", "not_visible", "unclear", "x")


def test_request_has_prompt_timed_images_and_schema():
    frame = np.zeros((720, 1280, 3), np.uint8)
    body = GeminiClient("k", "m").request_body("look", [(-2.0, frame), (0.0, frame)])
    parts = body["contents"][0]["parts"]
    assert parts[0] == {"text": "look"}
    assert [p["text"] for p in parts[1::2]] == ["t = -2.0 s", "t = +0.0 s"]
    assert all(p["inline_data"]["mime_type"] == "image/jpeg" for p in parts[2::2])
    assert body["generationConfig"]["response_mime_type"] == "application/json"
    assert "fall_type" in body["generationConfig"]["response_schema"]["required"]


def test_frame_history_keeps_recent_frames_spread_evenly():
    h = FrameHistory(keep_s=12.0, every_s=0.5)
    for i in range(100):  # 10 s at 10 fps -> one kept every 0.5 s
        h.add(np.full((48, 64, 3), i, np.uint8), i / 10)
    frames = h.recent(n=5, span_s=4.0, ref_t=9.5)  # last kept frame is at 9.5 s
    assert len(frames) == 5
    assert [round(t, 1) for t, _ in frames] == [-4.0, -3.0, -2.0, -1.0, 0.0]


# --- schedule ----------------------------------------------------------------------------
class FakeClient:
    def __init__(self):
        self.calls = []
        self.done = threading.Semaphore(0)

    def analyze(self, frames, minutes=None):
        self.calls.append(minutes)
        return report(movement="still")


def analyst(is_open=lambda _id: True):
    client, got = FakeClient(), []

    def on_report(inc_id, r, minutes):
        got.append((inc_id, minutes))
        client.done.release()
    a = FallAnalyst(client, on_report, is_open=is_open, update_s=300.0, max_updates=2)
    h = FrameHistory()
    for i in range(20):
        h.add(np.zeros((48, 64, 3), np.uint8), 100 + i / 2)
    return a, h, client, got


def test_first_look_then_updates_every_interval_until_the_cap():
    a, h, client, got = analyst()
    a.start("inc1", h, 110.0)
    for t in (200.0, 410.0, 720.0, 1100.0):
        a.tick(h, t)
    for _ in range(3):
        assert client.done.acquire(timeout=5)
    assert [(i, round(m)) for i, m in got] == [("inc1", 0), ("inc1", 5), ("inc1", 10)]
    assert a.watches == {}  # capped at max_updates


def test_updates_stop_when_answered_ok_or_back_up():
    open_ = {"inc1": True}
    a, h, client, got = analyst(is_open=lambda i: open_[i])
    a.start("inc1", h, 110.0)
    open_["inc1"] = False
    a.tick(h, 500.0)
    assert a.watches == {}
    a.start("inc1", h, 110.0)
    a.stop()
    assert a.watches == {}


# --- alerts hook ---------------------------------------------------------------------------
def test_reports_reach_the_monitor_once_alerted():
    cfg = Config(ntfy_person_topic="person-x", ntfy_monitor_topic="monitor-x")
    sent = []
    m = AlertManager(cfg, "http://pc:5000", sent.append, clock=lambda: 1_700_000_000.0)
    m.send(FallEvent(timestamp=1_700_000_000.0, kind="fall", peak_hip_vel=3, torso_angle=90,
                     snapshot_path=None, stream_t=1))
    inc_id = m.last_incident_id
    m.add_report(inc_id, report_dict(report(fall_type="collapse"), 0.0))
    assert len(sent) == 1  # monitor not alerted yet: stored only
    m.respond(inc_id, HELP)
    assert "fainted" in sent[-1]["message"] and "Automated guidance" in sent[-1]["message"]
    m.add_report(inc_id, report_dict(report(movement="still"), 5.0))
    assert sent[-1]["title"].startswith("Update, 5 min") and sent[-1]["priority"] == 5
    assert json.loads(json.dumps(m.incidents[inc_id].reports))  # served in /incidents.json
    m.respond(inc_id, OK)
    assert not m.is_open(inc_id)
