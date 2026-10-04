import json
import urllib.error
import urllib.request

import pytest

from alerts import HELP, NO_REPLY, OK, WAITING, AlertManager
from config import Config
from events import FallEvent

T0 = 1_791_066_000.0


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


def make(**cfg_kw):
    cfg = Config(ntfy_person_topic="person-x", ntfy_monitor_topic="monitor-x",
                 reply_timeout_s=30.0, **cfg_kw)
    sent, clock = [], Clock()
    return AlertManager(cfg, "http://192.168.1.9:5000", sent.append, clock), sent, clock


def fall(t=T0, snapshot="D:/x/events/20261003-153435-047.jpg"):
    return FallEvent(timestamp=t, kind="fall", peak_hip_vel=3.0, torso_angle=95.0,
                     snapshot_path=snapshot, stream_t=12.0)


def only_incident(m):
    (inc,) = m.incidents.values()
    return inc


def test_fall_notifies_person_with_reply_buttons():
    m, sent, _ = make(call_number="+15550001111")
    m.send(fall())
    (msg,) = sent
    assert msg["topic"] == "person-x" and msg["priority"] == 5
    inc = only_incident(m)
    urls = [a["url"] for a in msg["actions"]]
    assert urls == [f"http://192.168.1.9:5000/respond?id={inc.id}&answer=ok",
                    f"http://192.168.1.9:5000/respond?id={inc.id}&answer=help",
                    "tel:+15550001111"]
    assert msg["click"] == f"http://192.168.1.9:5000/person?id={inc.id}"
    assert inc.status == WAITING


def test_no_call_button_without_number():
    m, sent, _ = make()
    m.send(fall())
    assert [a["label"] for a in sent[0]["actions"]] == ["I'm OK", "I need help"]


def test_no_reply_escalates_once_after_timeout():
    m, sent, clock = make()
    m.send(fall())
    clock.t = T0 + 29; m.tick()
    assert len(sent) == 1
    clock.t = T0 + 30; m.tick(); m.tick()
    assert [s["topic"] for s in sent] == ["person-x", "monitor-x"]
    assert "did not reply" in sent[1]["message"]
    assert sent[1]["attach"].endswith("/events/20261003-153435-047.jpg")
    assert only_incident(m).status == NO_REPLY


def test_ok_reply_cancels_escalation():
    m, sent, clock = make()
    m.send(fall())
    assert m.respond(only_incident(m).id, OK) == "Glad you're OK"
    clock.t = T0 + 60; m.tick()
    assert len(sent) == 1 and only_incident(m).status == OK


def test_help_reply_alerts_monitor_immediately_and_once():
    m, sent, clock = make()
    m.send(fall())
    inc_id = only_incident(m).id
    m.respond(inc_id, HELP); m.respond(inc_id, HELP)
    clock.t = T0 + 60; m.tick()
    assert [s["topic"] for s in sent] == ["person-x", "monitor-x"]
    assert "asked for help" in sent[1]["message"]


def test_ok_after_escalation_tells_monitor_it_is_resolved():
    m, sent, clock = make()
    m.send(fall())
    clock.t = T0 + 31; m.tick()
    m.respond(only_incident(m).id, OK)
    assert sent[-1]["topic"] == "monitor-x" and sent[-1]["title"].startswith("Resolved")


def test_bad_replies_are_rejected():
    m, sent, _ = make()
    m.send(fall())
    assert m.respond("nope", OK) == "unknown incident"
    assert m.respond(only_incident(m).id, "maybe") == "unknown answer"
    assert only_incident(m).status == WAITING


def test_two_falls_in_the_same_second_get_separate_incidents():
    m, sent, _ = make()
    m.send(fall()); m.send(fall())
    assert len(m.incidents) == 2


def test_routes_over_real_server(tmp_path):
    from stream import Streamer
    m, sent, _ = make(call_number="+15550001111")
    s = Streamer(port=5098, events_dir=tmp_path)
    try:
        for path, h in m.routes().items():
            s.add_route(path, h)
        m.send(fall())
        inc_id = only_incident(m).id
        base = "http://127.0.0.1:5098"
        j = json.load(urllib.request.urlopen(base + "/incidents.json"))
        assert j["incidents"][0]["status"] == WAITING and j["reply_timeout_s"] == 30.0
        req = urllib.request.Request(f"{base}/respond?id={inc_id}&answer=ok", data=b"", method="POST")
        assert urllib.request.urlopen(req).read() == b"Glad you're OK"
        assert only_incident(m).status == OK
        page = urllib.request.urlopen(f"{base}/person?id={inc_id}").read().decode()
        assert "tel:+15550001111" in page and "{{CALL_NUMBER}}" not in page
        assert urllib.request.urlopen(base + "/events.json").status == 200  # old routes intact
    finally:
        s.close()


def test_quick_message_notifies_monitor_and_marks_help():
    m, sent, _ = make()
    m.send(fall())
    inc = only_incident(m)
    assert m.send_message("cant_get_up", inc.id) == "Sent"
    assert sent[-1]["topic"] == "monitor-x" and "I can't get up" in sent[-1]["message"]
    assert inc.status == HELP
    assert m.send_message("nonsense", inc.id) == "unknown message"
    assert [x["text"] for x in m.messages] == ["I can't get up"]


def test_help_message_stops_the_no_reply_escalation():
    m, sent, clock = make()
    m.send(fall())
    m.send_message("hurt", only_incident(m).id)
    clock.t = T0 + 60; m.tick()
    assert len(sent) == 2  # person prompt + the message; no extra "did not reply"


def test_voice_message_saved_served_and_announced(tmp_path):
    m, sent, _ = make(events_dir=str(tmp_path))
    audio = b"\x1aE\xdf\xa3" + b"x" * 500  # webm-ish bytes
    assert m.save_voice(audio, "webm") == "Sent"
    (f,) = (tmp_path / "voice").iterdir()
    assert f.read_bytes() == audio
    assert sent[-1]["click"] == f"http://192.168.1.9:5000/voice?f={f.name}"
    code, ctype, body = m._route_voice({"f": f.name})
    assert (code, ctype, body) == (200, "audio/webm", audio)


def test_voice_rejects_bad_input_and_path_tricks(tmp_path):
    m, sent, _ = make(events_dir=str(tmp_path))
    assert m.save_voice(b"x" * 500, "exe") == "unsupported audio type"
    assert m.save_voice(b"", "webm") == "recording was empty"
    (tmp_path / "secret.webm").write_bytes(b"x" * 500)
    assert m._route_voice({"f": "../secret.webm"})[0] == 404
    assert sent == []


def test_voice_upload_over_real_server(tmp_path):
    from stream import MAX_UPLOAD_BYTES, Streamer
    m, sent, _ = make(events_dir=str(tmp_path))
    s = Streamer(port=5096, events_dir=tmp_path)
    try:
        for path, h in m.routes().items():
            s.add_route(path, h)
        for path, h in m.upload_routes().items():
            s.add_route(path, h, body=True)
        base = "http://127.0.0.1:5096"
        audio = b"\x1aE\xdf\xa3" + b"y" * 2000
        req = urllib.request.Request(base + "/talk?ext=webm", data=audio, method="POST")
        assert urllib.request.urlopen(req).read() == b"Sent"
        name = sent[-1]["click"].split("f=")[1]
        assert urllib.request.urlopen(f"{base}/voice?f={name}").read() == audio
        req = urllib.request.Request(base + "/message?key=call_me", data=b"", method="POST")
        assert urllib.request.urlopen(req).read() == b"Sent"
        j = json.load(urllib.request.urlopen(base + "/messages.json"))
        assert [x["kind"] for x in j["messages"]] == ["voice", "text"]
        big = urllib.request.Request(base + "/talk?ext=webm", data=b"z" * (MAX_UPLOAD_BYTES + 1), method="POST")
        # Refused without reading the body: the client sees a 413 or the connection drop.
        with pytest.raises((urllib.error.HTTPError, ConnectionError)) as e:
            urllib.request.urlopen(big)
        if isinstance(e.value, urllib.error.HTTPError):
            assert e.value.code == 413
        assert len(list((tmp_path / "voice").iterdir())) == 1  # nothing saved for it
    finally:
        s.close()
