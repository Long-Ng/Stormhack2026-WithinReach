import json
import urllib.request

from alerts import AlertManager
from config import Config
from events import FallEvent
from gemini import GeminiClient


def test_usage_is_totalled_from_reply_metadata(capsys):
    c = GeminiClient("key", "model-a")
    c.record_usage("model-a", {"usageMetadata": {"promptTokenCount": 15000, "candidatesTokenCount": 120}})
    c.record_usage("model-a", {"usageMetadata": {"promptTokenCount": 1500, "candidatesTokenCount": 80,
                                                 "thoughtsTokenCount": 20}})
    c.record_usage("model-a", {})  # a reply without metadata still counts as a call
    assert c.usage == {"calls": 3, "input_tokens": 16500, "output_tokens": 220}
    assert "since start: 3 calls, 16720 tokens" in capsys.readouterr().out


def manager():
    cfg = Config(ntfy_person_topic="p", ntfy_monitor_topic="m")
    m = AlertManager(cfg, "http://127.0.0.1:5000", lambda msg: None)
    m.send(FallEvent(timestamp=1_791_066_000.0, kind="fall", peak_hip_vel=3.0, torso_angle=90.0,
                     snapshot_path="D:/x/events/20261003-153435-047.jpg", stream_t=1.0))
    return m


def test_marking_handled_stops_gemini_and_unmarking_resumes():
    m = manager()
    inc_id = m.last_incident_id
    assert m.is_open(inc_id)
    assert m.set_handled("20261003-153435-047.jpg", True)
    assert not m.is_open(inc_id)
    assert m.set_handled("20261003-153435-047.jpg", False)
    assert m.is_open(inc_id)
    assert not m.set_handled("unknown.jpg", True)


def test_handled_route_over_http(tmp_path):
    from stream import Streamer
    m = manager()
    s = Streamer(port=5092, events_dir=tmp_path)
    try:
        for path, h in m.routes().items():
            s.add_route(path, h)
        url = "http://127.0.0.1:5092/api/handled?event=20261003-153435-047.jpg&on=1"
        req = urllib.request.Request(url, data=b"", method="POST")
        assert json.load(urllib.request.urlopen(req)) == {"ok": True}
        assert not m.is_open(m.last_incident_id)
    finally:
        s.close()
