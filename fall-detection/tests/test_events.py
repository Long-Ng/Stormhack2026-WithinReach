import json
from pathlib import Path

import cv2
import numpy as np

from config import Config
from detector import Detection
from events import ConsoleSink, FallEvent, FileSink, dispatch, save_snapshot
from main import handle_detection
from tests.test_detector import DROP, LYING, STAND, run

FRAME = np.full((120, 160, 3), 80, dtype=np.uint8)


def event(**kw):
    base = dict(timestamp=1_700_000_000.0, kind="fall", peak_hip_vel=3.0,
                torso_angle=85.0, snapshot_path=None, stream_t=4.3)
    return FallEvent(**base | kw)


def test_file_sink_appends_one_json_line_per_event(tmp_path):
    sink = FileSink(tmp_path / "events")
    sink.send(event())
    sink.send(event(kind="prolonged_lying"))
    lines = (tmp_path / "events" / "events.jsonl").read_text().splitlines()
    assert [json.loads(l)["kind"] for l in lines] == ["fall", "prolonged_lying"]
    assert json.loads(lines[0])["peak_hip_vel"] == 3.0


def test_save_snapshot_writes_readable_jpg(tmp_path):
    path = save_snapshot(FRAME, tmp_path, 1_700_000_000.123)
    assert path.endswith(".jpg")
    assert cv2.imread(path).shape == FRAME.shape


def test_console_sink_prints_kind(capsys):
    ConsoleSink().send(event())
    assert "FALL" in capsys.readouterr().out


def test_failing_sink_does_not_stop_others(tmp_path, capsys):
    class Broken:
        def send(self, event):
            raise RuntimeError("network down")

    dispatch(event(), [Broken(), FileSink(tmp_path)])
    assert (tmp_path / "events.jsonl").exists()
    assert "Broken failed" in capsys.readouterr().err


def test_from_detection_copies_fields():
    e = FallEvent.from_detection(Detection(t=4.3, kind="fall", peak_hip_vel=2.8, torso_angle=80.0),
                                 timestamp=1.0, snapshot_path="x.jpg")
    assert (e.kind, e.peak_hip_vel, e.torso_angle, e.stream_t, e.snapshot_path) == \
        ("fall", 2.8, 80.0, 4.3, "x.jpg")


def test_staged_fall_writes_one_line_and_one_snapshot(tmp_path, capsys):
    """Acceptance: fall + 3 s still -> one console line, one JSONL line, one snapshot."""
    cfg = Config(events_dir=str(tmp_path))
    sinks = [ConsoleSink(), FileSink(cfg.events_dir)]
    _, detections = run([(1, STAND), (0.3, DROP), (10, LYING)], cfg)
    for d in detections:
        handle_detection(d, FRAME, None, cfg, sinks)

    assert capsys.readouterr().out.count("FALL") == 1
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert list(tmp_path.glob("*.jpg")) == [Path(rec["snapshot_path"])]
