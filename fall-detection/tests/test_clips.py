from pathlib import Path

import cv2
import numpy as np

from clips import ClipRecorder, estimate_fps

FPS = 10.0


def frame(i: int) -> np.ndarray:
    return np.full((120, 160, 3), (i * 7) % 256, dtype=np.uint8)


def frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def feed(rec: ClipRecorder, start: int, stop: int, down: bool) -> None:
    for i in range(start, stop):
        rec.add(frame(i), i / FPS, down)


def test_estimate_fps():
    assert estimate_fps([0.0, 0.1, 0.2, 0.3]) == 10.0
    assert estimate_fps([1.0]) == 20.0  # too few frames: default


def test_clip_has_pre_roll_and_runs_until_recovery(tmp_path):
    rec = ClipRecorder(tmp_path, pre_s=3.0, tail_s=1.0, max_after_s=60.0)
    feed(rec, 0, 100, down=False)            # 10 s upright; only the last 3 s are kept
    path = rec.trigger(1_700_000_000.0, 10.0)
    feed(rec, 100, 150, down=True)           # 5 s on the ground
    feed(rec, 150, 170, down=False)          # back up: recording stops after 1 s
    assert not rec.recording
    rec.close()
    assert Path(path).suffix == ".mp4"
    # 31 pre-roll frames (t = 6.9..9.9 s), 50 down, 10 up before the tail ends it
    assert frame_count(path) == 31 + 50 + 10


def test_second_trigger_extends_the_clip(tmp_path):
    rec = ClipRecorder(tmp_path, pre_s=1.0, tail_s=1.0, max_after_s=60.0)
    feed(rec, 0, 20, down=False)
    first = rec.trigger(1_700_000_000.0, 2.0)
    feed(rec, 20, 30, down=True)
    assert rec.trigger(1_700_000_005.0, 3.0) == first
    rec.close()
    assert list(tmp_path.glob("*.mp4")) == [Path(first)]


def test_recording_is_capped(tmp_path):
    rec = ClipRecorder(tmp_path, pre_s=1.0, tail_s=1.0, max_after_s=2.0)
    feed(rec, 0, 10, down=False)
    rec.trigger(1_700_000_000.0, 1.0)
    feed(rec, 10, 100, down=True)            # still down, but the cap ends it
    assert not rec.recording
    rec.close()


def test_clip_is_downscaled_and_thinned(tmp_path):
    rec = ClipRecorder(tmp_path, pre_s=10.0, tail_s=10.0, max_after_s=60.0, max_width=80, max_fps=5.0)
    path = None
    for i in range(30):                      # 3 s of 120x160 frames at 10 fps
        rec.add(frame(i), i / FPS, down=False)
        if i == 9:
            path = rec.trigger(1_700_000_000.0, 0.9)
    rec.close()
    cap = cv2.VideoCapture(path)
    assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (80, 60)
    cap.release()
    assert frame_count(path) == 15           # every other frame
