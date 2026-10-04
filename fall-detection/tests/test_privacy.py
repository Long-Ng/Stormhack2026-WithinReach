import json
import urllib.request

import numpy as np

from overlay import PRIVACY_BG, render_privacy_frame


def standing_lms():
    lms = np.zeros((33, 4))
    pts = {0: (320, 100), 11: (290, 160), 12: (350, 160), 23: (300, 280), 24: (340, 280),
           25: (300, 360), 26: (340, 360), 27: (300, 440), 28: (340, 440)}
    for i, (x, y) in pts.items():
        lms[i] = (x, y, 0, 1.0)
    return lms


def test_privacy_frame_is_mostly_background_with_an_outline():
    f = render_privacy_frame((480, 640, 3), standing_lms(), 0.5)
    assert f.shape == (480, 640, 3)
    bg = np.all(f == PRIVACY_BG, axis=2).mean()
    assert 0.8 < bg < 1.0  # an outline was drawn, everything else is plain background


def test_privacy_frame_without_a_person():
    f = render_privacy_frame((480, 640, 3), None, 0.5)
    assert np.all(f == PRIVACY_BG, axis=2).mean() > 0.97  # only the small labels


def test_privacy_switch_over_http(tmp_path):
    from stream import Streamer
    s = Streamer(port=5094, events_dir=tmp_path)
    try:
        base = "http://127.0.0.1:5094/api/privacy"
        assert json.load(urllib.request.urlopen(base)) == {"on": False}
        req = urllib.request.Request(base + "?on=1", data=b"", method="POST")
        assert json.load(urllib.request.urlopen(req)) == {"on": True}
        assert s.privacy is True
        req = urllib.request.Request(base + "?on=0", data=b"", method="POST")
        assert json.load(urllib.request.urlopen(req)) == {"on": False}
    finally:
        s.close()


def test_clear_preroll_drops_frames_from_before_the_switch(tmp_path):
    import cv2
    from clips import ClipRecorder
    rec = ClipRecorder(tmp_path, pre_s=30, tail_s=0.5, max_after_s=5, max_fps=30)
    camera = np.full((120, 160, 3), (0, 0, 255), np.uint8)    # "camera" frames: red
    outline = np.full((120, 160, 3), (255, 0, 0), np.uint8)   # privacy frames: blue
    t = 0.0
    for _ in range(20):
        rec.add(camera, t, False); t += 0.05
    rec.clear_preroll()
    for _ in range(10):
        rec.add(outline, t, False); t += 0.05
    path = rec.trigger(1_700_000_000.0, t)
    for _ in range(20):
        rec.add(outline, t, False); t += 0.05
    rec.close()
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    assert frames
    assert all(f[..., 0].mean() > f[..., 2].mean() for f in frames)  # no red camera frames


def test_demo_switches_over_http(tmp_path):
    from stream import Streamer
    s = Streamer(port=5090, events_dir=tmp_path)
    try:
        base = "http://127.0.0.1:5090/api/demo"
        assert json.load(urllib.request.urlopen(base)) == {"graph": False, "skeleton": False}
        req = urllib.request.Request(base + "?graph=1&skeleton=1", data=b"", method="POST")
        assert json.load(urllib.request.urlopen(req)) == {"graph": True, "skeleton": True}
        assert s.demo == {"graph": True, "skeleton": True}
    finally:
        s.close()


def test_demo_graph_without_a_phone_says_so():
    from overlay import draw_phone_graph
    f = np.full((480, 640, 3), 120, np.uint8)
    draw_phone_graph(f, None)
    assert f[-45, -20].mean() < 120  # dimmed panel with the message


def test_fallen_page_is_served(tmp_path):
    from stream import Streamer
    s = Streamer(port=5089, events_dir=tmp_path)
    try:
        html = urllib.request.urlopen("http://127.0.0.1:5089/fallen?src=phone&t=1&g=2.5").read().decode()
        assert "Are you OK?" in html and 'id="okBtn"' in html and 'id="callBtn"' in html
    finally:
        s.close()
