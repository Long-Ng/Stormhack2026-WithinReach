import numpy as np

import camera
from config import Config

FLAT = np.full((48, 64, 3), (0, 135, 0), dtype=np.uint8)  # DroidCam "no phone" green
REAL = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)


def noisy():
    return REAL ^ np.random.default_rng().integers(0, 2, REAL.shape, dtype=np.uint8)


CARD = np.zeros((48, 64, 3), dtype=np.uint8)
CARD[20:28, 10:54] = 255  # "Start DroidCam" text card


class FakeCap:
    """frame: an array (repeated), a callable (called per read), or a list (played in order,
    last one repeated)."""

    def __init__(self, frame, opened=True):
        self.frame, self.opened, self.released, self.i = frame, opened, False, 0

    def isOpened(self):
        return self.opened

    def read(self):
        f = self.frame
        if callable(f):
            return True, f()
        if isinstance(f, list):
            self.i += 1
            return True, f[min(self.i - 1, len(f) - 1)]
        return True, f

    def release(self):
        self.released = True


def test_find_droidcam_by_name():
    assert camera.find_droidcam(["Integrated Camera", "DroidCam Video"]) == 1
    assert camera.find_droidcam(["Integrated Camera"]) is None


def test_placeholder_frames_are_not_live():
    cfg = Config(droidcam_probe_s=0.05)
    assert not camera.is_live(FakeCap(FLAT), cfg)
    assert not camera.is_live(FakeCap([CARD, FLAT]), cfg)  # what DroidCam sends with no phone
    assert not camera.is_live(FakeCap(CARD), cfg)  # a static card repeated
    assert camera.is_live(FakeCap(noisy), cfg)


def test_frame_spread_ignores_solid_colour():
    assert camera.frame_spread(FLAT) == 0.0
    assert camera.frame_spread(REAL) > 50


def patch_cameras(monkeypatch, droidcam_frame):
    caps = {0: FakeCap(noisy), 1: FakeCap(droidcam_frame)}
    monkeypatch.setattr(camera, "list_cameras", lambda: ["Integrated Camera", "DroidCam Video"])
    monkeypatch.setattr(camera, "open_camera", lambda i, cfg: caps[i])
    return caps


def test_streaming_droidcam_is_preferred(monkeypatch):
    caps = patch_cameras(monkeypatch, noisy)
    src = camera.open_source(None, Config(droidcam_probe_s=0.05))
    assert src.cap is caps[1] and "DroidCam" in src.label and not src.is_file


def test_idle_droidcam_falls_back_to_camera_index(monkeypatch):
    caps = patch_cameras(monkeypatch, [CARD, FLAT])
    src = camera.open_source(None, Config(droidcam_probe_s=0.05))
    assert src.cap is caps[0]
    assert caps[1].released


def test_prefer_droidcam_off_uses_camera_index(monkeypatch):
    caps = patch_cameras(monkeypatch, noisy)
    src = camera.open_source(None, Config(prefer_droidcam=False))
    assert src.cap is caps[0]


def test_explicit_index_skips_droidcam_search(monkeypatch):
    caps = patch_cameras(monkeypatch, noisy)
    assert camera.open_source("0", Config()).cap is caps[0]


def test_url_is_live_stream_and_path_is_file(monkeypatch):
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda s: FakeCap(REAL))
    assert not camera.open_source("http://192.168.1.5:4747/video", Config()).is_file
    assert camera.open_source("clip.mp4", Config()).is_file
