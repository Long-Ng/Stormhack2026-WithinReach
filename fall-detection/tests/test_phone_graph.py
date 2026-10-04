import collections
import time

import numpy as np

from config import Config
from imu import Wearable
from overlay import draw_wearable


class Reader:
    def __init__(self, connected=True):
        self.connected = connected
        self.samples = collections.deque()


def test_graph_draws_curve_and_impact_marker():
    r = Reader(); w = Wearable(r, Config(imu_impact_ms2=20.0))
    now = time.perf_counter()
    w._last_t = now - 6
    for i in range(250):
        t = now - 5 + i / 50
        r.samples.append((t, 25.0 if i == 100 else 0.5))
    assert len(w.poll()) == 1 and len(w.impact_log) == 1
    frame = np.zeros((480, 640, 3), np.uint8)
    draw_wearable(frame, w)
    assert (frame[..., 1] > 150).sum() > 200          # green curve drawn
    assert ((frame[..., 2] > 200) & (frame[..., 1] < 80)).sum() > 20  # red impact dot


def test_graph_offline_does_not_crash():
    w = Wearable(Reader(connected=False), Config())
    frame = np.full((480, 640, 3), 100, np.uint8)
    draw_wearable(frame, w)
    assert frame[-40, -20].mean() < 100  # panel dimmed
