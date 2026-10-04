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


def test_graph_data_for_dashboard():
    r = Reader(); w = Wearable(r, Config(imu_impact_ms2=20.0))
    now = time.perf_counter()
    w._last_t = now - 20
    for i in range(750):  # 15 s at 50 Hz, impact 5 s ago
        r.samples.append((now - 15 + i / 50, 25.0 if i == 500 else 0.5))
    w.poll()
    d = w.graph(10.0, now)
    assert d["available"] and d["threshold"] == 20.0
    assert all(0 <= age <= 10 for age, _ in d["samples"]) and len(d["samples"]) > 400
    assert len(d["impacts"]) == 1 and abs(d["impacts"][0][0] - 5) < 0.1


def test_graph_data_offline():
    d = Wearable(Reader(connected=False), Config()).graph()
    assert d["available"] is False and "offline" in d["reason"]


def test_shock_marks_fall_suspected_for_a_while():
    r = Reader(); w = Wearable(r, Config(imu_impact_ms2=20.0, imu_suspect_s=10.0))
    now = time.perf_counter()
    w._last_t = now - 6
    for i in range(250):
        r.samples.append((now - 5 + i / 50, 25.0 if i == 100 else 0.5))  # shock 3 s ago
    w.poll()
    g = w.graph(now=now)
    assert (g["suspect"]["age"], g["suspect"]["g"]) == (3.0, 2.5)
    assert w.graph(now=now + 8)["suspect"] is None  # 11 s later: cleared


def test_window_says_fall_suspected_after_a_shock():
    r = Reader(); w = Wearable(r, Config(imu_impact_ms2=20.0))
    now = time.perf_counter()
    w._last_t = now - 2
    r.samples.extend([(now - 1.0, 25.0), (now - 0.9, 0.3)])
    w.poll()
    frame = np.zeros((480, 640, 3), np.uint8)
    draw_wearable(frame, w)
    top = frame[50:100, 150:490]  # big red "FALL SUSPECTED" near the top centre
    assert ((top[..., 2] > 200) & (top[..., 1] < 80)).sum() > 200


def test_suspect_has_wall_clock_time_of_the_shock():
    r = Reader(); w = Wearable(r, Config(imu_impact_ms2=20.0))
    now = time.perf_counter()
    w._last_t = now - 2
    r.samples.extend([(now - 1.0, 25.0), (now - 0.9, 0.3)])
    w.poll()
    s = w.graph()["suspect"]
    assert abs(s["t"] - (time.time() - 1.0)) < 0.5  # about one second ago, as epoch seconds
