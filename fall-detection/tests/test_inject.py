import numpy as np
import pytest

from inject import ClipInjector, find_clips


class FakeCap:
    def __init__(self, n, fps=20.0, opened=True):
        self.n, self.fps, self.opened, self.i, self.released = n, fps, opened, 0, False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return self.fps

    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        return True, np.full((4, 4, 3), self.i, np.uint8)

    def release(self):
        self.released = True


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.t += d


def make(tmp_path, names=("a.mp4",), n=3, fps=20.0):
    for name in names:
        (tmp_path / name).write_bytes(b"x")
    clock, opened = Clock(), []

    def opener(path):
        opened.append(path)
        return FakeCap(n, fps)

    target = tmp_path if len(names) > 1 else tmp_path / names[0]
    return ClipInjector(target, opener=opener, clock=clock, sleep=clock.sleep), clock, opened


def test_plays_paced_then_returns_to_live(tmp_path):
    inj, clock, _ = make(tmp_path, n=3, fps=20.0)
    assert not inj.active
    assert inj.toggle() == "playing demo clip a.mp4" and inj.active
    frames = [inj.read() for _ in range(3)]
    assert all(ok for ok, _ in frames)
    assert clock.t == pytest.approx(2 / 20.0)  # paced at the clip's own 20 fps
    assert inj.read() == (False, None) and not inj.active  # ended -> back to live


def test_pressing_again_stops_early(tmp_path):
    inj, _, _ = make(tmp_path, n=100)
    inj.toggle(); inj.read()
    cap = inj.cap
    assert inj.toggle() == "back to the live camera"
    assert not inj.active and cap.released


def test_folder_cycles_through_clips(tmp_path):
    inj, _, opened = make(tmp_path, names=("b.mp4", "a.avi", "notes.txt"))
    assert [c.name for c in inj.clips] == ["a.avi", "b.mp4"]
    for _ in range(3):
        inj.toggle(); inj.toggle()
    assert [p.split("\\")[-1].split("/")[-1] for p in opened] == ["a.avi", "b.mp4", "a.avi"]


def test_missing_path_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        ClipInjector(tmp_path / "nope.mp4")
    assert find_clips(tmp_path) == []


def test_elapsed_follows_the_clip_timeline_not_the_wall_clock(tmp_path):
    inj, clock, _ = make(tmp_path, n=5, fps=20.0)
    inj.toggle()
    for k in range(4):
        inj.read()
        clock.t += 0.5  # processing is slow: wall time races ahead
        assert inj.elapsed_ms == pytest.approx(k * 50.0)  # clip time stays at 20 fps
