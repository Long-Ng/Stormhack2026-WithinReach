import cv2
import numpy as np

from inject import Injector

FPS = 10.0


def make_clip(path, n=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), i * 20, np.uint8))
    w.release()
    return str(path)


class Clock:
    """Fake live clock: sleep() advances it instead of waiting."""

    def __init__(self, t=100.0):
        self.t = t

    def sleep(self, s):
        self.t += s


def play(inj, clock, step=0.0):
    """Read until the clip ends; `step` is the processing time per frame."""
    out = []
    while (r := inj.read(clock.t)) is not None:
        out.append(r)
        clock.t += step
    return out


def test_frames_are_paced_at_the_clip_rate_on_the_live_clock(tmp_path):
    clock = Clock()
    inj = Injector([make_clip(tmp_path / "a.mp4")], hold_s=0.0, sleep=clock.sleep)
    inj.start(clock.t)
    frames = play(inj, clock)
    assert len(frames) == 10
    assert [round(t - 100.0, 3) for _, t in frames] == [i / FPS for i in range(10)]
    assert not inj.active


def test_slow_processing_skips_frames_instead_of_slowing_down(tmp_path):
    clock = Clock()
    inj = Injector([make_clip(tmp_path / "a.mp4")], hold_s=0.0, sleep=clock.sleep)
    inj.start(clock.t)
    frames = play(inj, clock, step=0.25)  # 4 fps processing for a 10 fps clip
    times = [t - 100.0 for _, t in frames]
    assert len(frames) < 10 and times == sorted(times)
    assert times[-1] <= 1.0  # still finishes on time


def test_last_frame_is_held_then_the_clip_ends(tmp_path):
    clock = Clock()
    inj = Injector([make_clip(tmp_path / "a.mp4")], hold_s=2.0, sleep=clock.sleep)
    inj.start(clock.t)
    frames = play(inj, clock)
    assert len(frames) == 10 + 20  # 2 s of the last frame at 10 fps
    assert all((f == frames[9][0]).all() for f, _ in frames[10:])


def test_clips_play_in_turn(tmp_path):
    a, b = make_clip(tmp_path / "a.mp4"), make_clip(tmp_path / "b.mp4")
    inj = Injector([a, b], sleep=Clock().sleep)
    assert [inj.start(0.0), inj.start(0.0), inj.start(0.0)] == [a, b, a]
