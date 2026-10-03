import numpy as np
import pytest

from config import Config
from cover import CoverReset

FPS = 30.0
COVERED = np.full((48, 64, 3), 12, dtype=np.uint8)
OPEN = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
BRIGHT_WALL = np.full((48, 64, 3), 200, dtype=np.uint8)  # flat but not dark


def feed(cr, frame, seconds, t0):
    fired = 0
    n = round(seconds * FPS)
    for i in range(n):
        fired += cr.update(frame, t0 + i / FPS)
    return fired, t0 + n / FPS


def test_fires_once_after_cover_reset_s():
    cr = CoverReset(Config(cover_reset_s=3.0))
    fired, t = feed(cr, OPEN, 1, 0.0)
    fired, t = feed(cr, COVERED, 2.9, t)
    assert fired == 0
    assert cr.remaining == pytest.approx(0.1, abs=0.05)
    fired, t = feed(cr, COVERED, 5, t)
    assert fired == 1
    assert cr.fired and cr.remaining is None


def test_uncovering_early_cancels():
    cr = CoverReset(Config(cover_reset_s=3.0))
    fired, t = feed(cr, COVERED, 2, 0.0)
    fired2, t = feed(cr, OPEN, 0.1, t)
    fired3, t = feed(cr, COVERED, 2, t)
    assert fired + fired2 + fired3 == 0


def test_can_fire_again_after_uncovering():
    cr = CoverReset(Config(cover_reset_s=1.0))
    f1, t = feed(cr, COVERED, 2, 0.0)
    _, t = feed(cr, OPEN, 0.5, t)
    f2, t = feed(cr, COVERED, 2, t)
    assert (f1, f2) == (1, 1)


def test_flat_bright_frame_is_not_covered():
    cr = CoverReset(Config())
    cr.update(BRIGHT_WALL, 0.0)
    assert not cr.covered
