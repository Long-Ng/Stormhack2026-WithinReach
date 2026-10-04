"""Rest zones (bed, sofa, ...) where lying down is resting, not a fall, and the scene
check that asks Gemini to find them again when the room changes.

    data/zones.json  {"zones": [{"label", "box": [x0, y0, x1, y1], "source"}], ...}

Boxes are fractions of the frame (0-1), so they survive a resolution change. The
person counts as in a zone only when both the shoulder and hip midpoints are inside
it: lying in bed is in, lying on the floor next to the bed (hips overlapping the bed
in the picture) is not. Zones with "source": "manual" are never replaced by Gemini.

The scene check compares the empty room with the image the zones were found on.
Brightness is normalised so a lamp going on does not count; a moved sofa does.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

REST_LABELS = ["bed", "sofa", "armchair", "recliner"]
# Only the top part of a detected box counts: Gemini's bed boxes often reach down to the
# floor in front of the bed, and someone lying there must still count as fallen.
SURFACE_SHARE = 0.6
MAX_ZONE_AREA = 0.5  # a "bed" covering half the picture would hide real falls
MAX_ZONES = 4


@dataclass
class RestZone:
    label: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions of the frame
    source: str = "gemini"  # or "manual"

    def contains(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.box
        return x0 <= x <= x1 and y0 <= y <= y1

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def zones_from_gemini(items: list[dict]) -> list[RestZone]:
    """Gemini's [{"label", "box_2d": [ymin, xmin, ymax, xmax] in 0-1000}] -> checked zones."""
    out = []
    for it in items:
        label = str(it.get("label", "")).lower()
        box = it.get("box_2d") or []
        if label not in REST_LABELS or len(box) != 4:
            continue
        ymin, xmin, ymax, xmax = (min(max(float(v), 0.0), 1000.0) / 1000.0 for v in box)
        top, bottom = min(ymin, ymax), max(ymin, ymax)
        bottom = top + (bottom - top) * SURFACE_SHARE
        z = RestZone(label, (min(xmin, xmax), top, max(xmin, xmax), bottom))
        if 0.0 < z.area <= MAX_ZONE_AREA:
            out.append(z)
    return sorted(out, key=lambda z: -z.area)[:MAX_ZONES]


class ZoneStore:
    """Zones in data/zones.json; thread-safe because the Gemini scan writes from a worker."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.zones: list[RestZone] = []
        self.description = ""
        self.analyzed_t = 0.0
        self._lock = threading.Lock()
        if self.path.is_file():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                self.zones = [RestZone(z["label"], tuple(z["box"]), z.get("source", "gemini"))
                              for z in d.get("zones", [])]
                self.description = d.get("description", "")
                self.analyzed_t = float(d.get("analyzed_t", 0.0))
            except (ValueError, KeyError, TypeError) as e:
                print(f"[zones] ignoring unreadable {self.path}: {e!r}", file=sys.stderr)

    def replace_detected(self, zones: list[RestZone], description: str) -> None:
        """New Gemini zones; manual ones stay."""
        with self._lock:
            self.zones = [z for z in self.zones if z.source == "manual"] + zones
            self.description, self.analyzed_t = description, time.time()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(
                {"zones": [asdict(z) for z in self.zones], "description": self.description,
                 "analyzed_t": self.analyzed_t}, indent=1), encoding="utf-8")

    def zone_at(self, shoulder: tuple[float, float] | None, hip: tuple[float, float] | None,
                size: tuple[int, int]) -> RestZone | None:
        """The zone holding both midpoints (pixels in a frame of size (w, h)), if any."""
        if shoulder is None or hip is None:
            return None
        w, h = size
        pts = [(shoulder[0] / w, shoulder[1] / h), (hip[0] / w, hip[1] / h)]
        with self._lock:
            for z in self.zones:
                if all(z.contains(x, y) for x, y in pts):
                    return z
        return None

    def as_json(self) -> dict:
        with self._lock:
            return {"zones": [asdict(z) for z in self.zones], "description": self.description,
                    "analyzed_t": self.analyzed_t}


def draw_zones(frame: np.ndarray, zones: list[RestZone], active: RestZone | None) -> None:
    h, w = frame.shape[:2]
    for z in zones:
        x0, y0, x1, y1 = z.box
        color = (0, 200, 0) if z is active else (200, 160, 0)
        p0, p1 = (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h))
        cv2.rectangle(frame, p0, p1, color, 2)
        cv2.putText(frame, z.label + (" (resting)" if z is active else ""), (p0[0] + 4, p0[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


# --- scene change ----------------------------------------------------------------------
def thumbnail(frame: np.ndarray) -> np.ndarray:
    """Small, blurred, brightness-normalised grey image for comparing scenes."""
    g = cv2.cvtColor(cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g.astype(np.float32), (5, 5), 0)
    return (g - g.mean()) / (g.std() + 1e-6)


def changed_fraction(a: np.ndarray, b: np.ndarray, threshold: float = 0.8) -> float:
    return float((np.abs(a - b) > threshold).mean())


class SceneWatcher:
    """Says when to (re)scan the room: no scan yet, or the empty room looks different
    from the scanned one for change_s. Only judged while nobody is in view."""

    def __init__(self, empty_s: float = 5.0, change_frac: float = 0.1, change_s: float = 20.0,
                 min_interval_s: float = 600.0, have_zones: bool = False):
        self.empty_s, self.change_frac = empty_s, change_frac
        self.change_s, self.min_interval_s = change_s, min_interval_s
        self.reference: np.ndarray | None = None
        self.need_scan = not have_zones
        self._empty_since: float | None = None
        self._changed_since: float | None = None
        self._last_scan = float("-inf")
        self._last_check = float("-inf")
        self.last_fraction = 0.0

    def update(self, frame: np.ndarray, t: float, person_in_view: bool) -> bool:
        """True when a scan should start now (the caller then calls scanned())."""
        if person_in_view:
            self._empty_since = self._changed_since = None
            return False
        if self._empty_since is None:
            self._empty_since = t
        if t - self._empty_since < self.empty_s or t - self._last_check < 1.0:
            return False  # wait for an empty room; compare once a second
        self._last_check = t
        if t - self._last_scan < self.min_interval_s:
            return False
        if self.reference is None:
            if self.need_scan:
                return True
            self.reference = thumbnail(frame)  # zones from a file: this is the room they fit
            return False
        self.last_fraction = changed_fraction(self.reference, thumbnail(frame))
        if self.last_fraction < self.change_frac:
            self._changed_since = None
            return False
        if self._changed_since is None:
            self._changed_since = t
        return t - self._changed_since >= self.change_s

    def scanned(self, frame: np.ndarray, t: float) -> None:
        self.reference = thumbnail(frame)
        self.need_scan = False
        self._last_scan, self._changed_since = t, None


class SceneScanner:
    """Runs the Gemini room scan on a worker thread and stores the result."""

    def __init__(self, scan: Callable[[np.ndarray], tuple[str, list[dict]]], store: ZoneStore,
                 watcher: SceneWatcher | None = None):
        self.scan, self.store, self.watcher = scan, store, watcher
        self.busy = False

    def start(self, frame: np.ndarray) -> None:
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._run, args=(frame.copy(),), daemon=True).start()

    def _run(self, frame: np.ndarray) -> None:
        try:
            description, items = self.scan(frame)
            zones = zones_from_gemini(items)
            self.store.replace_detected(zones, description)
            names = ", ".join(z.label for z in zones) or "none"
            print(f"[zones] room scanned: {names}. {description}", flush=True)
        except Exception as e:
            print(f"[zones] room scan failed: {e!r}", file=sys.stderr, flush=True)
            if self.watcher is not None:  # try again after scene_min_interval_s
                self.watcher.reference, self.watcher.need_scan = None, True
        finally:
            self.busy = False
