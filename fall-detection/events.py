"""FallEvent and the alert sinks it is delivered through.

To add a channel (Telegram, web push, ...), write a class with
`send(event: FallEvent) -> None` and add one instance to the sink list in main.py.
The detector never sees sinks.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol

import cv2
import numpy as np

from detector import Detection


@dataclass
class FallEvent:
    timestamp: float            # epoch seconds
    kind: str                   # "fall" | "prolonged_lying"
    peak_hip_vel: float         # torso lengths/s
    torso_angle: float          # degrees
    snapshot_path: str | None
    stream_t: float             # seconds into the camera/video stream, for finding it in a replay
    clip_path: str | None = None  # .mp4; still being recorded when the event is sent

    @classmethod
    def from_detection(cls, d: Detection, timestamp: float, snapshot_path: str | None,
                       clip_path: str | None = None) -> FallEvent:
        return cls(timestamp=timestamp, kind=d.kind, peak_hip_vel=d.peak_hip_vel,
                   torso_angle=d.torso_angle, snapshot_path=snapshot_path, stream_t=d.t,
                   clip_path=clip_path)

    @property
    def time_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")


class AlertSink(Protocol):
    def send(self, event: FallEvent) -> None: ...


def save_snapshot(frame: np.ndarray, events_dir: str | Path, timestamp: float) -> str:
    """Write the frame as events/<YYYYmmdd-HHMMSS-mmm>.jpg and return its path.

    Saved before the sinks run so every sink, including later remote ones, gets the path.
    """
    d = Path(events_dir)
    d.mkdir(parents=True, exist_ok=True)
    name = datetime.fromtimestamp(timestamp).strftime("%Y%m%d-%H%M%S-%f")[:-3] + ".jpg"
    path = d / name
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"could not write snapshot {path}")
    return str(path)


class ConsoleSink:
    def send(self, event: FallEvent) -> None:
        print(f"[{event.time_str}] {event.kind.upper()}  "
              f"peak_vel={event.peak_hip_vel:.2f}  angle={event.torso_angle:.0f}  "
              f"stream_t={event.stream_t:.2f}s  snapshot={event.snapshot_path}  "
              f"clip={event.clip_path}", flush=True)


class FileSink:
    """Appends one JSON object per event to <events_dir>/events.jsonl."""

    def __init__(self, events_dir: str | Path):
        self.path = Path(events_dir) / "events.jsonl"

    def send(self, event: FallEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event)) + "\n")


def dispatch(event: FallEvent, sinks: Iterable[AlertSink]) -> None:
    """Send to every sink. One failing sink must not stop the others or the camera loop."""
    for sink in sinks:
        try:
            sink.send(event)
        except Exception as e:
            print(f"{type(sink).__name__} failed: {e!r}", file=sys.stderr)
