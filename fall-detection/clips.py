"""Event video clips: the last clip_pre_s seconds before a fall plus everything after it.

Every frame goes through `ClipRecorder.add`. Frames are shrunk to max_width and
thinned to max_fps (OpenCV's H.264 writer has a fixed bitrate of 1 bit per pixel per
frame, so these two set the file size: 640 px at 15 fps is ~25 MB/min), then kept as
JPEGs in a rolling pre-roll buffer, so 30 s costs a few MB rather than ~1 GB raw. On
`trigger` the buffer is written out and recording continues until the person has been
back up for clip_tail_s, or clip_max_after_s has passed.

Encoding and disk writes run on a worker thread so the camera loop never stalls; the
main thread only decides when a clip starts and stops.
"""

from __future__ import annotations

import queue
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

JPEG_QUALITY = 85
QUEUE_FRAMES = 120  # frames waiting for the worker; beyond this, frames are dropped


def clip_name(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y%m%d-%H%M%S-%f")[:-3] + ".mp4"


def estimate_fps(times: list[float], default: float = 20.0) -> float:
    """Frame rate from timestamps, so the clip plays back at real speed."""
    if len(times) < 2 or times[-1] <= times[0]:
        return default
    return min(max((len(times) - 1) / (times[-1] - times[0]), 1.0), 60.0)


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """H.264 via Media Foundation on Windows (plays in browsers); MPEG-4 Part 2 elsewhere."""
    if sys.platform == "win32":
        w = cv2.VideoWriter(str(path), cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"avc1"), fps, size)
        if w.isOpened():
            return w
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not w.isOpened():
        raise OSError(f"could not open video writer for {path}")
    return w


class ClipRecorder:
    def __init__(self, out_dir: str | Path, pre_s: float, tail_s: float, max_after_s: float,
                 max_width: int = 640, max_fps: float = 15.0):
        self.out_dir = Path(out_dir)
        self.pre_s, self.tail_s, self.max_after_s = pre_s, tail_s, max_after_s
        self.max_width, self.min_dt = max_width, 0.9 / max_fps  # 0.9: tolerate frame jitter
        # Main-thread state: the clip being recorded, when it started, last time still down.
        self.path: str | None = None
        self._start_t = 0.0
        self._down_t = 0.0
        self._q: queue.Queue = queue.Queue(maxsize=QUEUE_FRAMES)
        self._worker = threading.Thread(target=self._run, name="clip-recorder", daemon=True)
        self._worker.start()

    @property
    def recording(self) -> bool:
        return self.path is not None

    def trigger(self, timestamp: float, t: float) -> str:
        """Start a clip (pre-roll included) and return its path. A clip already in
        progress is extended rather than split, and its path is returned."""
        if self.path is None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.path = str(self.out_dir / clip_name(timestamp))
            self._start_t = t
            self._q.put(("start", self.path))  # blocking: control messages must not be dropped
        self._down_t = t
        return self.path

    def add(self, frame: np.ndarray, t: float, down: bool) -> None:
        """Feed every frame (clean, before overlays). down: person not yet recovered."""
        try:
            self._q.put_nowait(("frame", frame.copy(), t))
        except queue.Full:
            pass  # worker behind (slow disk): a dropped frame beats stalling detection
        if self.path is None:
            return
        if down:
            self._down_t = t
        if t - self._down_t >= self.tail_s or t - self._start_t >= self.max_after_s:
            self.stop()

    def stop(self) -> None:
        if self.path is not None:
            self.path = None
            self._q.put(("stop",))

    def close(self) -> None:
        """Finish any clip in progress and wait for it to be written."""
        self.stop()
        self._q.put(("close",))
        self._worker.join()

    def _run(self) -> None:
        buffer: deque[tuple[float, bytes]] = deque()  # (t, jpeg) pre-roll
        writer: cv2.VideoWriter | None = None
        size: tuple[int, int] | None = None
        path = ""
        kept_t = float("-inf")  # time of the last frame kept after thinning

        def write(frame: np.ndarray) -> None:
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)  # camera changed resolution mid-clip
            writer.write(frame)

        while True:
            msg = self._q.get()
            try:
                if msg[0] == "frame":
                    _, frame, t = msg
                    if t - kept_t < self.min_dt:
                        continue
                    kept_t = t
                    h, w = frame.shape[:2]
                    if w > self.max_width:
                        frame = cv2.resize(frame, (self.max_width, round(h * self.max_width / w / 2) * 2),
                                           interpolation=cv2.INTER_AREA)  # even height for H.264
                    if writer is not None:
                        write(frame)
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if ok:
                        buffer.append((t, jpg.tobytes()))
                    while buffer and t - buffer[0][0] > self.pre_s:
                        buffer.popleft()
                elif msg[0] == "start":
                    path = msg[1]
                    frames = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)
                              for _, j in buffer]
                    if not frames:
                        continue  # triggered before any frame arrived: nothing to record
                    size = (frames[-1].shape[1], frames[-1].shape[0])
                    writer = open_writer(Path(path), estimate_fps([t for t, _ in buffer]), size)
                    for f in frames:
                        write(f)
                    print(f"[clip] recording {path}", flush=True)
                elif msg[0] in ("stop", "close"):
                    if writer is not None:
                        writer.release()
                        writer = None
                        print(f"[clip] saved {path}", flush=True)
                    if msg[0] == "close":
                        return
            except Exception as e:  # a recording problem must never stop the detector
                print(f"[clip] failed: {e!r}", file=sys.stderr)
                if writer is not None:
                    writer.release()
                    writer = None
