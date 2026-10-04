"""Gemini looks at a detected fall: what kind of fall, and how the person is doing.

    fall confirmed --> frames from just before and after --> Gemini --> FallReport
    every gemini_update_s while the incident is open --> latest frames --> FallReport

Gemini only *describes* what it sees, as fixed categories (see SCHEMA). What the
monitor is told to do comes from `guidance()`, a table written here, so a model
cannot invent medical advice. Frames go to Google: this is off unless a key is set.

The REST API is called with urllib (no SDK) on a worker thread, so the camera loop
never waits for the network.
"""

from __future__ import annotations

import base64
import json
import queue
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

import cv2
import numpy as np

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
FRAME_WIDTH = 512  # images are shrunk to this before upload
JPEG_QUALITY = 80

FALL_TYPES = ["backward", "forward", "sideways", "slid_from_furniture", "collapse",
              "not_a_fall", "unclear"]
POSITIONS = ["on_back", "face_down", "on_side", "sitting_on_floor", "standing", "not_visible"]
MOVEMENTS = ["still", "small_movements", "trying_to_get_up", "up_and_moving", "not_visible"]
YES_NO = ["yes", "no", "unclear"]

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_fall": {"type": "BOOLEAN", "description": "Did a person fall (as opposed to sitting, "
                    "kneeling, exercising or lying down on purpose)?"},
        "fall_type": {"type": "STRING", "enum": FALL_TYPES,
                      "description": "collapse = went limp / fainted rather than tripping"},
        "position": {"type": "STRING", "enum": POSITIONS, "description": "In the LAST frame."},
        "movement": {"type": "STRING", "enum": MOVEMENTS, "description": "In the last few frames."},
        "head_struck_likely": {"type": "STRING", "enum": YES_NO},
        "description": {"type": "STRING", "description": "One or two short, plain sentences "
                        "for a family member. Describe only what is visible. No medical advice."},
    },
    "required": ["is_fall", "fall_type", "position", "movement", "head_struck_likely",
                 "description"],
}

PROMPT_FALL = """You are checking a home camera that detected a possible fall of an older adult.
The images are in time order, with seconds relative to the moment the system confirmed
the fall (negative = before). Classify what happened, using only what is visible."""

PROMPT_UPDATE = """You are checking on an older adult who fell {minutes} minutes ago in front of a
home camera. The images are the last few seconds, in time order. Classify how they are
now, using only what is visible. The fall_type field is about the original fall; answer
"unclear" if you cannot tell it from these images."""


@dataclass
class FallReport:
    is_fall: bool
    fall_type: str
    position: str
    movement: str
    head_struck_likely: str
    description: str

    @classmethod
    def from_json(cls, data: dict) -> FallReport:
        def pick(key, allowed):
            v = str(data.get(key, "")).strip().lower()
            return v if v in allowed else allowed[-1]  # last value is the "unknown" one
        return cls(is_fall=bool(data.get("is_fall", True)),
                   fall_type=pick("fall_type", FALL_TYPES),
                   position=pick("position", POSITIONS),
                   movement=pick("movement", MOVEMENTS),
                   head_struck_likely=pick("head_struck_likely", YES_NO),
                   description=str(data.get("description", "")).strip()[:400])


# --- what the monitor should do: a fixed table, never model output -------------------
URGENT, HIGH, MEDIUM, LOW = "urgent", "high", "medium", "low"
_RANK = {URGENT: 3, HIGH: 2, MEDIUM: 1, LOW: 0}


def guidance(r: FallReport, minutes_since_fall: float, emergency: str = "911") -> tuple[str, str]:
    """(urgency, instruction for the monitor) from the report's categories."""
    head = (HIGH, "Their head may have hit the floor. Ask them to stay still. "
                  f"Call {emergency} if they are confused, drowsy, vomiting or bleeding.")
    if r.movement == "up_and_moving" or r.position == "standing":
        # Back up: the fall-type advice no longer applies, a head strike still does.
        rules = [(LOW, "They appear to be up. Check in by voice anyway.")]
        return _combine(rules + ([head] if r.head_struck_likely == "yes" else []))
    if not r.is_fall or r.fall_type == "not_a_fall":
        return _combine([(LOW, "This may not be a fall. Check the live camera to be sure.")])
    rules: list[tuple[str, str]] = []
    if r.fall_type == "collapse":
        rules.append((URGENT, f"They may have fainted or lost consciousness. Call {emergency} now."))
    if r.position == "face_down" and r.movement == "still":
        rules.append((URGENT, f"Face down and not moving. Call {emergency} now."))
    if r.movement == "still" and minutes_since_fall >= 5:
        rules.append((URGENT, f"Not moving for {minutes_since_fall:.0f}+ minutes. Call {emergency}."))
    if r.head_struck_likely == "yes":
        rules.append(head)
    if r.fall_type == "sideways":
        rules.append((HIGH, "Sideways falls can injure the hip. If they cannot put weight on a "
                            "leg, do not help them stand; call for help."))
    if r.fall_type == "forward":
        rules.append((MEDIUM, "Check their face and wrists. Talk to them before they get up."))
    if r.fall_type == "backward":
        rules.append((MEDIUM, "Talk to them. Watch for head or back pain."))
    if r.fall_type == "slid_from_furniture":
        rules.append((MEDIUM, "Usually a lower-risk fall. Talk to them before they stand."))
    if r.movement == "trying_to_get_up":
        rules.append((MEDIUM, "They are trying to get up: tell them to rest first. "
                              "Standing too fast causes second falls."))
    return _combine(rules or [(MEDIUM, "Check the live camera and talk to them.")])


def _combine(rules: list[tuple[str, str]]) -> tuple[str, str]:
    """Highest urgency; most urgent first, at most two so the notification stays readable."""
    level = max((u for u, _ in rules), key=_RANK.get)
    return level, " ".join(t for _, t in sorted(rules, key=lambda x: -_RANK[x[0]])[:2])


# --- Gemini API ----------------------------------------------------------------------
def encode_frames(frames: list[tuple[float, np.ndarray]]) -> list[dict]:
    """[(seconds relative to the fall, frame)] -> alternating text / image parts."""
    parts = []
    for rel_t, frame in frames:
        h, w = frame.shape[:2]
        if w > FRAME_WIDTH:
            frame = cv2.resize(frame, (FRAME_WIDTH, round(h * FRAME_WIDTH / w)),
                               interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue
        parts.append({"text": f"t = {rel_t:+.1f} s"})
        parts.append({"inline_data": {"mime_type": "image/jpeg",
                                      "data": base64.b64encode(jpg.tobytes()).decode()}})
    return parts


class GeminiClient:
    def __init__(self, api_key: str, model: str, timeout_s: float = 30.0):
        self.api_key, self.model, self.timeout_s = api_key, model, timeout_s

    def request_body(self, prompt: str, frames: list[tuple[float, np.ndarray]]) -> dict:
        return {
            "contents": [{"parts": [{"text": prompt}] + encode_frames(frames)}],
            "generationConfig": {"response_mime_type": "application/json",
                                 "response_schema": SCHEMA, "temperature": 0.0},
        }

    def analyze(self, frames: list[tuple[float, np.ndarray]],
                minutes_since_fall: float | None = None) -> FallReport:
        """minutes_since_fall None: the first look at a new fall; else a status update."""
        prompt = (PROMPT_FALL if minutes_since_fall is None
                  else PROMPT_UPDATE.format(minutes=round(minutes_since_fall)))
        req = urllib.request.Request(
            API_URL.format(model=self.model),
            data=json.dumps(self.request_body(prompt, frames)).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            out = json.loads(resp.read())
        text = out["candidates"][0]["content"]["parts"][0]["text"]
        return FallReport.from_json(json.loads(text))


# --- frames to send --------------------------------------------------------------------
class FrameHistory:
    """The last `keep_s` seconds of frames, one every `every_s`, for the analysis."""

    def __init__(self, keep_s: float = 12.0, every_s: float = 0.5):
        self.keep_s, self.every_s = keep_s, every_s
        self._frames: deque[tuple[float, np.ndarray]] = deque()

    def add(self, frame: np.ndarray, t: float) -> None:
        if self._frames and t - self._frames[-1][0] < self.every_s:
            return
        h, w = frame.shape[:2]
        small = frame if w <= FRAME_WIDTH else cv2.resize(
            frame, (FRAME_WIDTH, round(h * FRAME_WIDTH / w)), interpolation=cv2.INTER_AREA)
        self._frames.append((t, small.copy()))
        while self._frames and t - self._frames[0][0] > self.keep_s:
            self._frames.popleft()

    def recent(self, n: int, span_s: float, ref_t: float) -> list[tuple[float, np.ndarray]]:
        """Up to n frames evenly spread over the last span_s, timed relative to ref_t."""
        if not self._frames:
            return []
        end = self._frames[-1][0]
        pool = [(t, f) for t, f in self._frames if t >= end - span_s]
        if len(pool) > n:
            idx = np.linspace(0, len(pool) - 1, n).round().astype(int)
            pool = [pool[i] for i in idx]
        return [(t - ref_t, f) for t, f in pool]


# --- schedule: first look, then every update_s while the incident is open --------------
@dataclass
class _Watch:
    fall_t: float      # stream time of the confirmation
    next_t: float
    updates: int = 0


class FallAnalyst:
    """Runs Gemini on a worker thread and reports through `on_report`.

    main.py calls start() when a fall is confirmed, tick() every frame, and stop()
    when the person is back up. `is_open(incident_id)` lets the alert manager end the
    updates when the person answered "I'm OK".
    """

    def __init__(self, client: GeminiClient, on_report: Callable[[str, FallReport, float], None],
                 is_open: Callable[[str], bool] = lambda _id: True,
                 update_s: float = 300.0, max_updates: int = 12):
        self.client, self.on_report, self.is_open = client, on_report, is_open
        self.update_s, self.max_updates = update_s, max_updates
        self.watches: dict[str, _Watch] = {}
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="gemini", daemon=True).start()

    def start(self, incident_id: str, history: FrameHistory, t: float) -> None:
        # Fall confirmation comes >= confirm_s after the descent: 10 s back covers it.
        frames = history.recent(n=8, span_s=10.0, ref_t=t)
        if frames:
            self._q.put((incident_id, frames, None))
        self.watches[incident_id] = _Watch(fall_t=t, next_t=t + self.update_s)

    def tick(self, history: FrameHistory, t: float) -> None:
        for inc_id, w in list(self.watches.items()):
            if not self.is_open(inc_id) or w.updates >= self.max_updates:
                del self.watches[inc_id]
            elif t >= w.next_t:
                w.updates += 1
                w.next_t = t + self.update_s
                minutes = (t - w.fall_t) / 60.0
                self._q.put((inc_id, history.recent(n=4, span_s=4.0, ref_t=t), minutes))

    def stop(self) -> None:
        """Person is back up: no more updates."""
        self.watches.clear()

    def _run(self) -> None:
        while True:
            inc_id, frames, minutes = self._q.get()
            try:
                report = self.client.analyze(frames, minutes)
            except Exception as e:  # network, quota, bad reply: never stop the detector
                print(f"[gemini] analysis failed: {e!r}", file=sys.stderr, flush=True)
                continue
            try:
                self.on_report(inc_id, report, 0.0 if minutes is None else minutes)
            except Exception as e:
                print(f"[gemini] report handler failed: {e!r}", file=sys.stderr, flush=True)


def report_dict(r: FallReport, minutes: float, emergency: str = "911") -> dict:
    level, text = guidance(r, minutes, emergency)
    return asdict(r) | {"urgency": level, "guidance": text, "minutes": round(minutes, 1),
                        "t": time.time()}
