"""Gemini looks at a detected fall: what kind of fall, and how the person is doing.

    fall confirmed --> 6 s video up to the confirmation --> Gemini --> FallReport
                       (8 still images instead if the video request fails)
    every gemini_update_s (1 min) while the incident is open --> latest frames --> FallReport

Video shows how fast the person went down, which is most of the difference between
a fall and lying down on purpose; the updates only need the current state.

Gemini only *describes* what it sees, as fixed categories (see SCHEMA). What the
monitor is told to do comes from `guidance()`, a table written here, so a model
cannot invent medical advice. Frames go to Google: this is off unless a key is set.

The REST API is called with urllib (no SDK) on a worker thread, so the camera loop
never waits for the network.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

import cv2
import numpy as np

from clips import estimate_fps, open_writer

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
FRAME_WIDTH = 512  # images are shrunk to this before upload
JPEG_QUALITY = 80
VIDEO_S = 6.0  # first look: this much video, ending at the confirmation

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

PROMPT_FALL_VIDEO = """You are checking a home camera that detected a possible fall of an older adult.
The video is the {seconds:.0f} seconds up to the moment the system confirmed the fall.
Pay attention to how fast the person went down: a fall is sudden and uncontrolled,
lying down on purpose is slow and supported. Classify what happened, using only what
is visible."""

PROMPT_UPDATE = """You are checking on an older adult who fell {minutes} minutes ago in front of a
home camera. The images are the last few seconds, in time order. Classify how they are
now, using only what is visible. The fall_type field is about the original fall; answer
"unclear" if you cannot tell it from these images."""


SCENE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "description": {"type": "STRING", "description": "One sentence: what room this is and "
                        "its main furniture."},
        "rest_areas": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {
                "label": {"type": "STRING", "enum": ["bed", "sofa", "armchair", "recliner"]},
                "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"},
                           "description": "[ymin, xmin, ymax, xmax], each 0-1000"},
            },
            "required": ["label", "box_2d"]}},
    },
    "required": ["description", "rest_areas"],
}

PROMPT_SCENE = """This is a fixed home camera watching an older adult's room. Find the
furniture a person may lie on to rest: beds, sofas, armchairs, recliners. For each one,
box ONLY its top surface where a body would lie or sit (the mattress, the seat cushions):
not the frame, legs, sides or headboard, and never the floor in front of it. Do not
include chairs at a table, tables, rugs, mats or the floor. An empty list is fine."""


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
ON_FLOOR = {"on_back", "face_down", "on_side", "sitting_on_floor"}
_RANK = {URGENT: 3, HIGH: 2, MEDIUM: 1, LOW: 0}


def guidance(r: FallReport, still_minutes: float = 0.0,
             emergency: str = "911") -> tuple[str, str]:
    """(urgency, instruction for the monitor) from the report's categories.
    still_minutes: how long consecutive checks have seen them not moving."""
    head = (HIGH, "Their head may have hit the floor. Ask them to stay still. "
                  f"Call {emergency} if they are confused, drowsy, vomiting or bleeding.")
    if r.movement == "up_and_moving" or r.position == "standing":
        # Back up: the fall-type advice no longer applies, a head strike still does.
        rules = [(LOW, "They appear to be up. Check in by voice anyway.")]
        return _combine(rules + ([head] if r.head_struck_likely == "yes" else []))
    rules: list[tuple[str, str]] = []
    if not r.is_fall or r.fall_type == "not_a_fall":
        if r.position not in ON_FLOOR:
            return _combine([(LOW, "This may not be a fall. Check the live camera to be sure.")])
        # Staged-looking or slow falls read as "lying down on purpose": still on the floor.
        rules.append((MEDIUM, "It may not have been a fall, but they are on the floor. "
                              "Check the live camera and talk to them."))
    if r.fall_type == "collapse":
        rules.append((URGENT, f"They may have fainted or lost consciousness. Call {emergency} now."))
    if r.position == "face_down" and r.movement == "still":
        rules.append((URGENT, f"Face down and not moving. Call {emergency} now."))
    if r.movement == "still" and still_minutes >= 5:
        rules.append((URGENT, f"Not moving for {still_minutes:.0f}+ minutes. Call {emergency}."))
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


RETRY_CODES = {500, 502, 503, 504}  # Google overloaded: worth another try in a few seconds
RETRY_WAITS_S = (2.0, 5.0)  # per model, between attempts
RATE_LIMITED = 429  # our quota: skip that model until its retry delay has passed
DEFAULT_COOLDOWN_S = 60.0  # per-minute limit, when Google gives no retry delay
DAILY_COOLDOWN_S = 3600.0  # daily limit used up: look again in an hour


def cooldown_s(error_body: bytes) -> float:
    # How long to leave a model alone after a 429, from Google's error details:
    # RetryInfo.retryDelay ("33s"), and a quotaId naming a per-day limit.
    try:
        details = json.loads(error_body).get("error", {}).get("details", [])
    except (ValueError, AttributeError):
        return DEFAULT_COOLDOWN_S
    delay = DEFAULT_COOLDOWN_S
    for d in details:
        if str(d.get("retryDelay", "")).endswith("s"):
            try:
                delay = max(1.0, float(d["retryDelay"][:-1]))
            except ValueError:
                pass
        for v in d.get("violations", []) or []:
            if "PerDay" in str(v.get("quotaId", "")):
                return max(delay, DAILY_COOLDOWN_S)
    return delay


class GeminiClient:
    def __init__(self, api_key: str, model: str, timeout_s: float = 60.0,
                 fallback_models: tuple[str, ...] = (), sleep=time.sleep,
                 clock=time.monotonic, video_fps: float = 5.0):
        self.api_key, self.model, self.timeout_s = api_key, model, timeout_s
        self.models = (model,) + tuple(m for m in fallback_models if m and m != model)
        self.video_fps = video_fps
        self._sleep, self._clock = sleep, clock
        self.rate_limited_until: dict[str, float] = {}  # model -> clock time it may be used again
        # Real usage from each reply's usageMetadata, totalled since start (any thread).
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self._usage_lock = threading.Lock()

    def record_usage(self, model: str, reply: dict) -> None:
        """Add one reply's usageMetadata to the totals and print it, so the real cost of
        a fall (video first look + one update a minute) is visible, not estimated."""
        meta = reply.get("usageMetadata") or {}
        n_in = int(meta.get("promptTokenCount", 0))
        n_out = int(meta.get("candidatesTokenCount", 0)) + int(meta.get("thoughtsTokenCount", 0))
        with self._usage_lock:
            u = self.usage
            u["calls"] += 1
            u["input_tokens"] += n_in
            u["output_tokens"] += n_out
            total = u["input_tokens"] + u["output_tokens"]
            calls = u["calls"]
        print(f"[gemini] {model}: {n_in} in + {n_out} out tokens "
              f"(since start: {calls} calls, {total} tokens)", flush=True)

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
        return self._generate(self.request_body(prompt, frames))

    def analyze_video(self, mp4: bytes, fps: float, seconds: float) -> FallReport:
        """First look at a new fall from a short clip."""
        return self._generate(self.video_request_body(mp4, fps, seconds))

    def video_request_body(self, mp4: bytes, fps: float, seconds: float) -> dict:
        video = {"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(mp4).decode()},
                 "video_metadata": {"fps": round(fps, 2)}}  # default sampling is only 1 fps
        return {
            "contents": [{"parts": [{"text": PROMPT_FALL_VIDEO.format(seconds=seconds)}, video]}],
            "generationConfig": {"response_mime_type": "application/json",
                                 "response_schema": SCHEMA, "temperature": 0.0},
        }

    def analyze_scene(self, frame: np.ndarray) -> tuple[str, list[dict]]:
        """Room scan: (description, [{"label", "box_2d"}]) for zones.zones_from_gemini."""
        body = {
            "contents": [{"parts": [{"text": PROMPT_SCENE}] + encode_frames([(0.0, frame)])[1:]}],
            "generationConfig": {"response_mime_type": "application/json",
                                 "response_schema": SCENE_SCHEMA, "temperature": 0.0},
        }
        out = self._generate(body, parse=lambda d: d)
        return str(out.get("description", ""))[:300], list(out.get("rest_areas", []))

    def _generate(self, body: dict, parse=FallReport.from_json):
        data = json.dumps(body).encode()
        last: Exception | None = None
        # 503 "high demand": retry a few seconds later. 429 is our own quota (free tier:
        # 5 requests a minute, 20 a day on 3.8 Flash), so waiting seconds does not help:
        # skip that model until Google's retry delay has passed and use the next one.
        for model in self.models:
            if self._clock() < self.rate_limited_until.get(model, float("-inf")):
                continue
            for attempt in range(len(RETRY_WAITS_S) + 1):
                if attempt:
                    self._sleep(RETRY_WAITS_S[attempt - 1])
                try:
                    return parse(self._call(model, data))
                except urllib.error.HTTPError as e:
                    if e.code == RATE_LIMITED:
                        wait = cooldown_s(e.read() if e.fp else b"")
                        self.rate_limited_until[model] = self._clock() + wait
                        print(f"[gemini] {model} rate limited: skipping it for {wait:.0f} s",
                              file=sys.stderr, flush=True)
                        last = e
                        break
                    if e.code not in RETRY_CODES:
                        raise  # bad key or bad request: retrying will not help
                    last = e
                except (TimeoutError, urllib.error.URLError) as e:
                    last = e
            else:
                print(f"[gemini] {model} unavailable ({last!r})", file=sys.stderr, flush=True)
        if last is None:
            raise RuntimeError("every Gemini model is rate limited for now")
        raise last

    def _call(self, model: str, data: bytes) -> dict:
        req = urllib.request.Request(
            API_URL.format(model=model), data=data,
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            out = json.loads(resp.read())
        self.record_usage(model, out)
        text = out["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)


# --- frames to send --------------------------------------------------------------------
class FrameHistory:
    """The last `keep_s` seconds at up to 1/every_s fps, kept as JPEGs (~3 MB for 12 s),
    for both the video and the still images."""

    def __init__(self, keep_s: float = 12.0, every_s: float = 0.1):
        self.keep_s, self.every_s = keep_s, every_s
        self._frames: deque[tuple[float, bytes]] = deque()

    def add(self, frame: np.ndarray, t: float) -> None:
        if self._frames and t - self._frames[-1][0] < self.every_s * 0.9:  # 0.9: frame jitter
            return
        h, w = frame.shape[:2]
        small = frame if w <= FRAME_WIDTH else cv2.resize(
            frame, (FRAME_WIDTH, round(h * FRAME_WIDTH / w / 2) * 2), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        self._frames.append((t, jpg.tobytes()))
        while self._frames and t - self._frames[0][0] > self.keep_s:
            self._frames.popleft()

    def span(self, span_s: float) -> list[tuple[float, bytes]]:
        """JPEGs from the last span_s, oldest first (decoded later, off the camera loop)."""
        if not self._frames:
            return []
        end = self._frames[-1][0]
        return [(t, j) for t, j in self._frames if t >= end - span_s]

    def recent(self, n: int, span_s: float, ref_t: float) -> list[tuple[float, np.ndarray]]:
        """Up to n frames evenly spread over the last span_s, timed relative to ref_t."""
        pool = self.span(span_s)
        if len(pool) > n:
            idx = np.linspace(0, len(pool) - 1, n).round().astype(int)
            pool = [pool[i] for i in idx]
        return [(t - ref_t, _decode(j)) for t, j in pool]


def _decode(jpg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


def thin(frames: list[tuple[float, bytes]], fps: float) -> list[tuple[float, bytes]]:
    """Keep at most `fps` frames per second (the buffer holds 10)."""
    out, last = [], float("-inf")
    for t, j in frames:
        if t - last >= 0.9 / fps:
            out.append((t, j))
            last = t
    return out


def encode_mp4(frames: list[tuple[float, bytes]]) -> tuple[bytes, float]:
    """(t, jpeg) frames -> (mp4 bytes, fps). H.264 on Windows, like the event clips."""
    fps = estimate_fps([t for t, _ in frames], default=10.0)
    first = _decode(frames[0][1])
    size = (first.shape[1], first.shape[0])
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        w = open_writer(path, fps, size)
        for _, j in frames:
            f = _decode(j)
            w.write(f if (f.shape[1], f.shape[0]) == size else cv2.resize(f, size))
        w.release()
        with open(path, "rb") as fh:
            return fh.read(), fps
    finally:
        os.remove(path)


# --- schedule: first look, then every update_s while the incident is open --------------
@dataclass
class _Watch:
    fall_t: float      # stream time of the confirmation
    check_t: float     # stream time of the last check (or the confirmation)
    updates: int = 0
    minutes: float = 0.0  # time since the fall as reported, sped up by fast-forward
    last_t: float = 0.0
    checked_min: float = 0.0  # `minutes` at the last check (0 = the first look)


class FallAnalyst:
    """Runs Gemini on a worker thread and reports through `on_report`.

    main.py calls start() when a fall is confirmed, tick() every frame, and stop()
    when the person is back up. `is_open(incident_id)` lets the alert manager end the
    updates when the person answered "I'm OK".
    """

    def __init__(self, client: GeminiClient,
                 on_report: Callable[[str, FallReport, float, float], None],
                 is_open: Callable[[str], bool] = lambda _id: True,
                 update_s: float = 60.0, max_updates: int = 60,
                 update_client: GeminiClient | None = None):
        self.client, self.on_report, self.is_open = client, on_report, is_open
        # The per-minute checks go to a model with a bigger daily allowance.
        self.update_client = update_client or client
        self.update_s, self.max_updates = update_s, max_updates
        # Demo fast-forward: 30 = a check every 2 s, each counting as a minute. Set from
        # the dashboard's server thread, read here on the camera loop.
        self.speed = 1.0
        self.watches: dict[str, _Watch] = {}
        self._still_since: dict[str, float] = {}  # worker thread: minute they were first seen still
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="gemini", daemon=True).start()

    def start(self, incident_id: str, history: FrameHistory, t: float) -> None:
        # Video of the last VIDEO_S, with 8 stills over 10 s as the fallback.
        clip = history.span(VIDEO_S)
        if clip:
            self._q.put((incident_id, clip, history.recent(n=8, span_s=10.0, ref_t=t), None))
        self.watches[incident_id] = _Watch(fall_t=t, check_t=t, last_t=t)

    def tick(self, history: FrameHistory, t: float) -> None:
        interval = self.update_s / self.speed
        for inc_id, w in list(self.watches.items()):
            w.minutes += (t - w.last_t) * self.speed / 60.0
            w.last_t = t
            if not self.is_open(inc_id) or w.updates >= self.max_updates:
                del self.watches[inc_id]
            elif t - w.check_t >= interval - 1e-6 and round(w.minutes) > round(w.checked_min):
                # Timed at the current speed, so a switch acts at once; never twice a minute.
                w.updates += 1
                w.check_t, w.checked_min = t, w.minutes
                self._q.put((inc_id, None, history.recent(n=4, span_s=4.0, ref_t=t), w.minutes))

    def stop(self) -> None:
        """Person is back up: no more updates."""
        self.watches.clear()

    def _run(self) -> None:
        while True:
            inc_id, clip, frames, minutes = self._q.get()
            report = first_look(self.client, clip) if clip else None
            try:
                if report is None:
                    report = (self.client if clip else self.update_client).analyze(frames, minutes)
            except Exception as e:  # network, quota, bad reply: never stop the detector
                print(f"[gemini] analysis failed: {e!r}", file=sys.stderr, flush=True)
                continue
            try:
                minutes = 0.0 if minutes is None else minutes
                if report.movement == "still":
                    since = self._still_since.setdefault(inc_id, minutes)
                else:  # moving again: the still count starts over
                    self._still_since.pop(inc_id, None)
                    since = minutes
                self.on_report(inc_id, report, minutes, minutes - since)
            except Exception as e:
                print(f"[gemini] report handler failed: {e!r}", file=sys.stderr, flush=True)


def first_look(client: GeminiClient, clip: list[tuple[float, bytes]]) -> FallReport | None:
    """Video analysis of a new fall, or None (the caller then sends still images)."""
    try:
        mp4, fps = encode_mp4(thin(clip, client.video_fps))
        return client.analyze_video(mp4, fps, clip[-1][0] - clip[0][0])
    except Exception as e:
        print(f"[gemini] video failed, using still images: {e!r}", file=sys.stderr, flush=True)
        return None


def report_dict(r: FallReport, minutes: float, emergency: str = "911",
                still_minutes: float = 0.0) -> dict:
    level, text = guidance(r, still_minutes, emergency)
    return asdict(r) | {"urgency": level, "guidance": text, "minutes": round(minutes, 1),
                        "still_minutes": round(still_minutes, 1), "t": time.time()}
