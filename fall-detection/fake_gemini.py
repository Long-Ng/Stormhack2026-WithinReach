"""Backup for the demo: Gemini answers from a script instead of the API.

Same interface as gemini.GeminiClient, so the analyst, the guidance table, the alerts
and the dashboard all run exactly as with the real model; only the answers are
preloaded. No key, no network, no rate limits. Turned on with gemini_fake = true.

The story: a sideways fall near the sofa; she tries to get up, sits, then lies back
down and stops moving, so the guidance climbs to "call 911" after 5 still minutes.
Each update answers for the minutes since the fall (fast-forward makes those 2 s).
"""

from __future__ import annotations

import time

from gemini import FallReport

FIRST_LOOK = dict(
    is_fall=True, fall_type="sideways", position="on_side", movement="small_movements",
    head_struck_likely="no",
    description="She lost her balance next to the sofa and fell onto her left side. "
                "She is moving her arms but has not got up.")

# Minute after the fall -> what Gemini "sees"; the last entry repeats from then on.
TIMELINE = [
    FIRST_LOOK,
    dict(position="on_side", movement="trying_to_get_up",
         description="She is pushing up on her arms but cannot get her legs under her."),
    dict(position="sitting_on_floor", movement="small_movements",
         description="She has rolled into a sitting position on the floor and is holding "
                     "her left hip."),
    dict(position="sitting_on_floor", movement="trying_to_get_up",
         description="She is reaching for the sofa to pull herself up, without success."),
    dict(position="on_side", movement="still",
         description="She has lain back down on her left side and stopped moving."),
    dict(position="on_side", movement="still",
         description="She is still lying on her side. No movement seen."),
    dict(position="on_side", movement="still",
         description="No change: lying on her left side, eyes closed, not moving."),
]


class FakeGeminiClient:
    video_fps = 5.0

    def __init__(self, latency_s: float = 1.0, sleep=time.sleep):
        self.latency_s, self._sleep = latency_s, sleep  # a short pause, like a real call
        self.model = "preloaded demo answers"

    def _report(self, minute: int) -> FallReport:
        step = FIRST_LOOK | TIMELINE[min(max(minute, 0), len(TIMELINE) - 1)]
        if self.latency_s:
            self._sleep(self.latency_s)
        print(f"[gemini-demo] minute {minute}: {step['position']}, {step['movement']}", flush=True)
        return FallReport.from_json(step)

    def analyze_video(self, mp4: bytes, fps: float, seconds: float) -> FallReport:
        return self._report(0)

    def analyze(self, frames, minutes_since_fall: float | None = None) -> FallReport:
        return self._report(0 if minutes_since_fall is None else round(minutes_since_fall))

    def analyze_scene(self, frame) -> tuple[str, list[dict]]:
        return "Demo room (preloaded answers).", []
