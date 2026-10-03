"""FallDetector: per-frame state machine over `Features`.

    UPRIGHT --> FALLING --> ON_GROUND --> FALL_CONFIRMED
       ^           |            |               |
       +-----------+------------+---------------+   (recovery)

Timers accumulate frame dt rather than comparing wall-clock start times, so a
gap in the pose (None frames) can be counted or ignored explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from config import Config
from features import Features


class State(Enum):
    UPRIGHT = "UPRIGHT"
    FALLING = "FALLING"
    ON_GROUND = "ON_GROUND"
    FALL_CONFIRMED = "FALL_CONFIRMED"


@dataclass
class Detection:
    t: float                # stream time in seconds
    kind: str               # "fall" | "prolonged_lying"
    peak_hip_vel: float     # torso lengths/s, max seen during the descent
    torso_angle: float      # degrees, at confirmation


class FallDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = State.UPRIGHT
        self._prev_t: float | None = None
        self._fall_start: float | None = None
        self._peak_vel = 0.0
        self._last_angle = 0.0
        self.still_time = 0.0    # stillness accumulated while ON_GROUND
        self.lying_time = 0.0    # time down without a fast descent (slow-fall path)
        self._upright_time = 0.0
        self._lost_time = 0.0
        self._unsteady_time = 0.0  # consecutive non-still time while ON_GROUND

    def is_down(self, f: Features) -> bool:
        """Horizontal (by torso angle or bbox shape) and, when the ankles are visible, low."""
        cfg = self.cfg
        horizontal = f.torso_angle > cfg.lying_angle or f.bbox_aspect > cfg.lying_aspect
        low = f.hip_height is None or f.hip_height < cfg.low_hip
        return horizontal and low

    @property
    def confirm_remaining(self) -> float | None:
        """Seconds of stillness left before confirming, or None when not ON_GROUND."""
        if self.state is not State.ON_GROUND:
            return None
        return max(self.cfg.confirm_s - self.still_time, 0.0)

    def _set_state(self, state: State) -> None:
        self.state = state
        self._upright_time = 0.0
        self.still_time = 0.0
        self.lying_time = 0.0
        self._unsteady_time = 0.0
        if state is State.UPRIGHT:
            self._fall_start = None
            self._peak_vel = 0.0

    def update(self, feats: Features | None, t: float) -> Detection | None:
        """Feed one frame (t in seconds). Returns a Detection on the frame a fall is confirmed."""
        dt = t - self._prev_t if self._prev_t is not None else 0.0
        self._prev_t = t
        if feats is None:
            return self._update_lost(t, dt)
        self._lost_time = 0.0
        self._last_angle = feats.torso_angle
        cfg = self.cfg

        # Recovery from any state.
        if feats.torso_angle < cfg.upright_angle:
            self._upright_time += dt
            if self.state is not State.UPRIGHT and self._upright_time >= cfg.recover_s:
                self._set_state(State.UPRIGHT)
                return None
        else:
            self._upright_time = 0.0

        down = self.is_down(feats)

        if self.state is State.UPRIGHT:
            if feats.hip_vel_peak > cfg.fall_vel:
                self._set_state(State.FALLING)
                self._fall_start = t
                self._peak_vel = feats.hip_vel_peak
                return self._update_falling(feats, t, down)
            # Slow-fall path: down for a long time without a fast descent.
            self.lying_time = self.lying_time + dt if down else 0.0
            if self.lying_time >= cfg.slow_fall_s:
                self._set_state(State.FALL_CONFIRMED)
                return Detection(t, "prolonged_lying", feats.hip_vel_peak, feats.torso_angle)
            return None

        if self.state is State.FALLING:
            self._peak_vel = max(self._peak_vel, feats.hip_vel_peak)
            return self._update_falling(feats, t, down)

        if self.state is State.ON_GROUND:
            if down and feats.motion < cfg.still_motion:
                self.still_time += dt
                self._unsteady_time = 0.0
            else:
                # Brief jitter pauses the timer; sustained movement resets it.
                self._unsteady_time += dt
                if self._unsteady_time > cfg.still_grace_s:
                    self.still_time = 0.0
            return self._maybe_confirm(t)

        return None  # FALL_CONFIRMED: wait for recovery; never re-emit

    def _update_falling(self, feats: Features, t: float, down: bool) -> None:
        if down:
            self._set_state(State.ON_GROUND)
        elif t - self._fall_start > self.cfg.fall_window_s:
            self._set_state(State.UPRIGHT)  # a sit or crouch, not a fall
        return None

    def _update_lost(self, t: float, dt: float) -> Detection | None:
        # A person on the floor is often occluded: losing the pose must not reset a fall.
        # UPRIGHT (left the frame) and FALL_CONFIRMED simply hold.
        self._lost_time += dt
        if self.state is State.FALLING:
            if t - self._fall_start > self.cfg.fall_window_s:
                self._set_state(State.UPRIGHT)
        elif self.state is State.ON_GROUND:
            # Hold during the grace period, then keep counting as if still.
            if self._lost_time > self.cfg.lost_grace_s:
                self.still_time += dt
            return self._maybe_confirm(t)
        return None

    def _maybe_confirm(self, t: float) -> Detection | None:
        if self.still_time < self.cfg.confirm_s:
            return None
        self._set_state(State.FALL_CONFIRMED)
        return Detection(t, "fall", self._peak_vel, self._last_angle)
