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
    kind: str               # "fall" | "prolonged_lying" | "sensor_fall"
    peak_hip_vel: float     # torso lengths/s, max seen during the descent
    torso_angle: float      # degrees, at confirmation
    sensor: bool = False    # confirmed with the help of a wearable impact


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
        self._impact_t: float | None = None  # latest wearable impact, stream time

    def add_impact(self, t: float) -> None:
        """Report a wearable impact (stream time in seconds)."""
        self._impact_t = t

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

    def update(self, feats: Features | None, t: float,
               phone_still_s: float | None = None) -> Detection | None:
        """Feed one frame (t in seconds). Returns a Detection on the frame a fall is confirmed.

        phone_still_s: how long the wearable has been still, or None without a wearable.
        """
        d = self._update_camera(feats, t)
        if d is not None:
            return d
        return self._update_wearable(feats, t, phone_still_s)

    def _update_wearable(self, feats: Features | None, t: float,
                         phone_still_s: float | None) -> Detection | None:
        """The wearable only adds detections; the camera-only path is unchanged."""
        if self._impact_t is None:
            return None
        cfg = self.cfg
        age = t - self._impact_t
        if age > cfg.imu_hold_s:
            self._impact_t = None
            return None

        # Camera saw the descent and the person is down: the impact confirms it now.
        if (self.state is State.ON_GROUND and self._fall_start is not None
                and self._fall_start - cfg.imu_match_s <= self._impact_t
                <= self._fall_start + cfg.fall_window_s + cfg.imu_match_s):
            return self._confirm(t, "fall", sensor=True)

        if self.state is State.UPRIGHT:
            # Still upright a while after the impact: the phone was dropped, not the person.
            if (feats is not None and age >= cfg.recover_s
                    and self._upright_time >= cfg.recover_s):
                self._impact_t = None
                return None
            # Camera missed the descent (out of frame, occluded, or no fast drop seen),
            # but the phone hit hard and has not moved since.
            if (phone_still_s is not None and phone_still_s >= cfg.imu_still_s
                    and (feats is None or self.is_down(feats))):
                return self._confirm(t, "sensor_fall", sensor=True)
        return None

    def _update_camera(self, feats: Features | None, t: float) -> Detection | None:
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
        return self._confirm(t, "fall")

    def _confirm(self, t: float, kind: str, sensor: bool = False) -> Detection:
        peak = self._peak_vel
        self._set_state(State.FALL_CONFIRMED)
        self._impact_t = None
        return Detection(t, kind, peak, self._last_angle, sensor)
