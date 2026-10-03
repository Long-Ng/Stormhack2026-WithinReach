"""All tunable settings live here. Nothing elsewhere should hardcode a threshold.

The values below are defaults. `params.toml` overrides any of them for tuning
without touching code (see `Config.load`).
"""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_PARAMS = Path(__file__).parent / "params.toml"
PATH_KEYS = {"model_path", "events_dir"}  # relative values resolve against the params file


@dataclass
class Config:
    # Pose estimation
    # Full model: better on lying/occluded poses; still ~30 fps on CPU here. Lite is the fallback.
    model_path: str = str(MODELS_DIR / "pose_landmarker_full.task")
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Capture
    camera_index: int = 0  # used when no DroidCam phone is streaming
    prefer_droidcam: bool = True  # with no --source, use a connected DroidCam phone if there is one
    droidcam_probe_s: float = 3.0  # how long to wait for a real picture from DroidCam at startup
    placeholder_max_std: float = 5.0  # frames this flat are DroidCam's "no phone" placeholder
    droidcam_live_frames: int = 3  # distinct real frames needed to count DroidCam as streaming
    frame_width: int = 640
    frame_height: int = 480

    # Features
    min_visibility: float = 0.5  # landmark visibility cutoff
    ema_alpha: float = 0.4  # smoothing for torso_angle, hip_vel, motion (1.0 = no smoothing)
    vel_window_s: float = 0.5  # window for peak hip velocity
    ref_upright_angle: float = 20.0  # torso_ref only learns from frames more upright than this
    torso_ref_frames: int = 30  # rolling median length for torso_ref

    # Detector (state machine)
    fall_vel: float = 1.5  # torso lengths/s downward; real falls peak ~2.5-3.5, sitting ~0.8
    fall_window_s: float = 1.5  # time allowed from fast descent to horizontal
    lying_angle: float = 60.0  # degrees from vertical
    lying_aspect: float = 1.2  # bbox width / height
    low_hip: float = 0.6  # normalized hip height
    still_motion: float = 0.3  # torso lengths/s
    confirm_s: float = 3.0  # stillness on the ground before confirming
    still_grace_s: float = 0.5  # unsteady frames shorter than this pause the stillness timer instead of resetting it
    upright_angle: float = 30.0  # degrees
    recover_s: float = 1.0  # upright time to reset to UPRIGHT
    lost_grace_s: float = 1.0  # pose-loss tolerance while down
    slow_fall_s: float = 15.0  # down this long without a fast descent -> prolonged_lying

    # Cover-to-reset: lens covered (dark AND flat) for cover_reset_s resets detection
    cover_reset: bool = True
    cover_reset_s: float = 3.0
    cover_max_brightness: float = 40.0  # mean pixel value 0-255
    cover_max_std: float = 12.0  # largest per-channel std; a covered lens is a flat blur

    # Wearable accelerometer (Phyphox remote access); live sources only, not file replays
    imu_url: str = ""  # e.g. "http://172.16.164.134:8080"; empty = no wearable
    imu_poll_s: float = 0.05  # how often to fetch new samples
    imu_timeout_s: float = 2.0  # HTTP timeout per request
    imu_impact_ms2: float = 20.0  # departure from rest counted as an impact (~2 g)
    imu_refractory_s: float = 1.0  # one impact per bounce sequence
    imu_match_s: float = 2.0  # impact this close to the camera's fall counts as the same fall
    imu_hold_s: float = 10.0  # an unmatched impact is forgotten after this long
    imu_still_ms2: float = 1.5  # phone below this is "still"
    imu_still_s: float = 3.0  # impact + phone still this long + person unseen/down -> sensor_fall

    # Events
    events_dir: str = str(Path(__file__).parent / "events")  # events.jsonl + snapshots

    # Display
    fps_smoothing: float = 0.9  # EMA factor for the FPS readout

    @classmethod
    def load(cls, path: str | Path | None = DEFAULT_PARAMS) -> "Config":
        """Defaults overridden by a TOML file. Section headers are only for grouping.

        A missing default file is fine; a missing explicit file, an unknown key or
        a wrong type is an error, so typos do not silently fall back to defaults.
        """
        cfg = cls()
        if path is None:
            return cfg
        path = Path(path)
        if not path.exists():
            if path == DEFAULT_PARAMS:
                return cfg
            raise FileNotFoundError(f"params file not found: {path}")
        with open(path, "rb") as f:
            data = tomllib.load(f)

        flat: dict = {}
        for key, value in data.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value

        types = {f.name: f.type for f in fields(cls)}
        for key, value in flat.items():
            if key not in types:
                raise ValueError(f"{path}: unknown parameter '{key}'")
            want = types[key]
            if want is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if not isinstance(value, want) or (isinstance(value, bool) and want is not bool):
                raise ValueError(f"{path}: '{key}' should be {want.__name__}, got {value!r}")
            if key in PATH_KEYS and not Path(value).is_absolute():
                value = str(path.parent / value)  # relative to the params file, not the cwd
            setattr(cfg, key, value)
        return cfg
