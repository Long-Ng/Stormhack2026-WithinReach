"""All tunable settings live here. Nothing elsewhere should hardcode a threshold."""

from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"


@dataclass
class Config:
    # Pose estimation
    # Full model: better on lying/occluded poses; still ~30 fps on CPU here. Lite is the fallback.
    model_path: str = str(MODELS_DIR / "pose_landmarker_full.task")
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Capture
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480

    # Features
    min_visibility: float = 0.5  # landmark visibility cutoff

    # Display
    fps_smoothing: float = 0.9  # EMA factor for the FPS readout
