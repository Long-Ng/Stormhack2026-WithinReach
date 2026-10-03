# Fall Detection: Phase 1 (Camera Detection)

Rule-based fall detection from MediaPipe pose keypoints. See `../../FALL_DETECTION_SPEC.md`.

## Setup (Windows, uses uv)

From `project-repo/`:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -r fall-detection\requirements.txt
.venv\Scripts\activate
```

Download the pose models (gitignored) into `fall-detection/models/`:

```powershell
$base = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
Invoke-WebRequest "$base/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile fall-detection\models\pose_landmarker_lite.task
Invoke-WebRequest "$base/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -OutFile fall-detection\models\pose_landmarker_full.task
```

## Notes

- Python 3.12, mediapipe 1.0.1. The legacy `mp.solutions.pose` API is **not** present in this release; use the Tasks API (`mediapipe.tasks.python.vision.PoseLandmarker`).
- `cv2` comes from `opencv-contrib-python`, which mediapipe installs. Do not also install `opencv-python`: both packages provide `cv2` and clash.
- On Windows, open the webcam with `cv2.VideoCapture(0, cv2.CAP_DSHOW)` for fast startup.

## Run

```powershell
cd fall-detection
python main.py                # webcam 0
python main.py --source clip.mp4
python main.py --params other.toml
pytest
```

## Tuning

Edit `params.toml` and restart. Every number there overrides the default in
`config.py`; delete a line to go back to the default. Unknown names or wrong
types stop the program with an error, so a typo cannot be silently ignored.
