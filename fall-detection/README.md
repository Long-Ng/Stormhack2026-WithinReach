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
python main.py                # DroidCam phone if streaming, else webcam 0
python main.py --source clip.mp4
python main.py --params other.toml
pytest
```

## Phone camera (DroidCam)

With the DroidCam Windows client installed and the phone app streaming,
`python main.py` uses the phone automatically. The "DroidCam Video" device is
always listed, but with no phone it only sends a placeholder card and then solid
green, so the program waits up to `droidcam_probe_s` for real frames and
otherwise falls back to `camera_index`. Set `prefer_droidcam = false` in
`params.toml` to skip the check. DroidCam opens only through Media Foundation
(DirectShow raises an error), so cameras try DirectShow first, then MSMF.

Without the client, use the phone's Wi-Fi stream directly:
`python main.py --source http://<phone-ip>:4747/video`

## Events

A confirmed fall prints one line, appends one JSON object to `events/events.jsonl`,
saves the frame as `events/<date-time>.jpg` (the dashboard lists these), and records
`events/<date-time>.mp4`: the 30 s before the confirmation, then everything after
until the person has been back up for 10 s (capped at 5 min; see `[events]` in
`params.toml`). The clip finishes writing in the background, or on exit. To add an
alert channel, write a class with `send(event)` and add it to `sinks` in `main.py`.

## Tuning

Edit `params.toml` and restart. Every number there overrides the default in
`config.py`; delete a line to go back to the default. Unknown names or wrong
types stop the program with an error, so a typo cannot be silently ignored.
