"""Fall detection entry point.

    python main.py                    # webcam 0, overlay on
    python main.py --source 1         # other camera index
    python main.py --source clip.mp4  # replay a video file
    python main.py --log-features f.csv  # dump per-frame features for tuning
    python main.py --no-display       # headless
    python main.py --params other.toml   # use a different parameter file
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time

import cv2

from config import Config
from detector import FallDetector
from features import FeatureExtractor, FeatureLogger
from overlay import draw_overlay
from pose import PoseEstimator

WINDOW = "Fall Detection"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Camera-based fall detection")
    p.add_argument("--source", default=None,
                   help="camera index (e.g. 0, 1) or path to a video file")
    p.add_argument("--log-features", metavar="CSV", default=None,
                   help="write per-frame features to a CSV file")
    p.add_argument("--params", metavar="TOML", default=None,
                   help="parameter file (default: params.toml next to main.py)")
    p.add_argument("--no-display", action="store_true", help="run headless")
    return p.parse_args()


def open_source(source: str | None, cfg: Config) -> tuple[cv2.VideoCapture, bool]:
    """Return (capture, is_file)."""
    if source is None or source.isdigit():
        index = cfg.camera_index if source is None else int(source)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # DirectShow opens fast on Windows
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
        return cap, False
    return cv2.VideoCapture(source), True


def main() -> int:
    args = parse_args()
    cfg = Config.load(args.params) if args.params else Config.load()

    cap, is_file = open_source(args.source, cfg)
    if not cap.isOpened():
        print(f"Could not open source: {args.source if args.source is not None else cfg.camera_index}",
              file=sys.stderr)
        return 1

    fps = 0.0
    last_wall = time.perf_counter()
    start_wall = last_wall

    extractor = FeatureExtractor(cfg)
    detector = FallDetector(cfg)

    with contextlib.ExitStack() as stack:
        estimator = stack.enter_context(PoseEstimator(cfg))
        logger = stack.enter_context(FeatureLogger(args.log_features)) if args.log_features else None
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Video files use their own timestamps so replays are reproducible.
            if is_file:
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            else:
                ts_ms = (time.perf_counter() - start_wall) * 1000.0

            lms = estimator.process(frame, ts_ms)
            feats = extractor.update(lms, ts_ms / 1000.0)
            detection = detector.update(feats, ts_ms / 1000.0)
            if detection is not None:
                # Placeholder until the sinks land in milestone 4.
                print(f"[{detection.t:7.2f}s] {detection.kind.upper()}  "
                      f"peak_vel={detection.peak_hip_vel:.2f}  angle={detection.torso_angle:.0f}")
            if logger is not None:
                logger.log(ts_ms / 1000.0, feats)

            now = time.perf_counter()
            inst = 1.0 / max(now - last_wall, 1e-6)
            last_wall = now
            fps = inst if fps == 0.0 else cfg.fps_smoothing * fps + (1 - cfg.fps_smoothing) * inst

            if not args.no_display:
                draw_overlay(frame, lms, feats, detector, fps, cfg.min_visibility)
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed with the X button

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
