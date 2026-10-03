"""Fall detection entry point.

    python main.py                    # DroidCam phone if streaming, else webcam 0
    python main.py --source 1         # other camera index
    python main.py --source http://192.168.1.5:4747/video  # DroidCam over Wi-Fi
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

from camera import open_source
from config import Config
from cover import CoverReset
from detector import FallDetector
from events import ConsoleSink, FallEvent, FileSink, dispatch, save_snapshot
from features import FeatureExtractor, FeatureLogger
from imu import open_wearable
from overlay import draw_overlay, draw_skeleton
from pose import PoseEstimator

WINDOW = "Fall Detection"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Camera-based fall detection")
    p.add_argument("--source", default=None,
                   help="camera index (e.g. 0, 1), stream URL, or path to a video file")
    p.add_argument("--log-features", metavar="CSV", default=None,
                   help="write per-frame features to a CSV file")
    p.add_argument("--params", metavar="TOML", default=None,
                   help="parameter file (default: params.toml next to main.py)")
    p.add_argument("--no-display", action="store_true", help="run headless")
    return p.parse_args()


def handle_detection(detection, frame, lms, cfg: Config, sinks) -> None:
    """Snapshot the frame (with skeleton, before the debug text) and send the event."""
    now = time.time()
    snap = frame.copy()
    if lms is not None:
        draw_skeleton(snap, lms, cfg.min_visibility)
    try:
        snapshot_path = save_snapshot(snap, cfg.events_dir, now)
    except Exception as e:
        print(f"snapshot failed: {e!r}", file=sys.stderr)
        snapshot_path = None
    dispatch(FallEvent.from_detection(detection, now, snapshot_path), sinks)


def main() -> int:
    args = parse_args()
    cfg = Config.load(args.params) if args.params else Config.load()

    source = open_source(args.source, cfg)
    cap, is_file = source.cap, source.is_file
    if not cap.isOpened():
        print(f"Could not open {source.label}", file=sys.stderr)
        return 1
    print(f"Using {source.label}")

    fps = 0.0
    last_wall = time.perf_counter()
    start_wall = last_wall

    extractor = FeatureExtractor(cfg)
    detector = FallDetector(cfg)
    cover = CoverReset(cfg) if cfg.cover_reset else None
    # Phone accelerometer; only for live sources, since its clock is the PC's.
    wearable = None if is_file else open_wearable(cfg)
    # Add new alert channels here; nothing else needs to change.
    sinks = [ConsoleSink(), FileSink(cfg.events_dir)]

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

            # Cover the lens for cover_reset_s to start detection over.
            if cover is not None and cover.update(frame, ts_ms / 1000.0):
                extractor = FeatureExtractor(cfg)
                detector = FallDetector(cfg)
                print("Reset (camera covered)")

            lms = estimator.process(frame, ts_ms)
            feats = extractor.update(lms, ts_ms / 1000.0)
            phone_still = None
            if wearable is not None:
                for hit in wearable.poll():
                    detector.add_impact(hit.t - start_wall)  # perf_counter -> stream time
                    print(f"Phone impact {hit.peak / 9.81:.1f} g")
                if wearable.connected:
                    phone_still = wearable.still_for(time.perf_counter())
            detection = detector.update(feats, ts_ms / 1000.0, phone_still)
            if detection is not None:
                if detection.sensor:
                    print("(confirmed with phone sensor)")
                handle_detection(detection, frame, lms, cfg, sinks)
            if logger is not None:
                logger.log(ts_ms / 1000.0, feats)

            now = time.perf_counter()
            inst = 1.0 / max(now - last_wall, 1e-6)
            last_wall = now
            fps = inst if fps == 0.0 else cfg.fps_smoothing * fps + (1 - cfg.fps_smoothing) * inst

            if not args.no_display:
                draw_overlay(frame, lms, feats, detector, fps, cfg.min_visibility, cover, wearable)
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed with the X button

    if wearable is not None:
        wearable.close()
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
