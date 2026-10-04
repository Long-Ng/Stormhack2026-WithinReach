"""Measure the Gemini second opinion on fall datasets (needs a Gemini key).

    python eval_gemini.py <dataset_root> <glob> <out.json> [--hold 5] [--limit N]
    python eval_gemini.py "D:/.../CAUCAFall" "*/*/*.avi" cauca_gemini.json
    python eval_gemini.py "D:/.../GMDCSA24-...-master" "*/*/*.mp4" gmdcsa_gemini.json

Each video is replayed through the detector like a live run; the last frame is held
`--hold` s (dataset clips end right after the fall). Every confirmed fall is sent to
Gemini with the same frames the live system would send. Ground truth comes from the
path: a "Fall" folder (GMDCSA24) or a "Fall ..." activity (CAUCAFall) is a fall.
The summary shows how many false alarms Gemini would remove and how many real falls
it would wrongly call "not a fall".
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2

from config import Config
from detector import FallDetector
from features import FeatureExtractor
from gemini import FrameHistory, GeminiClient, guidance
from pose import PoseEstimator


def is_fall_video(rel: str) -> bool:
    parts = Path(rel).parts
    return any(p == "Fall" or p.startswith("Fall ") for p in parts[:-1])


def replay(path: str, cfg: Config, hold_s: float):
    """Yield (stream time, FrameHistory) at each confirmed fall."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    ext, det, hist = FeatureExtractor(cfg), FallDetector(cfg), FrameHistory()
    n, last = 0, None
    with PoseEstimator(cfg) as est:
        def step(frame, t):
            hist.add(frame, t)
            d = det.update(ext.update(est.process(frame, t * 1000.0), t), t)
            return d is not None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = n / fps
            if step(frame, t):
                yield t, hist
            last, n = frame, n + 1
        for i in range(int(hold_s * fps)):
            t = (n + i) / fps
            if last is not None and step(last, t):
                yield t, hist


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root"), p.add_argument("pattern"), p.add_argument("out")
    p.add_argument("--hold", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=0, help="only the first N videos (0 = all)")
    args = p.parse_args()
    cfg = Config.load()
    key = cfg.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        print("set gemini_api_key in params.local.toml or GEMINI_API_KEY", file=sys.stderr)
        return 1
    client = GeminiClient(key, cfg.gemini_model, fallback_models=(cfg.gemini_fallback_model,))
    vids = sorted(glob.glob(str(Path(args.root) / args.pattern)))
    if args.limit:
        vids = vids[:args.limit]
    rows = []
    for v in vids:
        rel = Path(v).relative_to(args.root).as_posix()
        truth = is_fall_video(rel)
        for t, hist in replay(v, cfg, args.hold):
            try:
                r = client.analyze(hist.recent(n=8, span_s=10.0, ref_t=t))
            except Exception as e:
                print(f"{rel}: gemini failed: {e!r}", file=sys.stderr)
                continue
            level, text = guidance(r, 0.0, cfg.emergency_number)
            row = {"video": rel, "truth_fall": truth, "t": round(t, 2), **vars(r),
                   "urgency": level, "guidance": text}
            rows.append(row)
            print(f"{rel}  truth={'FALL' if truth else 'adl '}  gemini_is_fall={r.is_fall}  "
                  f"{r.fall_type}/{r.position}/{r.movement}  {level}", flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))

    tp = sum(r["truth_fall"] and r["is_fall"] for r in rows)
    fn = sum(r["truth_fall"] and not r["is_fall"] for r in rows)
    fp = sum(not r["truth_fall"] and r["is_fall"] for r in rows)
    tn = sum(not r["truth_fall"] and not r["is_fall"] for r in rows)
    print(f"\n{len(rows)} detections sent to Gemini")
    print(f"  real falls:   {tp} confirmed, {fn} wrongly called 'not a fall'")
    print(f"  false alarms: {tn} rejected by Gemini, {fp} still called a fall")
    print("  fall types on real falls:",
          dict(Counter(r["fall_type"] for r in rows if r["truth_fall"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
