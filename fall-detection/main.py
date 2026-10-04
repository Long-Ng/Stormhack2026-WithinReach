"""Fall detection entry point.

python main.py                                        # DroidCam phone if streaming, else webcam 0
python main.py --source 1                             # other camera index
python main.py --source http://192.168.1.5:4747/video # DroidCam over Wi-Fi
python main.py --source clip.mp4                      # replay a video file
python main.py --log-features f.csv                   # dump per-frame features for tuning
python main.py --no-display                           # headless
python main.py --params other.toml                    # use a different parameter file
python main.py --port 5050                            # dashboard server port (default 5000)
python main.py --no-dashboard                         # do not start the dashboard server
python main.py --ntfy my-secret-topic --name Nick     # extra phone push through the ntfy app

While running: desktop dashboard http://localhost:5000/ , monitor page http://<PC address>:5000/monitor,
               person page http://<PC address>:5000/granny
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import cv2

from camera import open_source
from alerts import open_alerts
from clips import ClipRecorder
from config import Config
from cover import CoverReset
from detector import FallDetector, State
from events import ConsoleSink, FallEvent, FileSink, dispatch, save_snapshot
from features import FeatureExtractor, FeatureLogger
from gemini import FallAnalyst, FrameHistory, GeminiClient, report_dict
from zones import SceneScanner, SceneWatcher, ZoneStore, draw_zones
from imu import open_wearable
from inject import KEYS as INJECT_KEYS, ClipInjector
from overlay import draw_overlay, draw_skeleton, draw_text, render_privacy_frame
from pose import PoseEstimator
from smoothing import SkeletonStabilizer

WINDOW = "Fall Detection"

# True  = saved event snapshots include the skeleton (original behaviour)
# False = clean snapshot (what the dashboard shows is just the person)
SNAPSHOT_SKELETON = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Camera-based fall detection")
    p.add_argument("--source", default=None,
                   help="camera index (e.g. 0, 1), stream URL, or path to a video file")
    p.add_argument("--log-features", metavar="CSV", default=None,
                   help="write per-frame features to a CSV file")
    p.add_argument("--params", metavar="TOML", default=None,
                   help="parameter file (default: params.toml next to main.py)")
    p.add_argument("--no-display", action="store_true", help="run headless")
    p.add_argument("--inject", metavar="VIDEO_OR_FOLDER", default=None,
                   help="demo: press I or Space in the camera window to play this clip as if it were live")
    p.add_argument("--port", type=int, default=5000, help="dashboard server port")
    p.add_argument("--no-dashboard", action="store_true", help="do not start the dashboard server")
    p.add_argument("--ntfy", metavar="TOPIC", default=os.environ.get("FALL_NTFY_TOPIC"),
                   help="ntfy topic for an extra phone push (or set env FALL_NTFY_TOPIC)")
    p.add_argument("--ntfy-server", default=os.environ.get("FALL_NTFY_SERVER", "https://ntfy.sh"),
                   help="ntfy server (default https://ntfy.sh)")
    p.add_argument("--name", default="Your family member", help="name of the monitored person (phone page + push)")
    p.add_argument("--room", default="Living room", help="room name shown on the phone page")
    p.add_argument("--address", default="", help="home address shown on the phone page")
    p.add_argument("--phone", default="911", help="emergency number for the Call button")
    p.add_argument("--countdown", type=int, default=120,
                   help="seconds before the phone page says time is up to call (0 = off; it never calls by itself)")
    return p.parse_args()


def person_box(lms, min_vis):
    """Pixel box (x1, y1, x2, y2) around the visible landmarks, or None. lms is (33, 4): x_px, y_px, z, vis."""
    if lms is None:
        return None
    pts = lms[lms[:, 3] >= min_vis]
    if len(pts) == 0:
        return None
    pad = 20
    return (pts[:, 0].min() - pad, pts[:, 1].min() - pad, pts[:, 0].max() + pad, pts[:, 1].max() + pad)


def handle_detection(detection, frame, lms, cfg: Config, sinks, streamer=None,
                     recorder: ClipRecorder | None = None) -> None:
    """Snapshot the frame (before the debug text), start the clip, send the event,
    flag the live stream."""
    now = time.time()
    snap = frame.copy()
    if SNAPSHOT_SKELETON and lms is not None:
        draw_skeleton(snap, lms, cfg.min_visibility)
    try:
        snapshot_path = save_snapshot(snap, cfg.events_dir, now)
    except Exception as e:
        print(f"snapshot failed: {e!r}", file=sys.stderr)
        snapshot_path = None
    clip_path = recorder.trigger(now, detection.t) if recorder is not None else None
    dispatch(FallEvent.from_detection(detection, now, snapshot_path, clip_path), sinks)
    if streamer is not None:
        try:  # a dashboard problem must never stop the detector
            streamer.alert(f"{detection.kind.replace('_', ' ').upper()} DETECTED")
        except Exception as e:
            print(f"dashboard alert failed: {e!r}", file=sys.stderr)


def start_dashboard(args, cfg: Config):
    """Start the dashboard/video server. Never lets a dashboard problem stop the detector."""
    try:
        from stream import Streamer
        s = Streamer(port=args.port, events_dir=cfg.events_dir)
        s.set_config(name=args.name, room=args.room, address=args.address, phone=args.phone,
                     countdown=args.countdown, push=True, talk=False)
        return s
    except Exception as e:
        print(f"[dashboard] disabled: {e!r}", file=sys.stderr)
        return None


def main() -> int:
    args = parse_args()
    cfg = Config.load(args.params) if args.params else Config.load()

    source = open_source(args.source, cfg)
    cap, is_file = source.cap, source.is_file
    injector = None
    inject_base_ms = live_offset_ms = 0.0  # timeline bookkeeping for demo clips
    last_ts_ms = -1.0
    if args.inject:
        if is_file:
            print("[inject] only works with a live camera; ignoring --inject")
        else:
            try:
                injector = ClipInjector(args.inject)
                print(f"[inject] press I or Space in the camera window to play: "
                      + ", ".join(c.name for c in injector.clips))
            except FileNotFoundError as e:
                print(f"[inject] {e}", file=sys.stderr)
    if not cap.isOpened():
        print(f"Could not open {source.label}", file=sys.stderr)
        return 1
    print(f"Using {source.label}")

    streamer = None if args.no_dashboard else start_dashboard(args, cfg)
    if streamer is not None:
        streamer.privacy = cfg.privacy_view

    fps = 0.0
    last_wall = time.perf_counter()
    start_wall = last_wall

    extractor = FeatureExtractor(cfg)
    detector = FallDetector(cfg)
    stabilizer = SkeletonStabilizer(cfg)  # calmer skeleton for display only
    was_private = False
    cover = CoverReset(cfg) if cfg.cover_reset else None
    # Phone accelerometer; only for live sources, since its clock is the PC's.
    wearable = None if is_file else open_wearable(cfg)
    # Add new alert channels here; nothing else needs to change.
    sinks = [ConsoleSink(), FileSink(cfg.events_dir)]
    # Phone alerts: person first, then the monitor. Replies arrive on the dashboard server.
    alerts = open_alerts(cfg, args.port) if streamer is not None else None
    if alerts is not None:
        manager, publisher = alerts
        sinks.append(manager)
        for path, handler in manager.routes().items():
            streamer.add_route(path, handler)
        for path, handler in manager.upload_routes().items():
            streamer.add_post_route(path, handler)
    # Optional extra push to the ntfy app (snapshot attached, buttons: camera / talk / call).
    if args.ntfy:
        try:
            from notify import NtfySink
            sinks.append(NtfySink(args.ntfy, args.ntfy_server, args.port, args.name, args.phone))
        except Exception as e:
            print(f"[notify] disabled: {e!r}", file=sys.stderr)
    recorder = ClipRecorder(cfg.events_dir, cfg.clip_pre_s, cfg.clip_tail_s, cfg.clip_max_after_s,
                            cfg.clip_max_width, cfg.clip_max_fps)
    # Gemini fall analysis; its reports reach the monitor through the phone alerts.
    history = analyst = None
    gemini_key = cfg.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if alerts is not None and gemini_key:
        history = FrameHistory()
        analyst = FallAnalyst(
            GeminiClient(gemini_key, cfg.gemini_model, fallback_models=(cfg.gemini_fallback_model,)),
            on_report=lambda inc_id, r, minutes, still: manager.add_report(
                inc_id, report_dict(r, minutes, cfg.emergency_number, still)),
            is_open=manager.is_open, update_s=cfg.gemini_update_s,
            max_updates=int(cfg.gemini_max_minutes * 60 / cfg.gemini_update_s))
        if streamer is not None:
            streamer.set_config(gemini_falls=True)  # the monitor page words Privacy view honestly
        print(f"[gemini] fall analysis on ({cfg.gemini_model}), "
              f"checks every {cfg.gemini_update_s:g} s for {cfg.gemini_max_minutes:g} min")
    elif alerts is not None:
        print("[gemini] off: set gemini_api_key in params.local.toml to describe falls")

    # Rest zones (bed, sofa): lying there is not a fall. Gemini finds them when it can.
    zone_store = watcher = scanner = None
    if cfg.rest_zones:
        zone_store = ZoneStore(cfg.zones_file)
        if gemini_key:
            watcher = SceneWatcher(cfg.scene_empty_s, cfg.scene_change_frac, cfg.scene_change_s,
                                   cfg.scene_min_interval_s, have_zones=bool(zone_store.zones))
            scanner = SceneScanner(
                GeminiClient(gemini_key, cfg.gemini_model,
                             fallback_models=(cfg.gemini_fallback_model,)).analyze_scene,
                zone_store, watcher)
            if streamer is not None:
                streamer.set_config(gemini_scan=True)  # empty-room pictures go to Gemini
        names = ", ".join(z.label for z in zone_store.zones) or "none yet"
        print(f"[zones] rest zones: {names}"
              + ("; Gemini rescans when the empty room changes" if scanner else
                 "; add a Gemini key for automatic room scans"))
        if streamer is not None:
            streamer.add_route("/zones.json", lambda q: (
                200, "application/json", json.dumps(zone_store.as_json()).encode()))

    with contextlib.ExitStack() as stack:
        stack.callback(recorder.close)  # finish the clip in progress on exit
        if alerts is not None:
            stack.callback(publisher.close)  # send queued notifications before exiting
        estimator = stack.enter_context(PoseEstimator(cfg))
        logger = stack.enter_context(FeatureLogger(args.log_features)) if args.log_features else None
        while True:
            if injector is not None and injector.active:
                ok, frame = injector.read()
                if not ok:  # clip finished
                    print("[inject] clip ended: back to the live camera")
                    extractor = FeatureExtractor(cfg)  # no fake jump across the switch
                    live_offset_ms = last_ts_ms - (time.perf_counter() - start_wall) * 1000.0
                    for _ in range(5):
                        cap.grab()  # skip camera frames queued while the clip played
                    ok, frame = cap.read()
            else:
                ok, frame = cap.read()
            if not ok:
                break
            if alerts is not None:
                manager.tick()  # escalate to the monitor when the person has not replied

            # Video files use their own timestamps so replays are reproducible; so does an
            # injected demo clip (continuing from where the live clock was).
            if injector is not None and injector.active:
                ts_ms = inject_base_ms + injector.elapsed_ms
            elif is_file:
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            else:
                ts_ms = (time.perf_counter() - start_wall) * 1000.0 + live_offset_ms
            ts_ms = max(ts_ms, last_ts_ms + 1.0)  # strictly increasing across switches
            last_ts_ms = ts_ms

            # Cover the lens for cover_reset_s to start detection over.
            if cover is not None and cover.update(frame, ts_ms / 1000.0):
                extractor = FeatureExtractor(cfg)
                detector = FallDetector(cfg)
                print("Reset (camera covered)")

            if history is not None:
                history.add(frame, ts_ms / 1000.0)  # clean frame, before any overlay
            lms = estimator.process(frame, ts_ms)
            shown_lms = stabilizer.update(lms, ts_ms / 1000.0)  # drawing only; detection uses lms
            # What the monitor may see: the camera picture, or in Privacy view a stick figure
            # with no camera pixels (live view, snapshots and clips alike).
            private = streamer is not None and streamer.privacy
            if private and not was_private:
                recorder.clear_preroll()  # no camera footage from before the switch in clips
            was_private = private
            view = render_privacy_frame(frame.shape, shown_lms, cfg.min_visibility) if private else frame
            if streamer is not None:  # clean frame, before the debug overlay is drawn
                try:
                    streamer.update(view)
                except Exception as e:
                    print(f"[dashboard] update failed: {e!r}", file=sys.stderr)
            feats = extractor.update(lms, ts_ms / 1000.0)
            rest_zone = None
            if zone_store is not None:
                if feats is not None:
                    rest_zone = zone_store.zone_at(feats.shoulder_xy, feats.hip_xy,
                                                   (frame.shape[1], frame.shape[0]))
                detector.in_rest_zone = rest_zone is not None
                if watcher is not None and watcher.update(frame, ts_ms / 1000.0, lms is not None):
                    watcher.scanned(frame, ts_ms / 1000.0)
                    scanner.start(frame)  # clean frame: no overlay drawn yet
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
                # Privacy view: the snapshot is the stick figure too (it already has the skeleton).
                handle_detection(detection, view, None if private else lms, cfg, sinks, streamer, recorder)
                if analyst is not None and manager.last_incident_id:
                    analyst.start(manager.last_incident_id, history, ts_ms / 1000.0)
            if analyst is not None:
                if detector.state is State.UPRIGHT:
                    analyst.stop()  # back up: no more updates
                analyst.tick(history, ts_ms / 1000.0)
            # Clean frame (no overlay yet); the clip runs until the person is back up.
            recorder.add(view, ts_ms / 1000.0, detector.state is not State.UPRIGHT)

            # Keep the dashboard alert (and the red box around the person) on while the fall stays confirmed.
            if streamer is not None:
                try:
                    confirmed = detector.state is State.FALL_CONFIRMED
                    streamer.set_box(person_box(lms, cfg.min_visibility) if confirmed else None)
                    if confirmed:
                        streamer.alert("FALL CONFIRMED")
                except Exception as e:
                    print(f"[dashboard] alert failed: {e!r}", file=sys.stderr)
            if logger is not None:
                logger.log(ts_ms / 1000.0, feats)

            now = time.perf_counter()
            inst = 1.0 / max(now - last_wall, 1e-6)
            last_wall = now
            fps = inst if fps == 0.0 else cfg.fps_smoothing * fps + (1 - cfg.fps_smoothing) * inst

            if not args.no_display:
                draw_overlay(frame, shown_lms, feats, detector, fps, cfg.min_visibility, cover, wearable)
                if zone_store is not None:
                    draw_zones(frame, zone_store.zones, rest_zone)
                if injector is not None and injector.active:  # this window only, not the dashboard
                    draw_text(frame, f"DEMO CLIP: {injector.name}", (frame.shape[1] // 2 - 140, 30),
                              (0, 0, 255), scale=0.7)
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    break
                if key in INJECT_KEYS:
                    if injector is None:
                        print("[inject] start main.py with --inject <video or folder> to use I / Space")
                    else:
                        print(f"[inject] {injector.toggle()}")
                        extractor = FeatureExtractor(cfg)  # no fake jump across the switch
                        if injector.active:
                            inject_base_ms = last_ts_ms + 1.0
                        else:
                            live_offset_ms = last_ts_ms - (time.perf_counter() - start_wall) * 1000.0
                            for _ in range(5):
                                cap.grab()
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed with the X button

    if wearable is not None:
        wearable.close()
    if streamer is not None:
        streamer.close()
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
