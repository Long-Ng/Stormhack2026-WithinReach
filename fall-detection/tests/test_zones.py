import json

import numpy as np

from config import Config
from detector import FallDetector, State
from tests.test_detector import DROP, FPS, LYING, STAND, feats
from zones import (RestZone, SceneScanner, SceneWatcher, ZoneStore, changed_fraction, thumbnail,
                   zones_from_gemini)

SIZE = (640, 480)


# --- zones from Gemini, with guardrails ---------------------------------------------------
def test_gemini_boxes_become_fractions_and_bad_ones_are_dropped():
    zones = zones_from_gemini([
        {"label": "bed", "box_2d": [500, 100, 900, 600]},       # ymin, xmin, ymax, xmax
        {"label": "Sofa", "box_2d": [400, 650, 700, 950]},
        {"label": "floor", "box_2d": [600, 0, 1000, 1000]},     # not a rest label
        {"label": "bed", "box_2d": [0, 0, 1000, 900]},          # 90% of the picture
        {"label": "armchair", "box_2d": [1, 2, 3]},             # malformed
    ])
    # Only the top 60% of each box: the floor strip in front of the furniture is left out.
    assert [(z.label, tuple(round(v, 3) for v in z.box)) for z in zones] == [
        ("bed", (0.1, 0.5, 0.6, 0.74)), ("sofa", (0.65, 0.4, 0.95, 0.58))]


def test_floor_in_front_of_a_tall_bed_box_is_not_resting():
    # The box Gemini gave for the GMDCSA24 bed, which reached down to the floor.
    store = ZoneStore("unused.json")
    store.zones = zones_from_gemini([{"label": "bed", "box_2d": [500, 96, 946, 864]}])
    size = (1280, 720)
    assert store.zone_at((500, 400), (800, 420), size) is not None   # lying on the mattress
    assert store.zone_at((500, 640), (800, 660), size) is None       # on the floor in front


def test_both_shoulders_and_hips_must_be_inside():
    store = ZoneStore("unused.json")
    store.zones = [RestZone("bed", (0.1, 0.5, 0.6, 0.9))]
    in_bed = store.zone_at((200, 300), (300, 330), SIZE)
    beside = store.zone_at((200, 200), (300, 260), SIZE)  # shoulders above the bed: on the floor
    assert in_bed.label == "bed" and beside is None
    assert store.zone_at(None, (300, 330), SIZE) is None


def test_store_keeps_manual_zones_and_survives_a_restart(tmp_path):
    path = tmp_path / "zones.json"
    path.write_text(json.dumps({"zones": [
        {"label": "sofa", "box": [0.6, 0.4, 0.9, 0.7], "source": "manual"},
        {"label": "bed", "box": [0.0, 0.0, 0.2, 0.2], "source": "gemini"}]}))
    store = ZoneStore(path)
    store.replace_detected([RestZone("bed", (0.1, 0.5, 0.6, 0.9))], "A bedroom.")
    again = ZoneStore(path)
    assert [(z.label, z.source) for z in again.zones] == [("sofa", "manual"), ("bed", "gemini")]
    assert again.description == "A bedroom." and again.analyzed_t > 0


def test_unreadable_file_means_no_zones(tmp_path):
    (tmp_path / "zones.json").write_text("{not json")
    assert ZoneStore(tmp_path / "zones.json").zones == []


# --- detector: lying in a rest zone is not a fall -------------------------------------------
def detections(in_rest_zone, phases):
    det, out, frame = FallDetector(Config()), [], 0
    for duration, values in phases:
        for _ in range(round(duration * FPS)):
            det.in_rest_zone = in_rest_zone(frame / FPS)
            if det.update(feats(frame / FPS, **values), frame / FPS) is not None:
                out.append(frame / FPS)
            frame += 1
    return det, out


def test_lying_down_in_bed_is_not_a_fall_but_on_the_floor_is():
    lie_down = [(2, STAND), (0.3, DROP), (20, LYING)]  # long enough for prolonged_lying too
    in_bed, events = detections(lambda t: True, lie_down)
    assert events == [] and in_bed.state is State.UPRIGHT
    _, events = detections(lambda t: False, lie_down)
    assert len(events) == 1


def test_rolling_out_of_bed_onto_the_floor_is_a_fall():
    # In bed for 10 s, then out of the zone (on the floor), lying: prolonged_lying fires.
    _, events = detections(lambda t: t < 12.0, [(2, STAND), (0.3, DROP), (40, LYING)])
    assert len(events) == 1 and events[0] > 12.0


# --- scene change ---------------------------------------------------------------------------
def room(sofa_x=100, brightness=1.0):
    img = np.full((480, 640, 3), 120, np.uint8)
    img[:240] = 170  # wall
    img[300:420, sofa_x:sofa_x + 250] = 60  # dark sofa
    img[150:250, 450:560] = 230  # window
    return np.clip(img * brightness, 0, 255).astype(np.uint8)


def test_lighting_change_is_not_a_scene_change_but_moved_furniture_is():
    ref = thumbnail(room())
    assert changed_fraction(ref, thumbnail(room(brightness=0.6))) < 0.05
    assert changed_fraction(ref, thumbnail(room(sofa_x=360))) > 0.15


def feed(w, frame, start, seconds, person=False):
    hits = []
    for i in range(int(seconds * 10)):
        t = start + i / 10
        if w.update(frame, t, person):
            hits.append(round(t, 1))
            w.scanned(frame, t)
    return hits


def test_first_scan_waits_for_an_empty_room():
    w = SceneWatcher(empty_s=5.0, min_interval_s=600.0, have_zones=False)
    assert feed(w, room(), 0, 10, person=True) == []
    assert feed(w, room(), 10, 10) == [15.0]


def test_rescan_after_sustained_change_and_not_too_often():
    w = SceneWatcher(empty_s=5.0, change_frac=0.1, change_s=20.0, min_interval_s=600.0,
                     have_zones=False)
    feed(w, room(), 0, 10)                                   # first scan at 5 s
    assert feed(w, room(sofa_x=360), 700, 10) == []          # changed, but not for 20 s yet
    assert feed(w, room(sofa_x=360), 710, 30) == [720.0]     # 20 s after the change, seen at 700
    assert feed(w, room(sofa_x=100), 800, 60) == []          # changed back: within 10 min


def test_zones_loaded_from_a_file_are_not_rescanned_until_the_room_changes():
    w = SceneWatcher(empty_s=5.0, change_s=20.0, have_zones=True)
    assert feed(w, room(), 0, 60) == []
    assert feed(w, room(sofa_x=360), 60, 30) != []


def test_failed_scan_is_retried_later(tmp_path):
    w = SceneWatcher(have_zones=False)

    def boom(frame):
        raise TimeoutError("gemini down")
    s = SceneScanner(boom, ZoneStore(tmp_path / "z.json"), w)
    w.scanned(room(), 0.0)
    s._run(room())
    assert w.need_scan and w.reference is None and not s.busy
