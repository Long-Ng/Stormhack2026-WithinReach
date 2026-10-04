"""Phone alerts through ntfy: ask the person first, then escalate to the monitor.

    fall confirmed --> person's phone: "Did you fall?"  [I'm OK] [I need help] [Call]
        "I'm OK"                              --> resolved (logged as a false alarm)
        "I need help", or no reply in time    --> monitor's phone + dashboard

ntfy (https://ntfy.sh) is a push service with a ready-made phone app, so the phones
need no app of our own: each subscribes to a topic. Button presses come back to this
PC as HTTP requests to /respond on the dashboard server, so the phones must be able
to reach it (same Wi-Fi for the demo).

`AlertManager` is an AlertSink; main.py calls `tick()` every frame to run the reply
timeout, and registers the routes from `routes()` on the dashboard server.
"""

from __future__ import annotations

import json
import queue
import socket
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from config import Config
from events import FallEvent

PERSON_PAGE = Path(__file__).parent / "person.html"

# Quick messages on the person page: key -> (text, means the person needs help)
QUICK_MESSAGES = {
    "cant_get_up": ("I can't get up", True),
    "hurt": ("I'm hurt", True),
    "call_me": ("Please call me", True),
    "ok_now": ("I'm OK now", False),
}
VOICE_TYPES = {"webm": "audio/webm", "ogg": "audio/ogg", "mp4": "audio/mp4"}

# Incident status
WAITING = "waiting"     # person notified, no reply yet
OK = "ok"               # person said they are fine
HELP = "help"           # person asked for help
NO_REPLY = "no_reply"   # timed out; monitor alerted


@dataclass
class Incident:
    id: str
    t: float                    # epoch seconds of the detection
    kind: str
    status: str
    snapshot: str | None        # file name inside events_dir
    replied_t: float | None = None


def lan_url(port: int) -> str:
    """http://<this PC's LAN IP>:port, the address phones on the same Wi-Fi can reach."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packet is sent; just picks the outgoing interface
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    return f"http://{ip}:{port}"


class NtfyPublisher:
    """Posts notifications on a background thread so the camera loop never waits."""

    def __init__(self, server: str, timeout_s: float = 5.0):
        self.server = server.rstrip("/")
        self.timeout_s = timeout_s
        self._q: queue.Queue[dict | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def publish(self, message: dict) -> None:
        self._q.put(message)

    def _run(self) -> None:
        while (msg := self._q.get()) is not None:
            try:
                req = urllib.request.Request(self.server, data=json.dumps(msg).encode(),
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=self.timeout_s).read()
            except Exception as e:
                print(f"[alerts] ntfy publish to '{msg.get('topic')}' failed: {e!r}",
                      file=sys.stderr)

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=self.timeout_s)


class AlertManager:
    def __init__(self, cfg: Config, base_url: str, publish: Callable[[dict], None],
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.base_url = base_url.rstrip("/")
        self.publish = publish
        self.clock = clock
        self.incidents: dict[str, Incident] = {}
        self.messages: list[dict] = []  # person -> monitor: {"t", "kind", "text", "file"}
        self.voice_dir = Path(cfg.events_dir) / "voice"
        self._lock = threading.Lock()  # replies arrive on the server's threads

    # --- AlertSink -----------------------------------------------------------------
    def send(self, event: FallEvent) -> None:
        snap = Path(event.snapshot_path).name if event.snapshot_path else None
        base_id = datetime.fromtimestamp(event.timestamp).strftime("%Y%m%d-%H%M%S")
        with self._lock:
            inc_id, n = base_id, 1
            while inc_id in self.incidents:  # two events in the same second
                n += 1
                inc_id = f"{base_id}-{n}"
            inc = Incident(id=inc_id, t=event.timestamp, kind=event.kind, status=WAITING,
                           snapshot=snap)
            self.incidents[inc.id] = inc
        self.publish(self._person_message(inc))

    # --- called by main.py every frame ----------------------------------------------
    def tick(self) -> None:
        now = self.clock()
        with self._lock:
            due = [i for i in self.incidents.values()
                   if i.status == WAITING and now - i.t >= self.cfg.reply_timeout_s]
            for inc in due:
                inc.status = NO_REPLY
        for inc in due:
            self.publish(self._monitor_message(inc))

    # --- replies from the person's phone --------------------------------------------
    def respond(self, incident_id: str, answer: str) -> str:
        """Handle a button press. Returns a short text for the HTTP reply."""
        if answer not in (OK, HELP):
            return "unknown answer"
        with self._lock:
            inc = self.incidents.get(incident_id)
            if inc is None:
                return "unknown incident"
            previous, inc.status, inc.replied_t = inc.status, answer, self.clock()
        if answer == HELP and previous != HELP:
            self.publish(self._monitor_message(inc))
        elif answer == OK and previous in (HELP, NO_REPLY):
            # The monitor was already alerted; tell them it is resolved.
            self.publish(self._monitor_ok_message(inc))
        return "Help is on the way" if answer == HELP else "Glad you're OK"

    # --- dashboard server routes ----------------------------------------------------
    def routes(self) -> dict:
        """path -> handler(query: dict[str, str]) -> (status, content_type, body)."""
        return {
            "/respond": self._route_respond,
            "/incidents.json": self._route_incidents,
            "/person": self._route_person,
            "/message": self._route_message,
            "/voice": self._route_voice,
            "/messages.json": self._route_messages,
        }

    def upload_routes(self) -> dict:
        """POST routes that need the request body: handler(query, body, headers),
        for Streamer.add_post_route."""
        return {"/talk": self._route_talk}

    # --- person -> monitor communication --------------------------------------------
    def send_message(self, key: str, incident_id: str = "") -> str:
        if key not in QUICK_MESSAGES:
            return "unknown message"
        text, needs_help = QUICK_MESSAGES[key]
        self._mark(incident_id, HELP if needs_help else OK)
        self._log("text", text)
        self.publish({
            "topic": self.cfg.ntfy_monitor_topic,
            "title": f"Message - {self.cfg.room_name}",
            "message": f'The person says: "{text}" ({_hhmm(self.clock())})',
            "priority": 5 if needs_help else 3,
            "tags": ["speech_balloon"],
            "click": f"{self.base_url}/",
        })
        return "Sent"

    def save_voice(self, audio: bytes, ext: str, incident_id: str = "") -> str:
        if ext not in VOICE_TYPES:
            return "unsupported audio type"
        if len(audio) < 100:
            return "recording was empty"
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        name = "voice-" + datetime.fromtimestamp(self.clock()).strftime("%Y%m%d-%H%M%S-%f")[:-3]
        name = f"{name}.{ext}"
        (self.voice_dir / name).write_bytes(audio)
        self._mark(incident_id, HELP)  # talking after a fall: treat as needing attention
        self._log("voice", "Voice message", name)
        url = f"{self.base_url}/voice?f={quote(name)}"
        self.publish({
            "topic": self.cfg.ntfy_monitor_topic,
            "title": f"Voice message - {self.cfg.room_name}",
            "message": f"The person sent a voice message ({_hhmm(self.clock())}). Tap to listen.",
            "priority": 5,
            "tags": ["speaking_head"],
            "click": url,
            "actions": [{"action": "view", "label": "Listen", "url": url}],
        })
        return "Sent"

    def _mark(self, incident_id: str, status: str) -> None:
        """Update an open incident from a message, without a second "help" notification."""
        with self._lock:
            inc = self.incidents.get(incident_id)
            if inc is not None and inc.status != status:
                inc.status, inc.replied_t = status, self.clock()

    def _log(self, kind: str, text: str, file: str | None = None) -> None:
        with self._lock:
            self.messages.append({"t": self.clock(), "kind": kind, "text": text, "file": file})

    def _route_message(self, q):
        msg = self.send_message(q.get("key", ""), q.get("id", ""))
        return (200 if msg == "Sent" else 400), "text/plain; charset=utf-8", msg.encode()

    def _route_talk(self, q, body, headers=None):
        msg = self.save_voice(body, q.get("ext", "webm"), q.get("id", ""))
        return (200 if msg == "Sent" else 400), "text/plain; charset=utf-8", msg.encode()

    def _route_voice(self, q):
        name = Path(q.get("f", "")).name  # no directories: only files in voice_dir
        f = self.voice_dir / name
        ext = f.suffix.lstrip(".")
        if not name or ext not in VOICE_TYPES or not f.is_file():
            return 404, "text/plain", b"not found"
        return 200, VOICE_TYPES[ext], f.read_bytes()

    def _route_messages(self, q):
        with self._lock:
            items = list(self.messages)
        return 200, "application/json", json.dumps({"now": self.clock(), "messages": items}).encode()

    def _route_respond(self, q):
        msg = self.respond(q.get("id", ""), q.get("answer", ""))
        return 200, "text/plain; charset=utf-8", msg.encode()

    def _route_incidents(self, q):
        with self._lock:
            items = [asdict(i) for i in sorted(self.incidents.values(), key=lambda i: i.t)]
        body = {"now": self.clock(), "reply_timeout_s": self.cfg.reply_timeout_s,
                "incidents": items}
        return 200, "application/json", json.dumps(body).encode()

    def _route_person(self, q):
        html = PERSON_PAGE.read_text(encoding="utf-8")
        html = html.replace("{{CALL_NUMBER}}", self.cfg.call_number)
        return 200, "text/html; charset=utf-8", html.encode()

    # --- messages -------------------------------------------------------------------
    def _respond_url(self, inc: Incident, answer: str) -> str:
        return f"{self.base_url}/respond?id={quote(inc.id)}&answer={answer}"

    def _person_message(self, inc: Incident) -> dict:
        actions = [
            {"action": "http", "label": "I'm OK", "url": self._respond_url(inc, OK),
             "method": "POST", "clear": True},
            {"action": "http", "label": "I need help", "url": self._respond_url(inc, HELP),
             "method": "POST", "clear": True},
        ]
        if self.cfg.call_number:
            actions.append({"action": "view", "label": "Call",
                            "url": f"tel:{self.cfg.call_number}"})
        return {
            "topic": self.cfg.ntfy_person_topic,
            "title": "Did you fall?",
            "message": (f"A fall was detected at {_hhmm(inc.t)}. Tap I'm OK if you are fine. "
                        f"Otherwise your contact is alerted in {self.cfg.reply_timeout_s:.0f} s."),
            "priority": 5,
            "tags": ["rotating_light"],
            "click": f"{self.base_url}/person?id={quote(inc.id)}",
            "actions": actions,
        }

    def _monitor_message(self, inc: Incident) -> dict:
        why = ("asked for help" if inc.status == HELP
               else f"did not reply within {self.cfg.reply_timeout_s:.0f} s")
        msg = {
            "topic": self.cfg.ntfy_monitor_topic,
            "title": f"FALL - {self.cfg.room_name}",
            "message": f"Fall detected at {_hhmm(inc.t)}. The person {why}.",
            "priority": 5,
            "tags": ["warning"],
            "click": self._dashboard_url(inc),
            "actions": [{"action": "view", "label": "Open dashboard",
                         "url": self._dashboard_url(inc)}],
        }
        if self.cfg.person_number:
            msg["actions"].append({"action": "view", "label": "Call person",
                                   "url": f"tel:{self.cfg.person_number}"})
        if inc.snapshot and self.cfg.ntfy_attach_snapshot:
            msg["attach"] = f"{self.base_url}/events/{quote(inc.snapshot)}"
        return msg

    def _monitor_ok_message(self, inc: Incident) -> dict:
        return {
            "topic": self.cfg.ntfy_monitor_topic,
            "title": f"Resolved - {self.cfg.room_name}",
            "message": f"The person says they are OK (fall at {_hhmm(inc.t)}).",
            "priority": 3,
            "tags": ["white_check_mark"],
            "click": self._dashboard_url(inc),
        }

    def _dashboard_url(self, inc: Incident) -> str:
        return f"{self.base_url}/?event={quote(inc.snapshot)}" if inc.snapshot else f"{self.base_url}/"


def _hhmm(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%H:%M:%S")


def open_alerts(cfg: Config, port: int) -> tuple[AlertManager, NtfyPublisher] | None:
    """Build the alert manager, or None when no ntfy topics are configured."""
    if not (cfg.ntfy_person_topic and cfg.ntfy_monitor_topic):
        print("[alerts] off: set ntfy_person_topic and ntfy_monitor_topic in params.local.toml")
        return None
    base = cfg.public_url or lan_url(port)
    pub = NtfyPublisher(cfg.ntfy_server)
    print(f"[alerts] ntfy {cfg.ntfy_server} person='{cfg.ntfy_person_topic}' "
          f"monitor='{cfg.ntfy_monitor_topic}', replies via {base}")
    return AlertManager(cfg, base, pub.publish), pub


def demo() -> int:
    """python alerts.py --demo [port]: serve the reply routes and send one fake fall,
    to rehearse the phone flow without anyone falling. Ctrl+C to stop."""
    from stream import Streamer
    cfg = Config.load()
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    built = open_alerts(cfg, port)
    if built is None:
        return 1
    manager, publisher = built
    streamer = Streamer(port=port, events_dir=cfg.events_dir)
    for path, handler in manager.routes().items():
        streamer.add_route(path, handler)
    for path, handler in manager.upload_routes().items():
        streamer.add_post_route(path, handler)
    snaps = sorted(Path(cfg.events_dir).glob("*.jpg"), key=lambda p: p.stat().st_mtime)
    manager.send(FallEvent(timestamp=time.time(), kind="fall", peak_hip_vel=3.0,
                           torso_angle=95.0, snapshot_path=str(snaps[-1]) if snaps else None,
                           stream_t=0.0))
    print(f"Fake fall sent. Person page: {manager.base_url}/person  (Ctrl+C to stop)")
    last = None
    try:
        while True:
            manager.tick()
            status = [i.status for i in manager.incidents.values()]
            if status != last:
                print(f"[demo] incident status: {status[-1]}")
                last = status
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.close()
        streamer.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        sys.exit(demo())
    print(__doc__)
