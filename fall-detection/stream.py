"""stream.py - put next to main.py (fall-detection/).

Tiny web server (default http://localhost:5000):
  /                desktop dashboard (dashboard/index.html or dashboard.html)
  /monitor         monitor's phone page (dashboard/phone.html); first visit goes through /onboarding
  /phone, /person  old names: redirect to /monitor and /granny
  /onboarding      onboarding flow (dashboard/onboarding.html); /onboarding?force=1 runs it again
  /config.json     name / room / address / emergency number (+ onboarding data when present)
  /video           live MJPEG stream
  /snapshot        latest frame
  /events.json     saved fall snapshots + live alert state (same as /events_api)
  /events/<file>   one snapshot image (or clip)
  /api/...         onboarding data (see onboarding.py)
Extra routes: add_route(path, handler(query)) for GET and POST, add_post_route(path, handler(query, body, headers)).
Call update(frame) with a clean frame; the red alert box is drawn here, not in main.py.
"""
import json
import re
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import cv2

HERE = Path(__file__).resolve().parent
NAME_RE = re.compile(r"^(\d{8})-(\d{6})-(\d{3})\.jpg$")
ALERT_SECONDS = 10.0
MAX_BODY = 21 * 1024 * 1024
DASHBOARDS = [
    HERE.parent / "dashboard" / "index.html",
    HERE.parent / "dashboard" / "dashboard.html",
    HERE / "dashboard.html",
    HERE / "index.html",
]
PHONE_PAGES = [HERE.parent / "dashboard" / "phone.html", HERE / "phone.html"]
ONBOARDING_PAGES = [HERE.parent / "dashboard" / "onboarding.html", HERE / "onboarding.html"]
MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".mp4": "video/mp4", ".webm": "video/webm",
               ".avi": "video/x-msvideo"}
OLD_PATHS = {"/phone": "/monitor", "/phone.html": "/monitor", "/person": "/granny"}
MANIFEST = {"name": "Within Reach", "short_name": "Within Reach", "start_url": "/monitor", "display": "standalone",
            "background_color": "#f4f4f1", "theme_color": "#b3261e", "icons": []}

# Old dashboards without the "no-inject" marker get this small panel added.
INJECT = """
<style>
#fd-panel{position:fixed;left:16px;bottom:16px;width:340px;max-height:60vh;overflow:auto;background:#fff;
border-radius:14px;box-shadow:0 6px 24px rgba(0,0,0,.25);padding:12px;z-index:99998;font-family:sans-serif}
#fd-panel h4{margin:0 0 8px;font-size:14px;color:#0b2a5b}
#fd-panel .row{display:flex;gap:8px;margin-bottom:8px;align-items:center;font-size:12px;color:#333}
#fd-panel img{width:110px;border-radius:8px;cursor:pointer;border:2px solid #e11d48}
#fd-banner{position:fixed;top:0;left:0;right:0;background:#e11d48;color:#fff;text-align:center;padding:10px;
font:700 18px sans-serif;z-index:99999;display:none}
</style>
<div id="fd-banner">&#9888; FALL DETECTED</div>
<div id="fd-panel"><h4>&#128247; Saved fall snapshots</h4><div id="fd-list">No snapshots yet.</div></div>
<script>
(function(){
  function fmt(t){return new Date(t*1000).toLocaleString();}
  function tick(){
    fetch('/events.json').then(function(r){return r.json();}).then(function(d){
      var ev = (d.events || []).slice().reverse();
      var list = document.getElementById('fd-list');
      if(ev.length){
        list.innerHTML = ev.map(function(e){
          return '<div class="row"><a href="'+e.url+'" target="_blank"><img src="'+e.url+'"></a><div><b>'+
            (e.kind||'fall').toUpperCase()+'</b><br>'+fmt(e.t)+'</div></div>';
        }).join('');
      }
      var b = document.getElementById('fd-banner');
      if(d.alert_text){ b.textContent = '\\u26A0 ' + d.alert_text; }
      b.style.display = d.alert ? 'block' : 'none';
    }).catch(function(){});
  }
  tick(); setInterval(tick, 1500);
})();
</script>
"""


class Streamer:
    def __init__(self, port=5000, events_dir=None, width=640, quality=70, fps=15):
        self.port, self.width, self.quality, self.min_dt = port, width, quality, 1.0 / fps
        self._explicit = Path(events_dir) if events_dir else None
        self._jpg, self._last_enc = None, 0.0
        self._alert_until, self._alert_text, self._box = 0.0, "", None
        self._closed = False
        self.config = {}
        self.routes = {}       # path -> handler(query) -> (status, content_type, body); GET and POST
        self.post_routes = {}  # path -> handler(query, body, headers) -> (status, content_type, body); POST only
        self.onboarding = None
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body=b"", ctype="text/plain"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

            def _redirect(self, to):
                self.send_response(302)
                self.send_header("Location", to)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _route(self):
                """Serve a route added with add_route. True if one handled the request."""
                path, _, qs = self.path.partition("?")
                handler = outer.routes.get(path)
                if handler is None:
                    return False
                query = {k: v[-1] for k, v in parse_qs(qs).items()}
                try:
                    code, ctype, body = handler(query)
                except Exception as e:
                    code, ctype, body = 500, "text/plain", repr(e).encode()
                self._send(code, body, ctype)
                return True

            def do_POST(self):
                path, _, qs = self.path.partition("?")
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    if length > MAX_BODY:
                        self.close_connection = True
                        return self._send(413, b"too large")
                    body = self.rfile.read(length) if length else b""
                    handler = outer.post_routes.get(path)
                    if handler is not None:
                        query = {k: v[-1] for k, v in parse_qs(qs).items()}
                        try:
                            code, ctype, resp = handler(query, body, self.headers)
                        except Exception as e:
                            code, ctype, resp = 500, "text/plain", repr(e).encode()
                        return self._send(code, resp, ctype)
                    if not self._route():
                        self._send(404, b"not found")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

            def do_GET(self):
                path, _, qs = self.path.partition("?")
                try:
                    if path in OLD_PATHS:  # links in notifications already sent, bookmarks
                        return self._redirect(OLD_PATHS[path] + ("?" + qs if qs else ""))
                    if self._route():
                        return
                    if path in ("/", "/dashboard", "/dashboard.html", "/index.html"):
                        f = outer._find(DASHBOARDS, "dashboard.html", "index.html")
                        if not f:
                            return self._send(404, b"dashboard html not found in ../dashboard/ or next to stream.py")
                        html = f.read_text(encoding="utf-8")
                        if "no-inject" not in html:
                            html = html.replace("</body>", INJECT + "</body>", 1) if "</body>" in html else html + INJECT
                        return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                    if path in ("/monitor", "/monitor.html"):
                        if outer.onboarding is not None and not outer.onboarding.done() and "skip" not in parse_qs(qs):
                            return self._redirect("/onboarding")
                        f = outer._find(PHONE_PAGES, "phone.html")
                        if not f:
                            return self._send(404, b"phone.html not found in ../dashboard/ or next to stream.py")
                        return self._send(200, f.read_bytes(), "text/html; charset=utf-8")
                    if path in ("/onboarding", "/onboarding.html"):
                        f = outer._find(ONBOARDING_PAGES, "onboarding.html")
                        if not f:
                            return self._send(404, b"onboarding.html not found in ../dashboard/ or next to stream.py")
                        return self._send(200, f.read_bytes(), "text/html; charset=utf-8")
                    if path == "/manifest.json":
                        return self._send(200, json.dumps(MANIFEST).encode(), "application/manifest+json")
                    if path == "/config.json":
                        cfg = dict(outer.config)
                        if outer.onboarding is not None:
                            cfg.update(outer.onboarding.config_overrides())
                        return self._send(200, json.dumps(cfg, ensure_ascii=False).encode("utf-8"),
                                          "application/json; charset=utf-8")
                    if path in ("/events.json", "/events_api"):
                        return self._send(200, json.dumps(outer.list_events()).encode(), "application/json")
                    if path.startswith("/events/"):
                        name = Path(path[len("/events/"):]).name
                        d = outer.events_path()
                        f = d / name if d else None
                        if f and f.suffix.lower() in MEDIA_TYPES and f.is_file():
                            return self._send(200, f.read_bytes(), MEDIA_TYPES[f.suffix.lower()])
                        return self._send(404, b"not found")
                    if path == "/snapshot":
                        return self._send(200, outer._jpg, "image/jpeg") if outer._jpg else self._send(503)
                    if path == "/video":
                        self.send_response(200)
                        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        while not outer._closed:
                            jpg = outer._jpg
                            if jpg:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                            time.sleep(outer.min_dt)
                        return
                    self._send(404, b"not found")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        self.server = ThreadingHTTPServer(("0.0.0.0", port), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"[stream] dashboard -> http://localhost:{port}/   monitor page -> /monitor   onboarding -> /onboarding")
        print(f"[stream] events folder -> {self.events_path() or '(not created yet)'}")
        try:  # onboarding is optional: the server still runs without onboarding.py
            from onboarding import Onboarding
            Onboarding(HERE / "data").attach(self)
            print(f"[stream] onboarding data -> {HERE / 'data'}")
        except Exception as e:
            print(f"[stream] onboarding disabled: {e!r}")

    def add_route(self, path, handler):
        """Serve path (GET and POST) with handler(query) -> (status, content_type, body)."""
        self.routes[path] = handler

    def add_post_route(self, path, handler):
        """Serve path (POST) with handler(query, body, headers) -> (status, content_type, body)."""
        self.post_routes[path] = handler

    def set_config(self, **kw):
        """Values shown on the phone page: name, room, address, phone, countdown, push, talk."""
        self.config.update(kw)

    def _find(self, candidates, *names):
        extra = [Path.cwd() / n for n in names] + [Path.cwd().parent / "dashboard" / n for n in names]
        for p in list(candidates) + extra:
            if p.is_file():
                return p
        return None

    def events_path(self):
        cands = [self._explicit] if self._explicit else []
        cands += [Path.cwd() / "events", HERE / "events", HERE.parent / "events"]
        for c in cands:
            if c and c.is_dir():
                return c
        return None

    def alert(self, text="FALL DETECTED"):
        """Show the red alert on the live video and dashboard for ALERT_SECONDS."""
        self._alert_text = text
        self._alert_until = time.time() + ALERT_SECONDS

    def set_box(self, box):
        """Pixel rectangle (x1, y1, x2, y2) drawn around the person, or None."""
        self._box = box

    def list_events(self):
        d = self.events_path()
        info = {}
        if d and (d / "events.jsonl").is_file():
            for line in (d / "events.jsonl").read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                    if e.get("snapshot_path"):
                        info[Path(e["snapshot_path"]).name] = e
                except ValueError:
                    pass
        out = []
        if d:
            for f in sorted(d.glob("*.jpg")):
                m = NAME_RE.match(f.name)
                if m:
                    t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp() + int(m.group(3)) / 1000
                else:
                    t = f.stat().st_mtime
                e = info.get(f.name, {})
                item = {"name": f.name, "t": t, "url": f"/events/{f.name}", "kind": e.get("kind", "fall"),
                        "peak_hip_vel": e.get("peak_hip_vel"), "torso_angle": e.get("torso_angle"),
                        "stream_t": e.get("stream_t")}
                if e.get("clip_path"):
                    clip = Path(e["clip_path"]).name
                    if (d / clip).is_file():
                        item["clip"] = f"/events/{clip}"
                out.append(item)
        return {"now": time.time(), "events": out[-100:],
                "alert": time.time() < self._alert_until, "alert_text": self._alert_text}

    def update(self, frame):
        now = time.time()
        if now - self._last_enc < self.min_dt:
            return
        self._last_enc = now
        if now < self._alert_until:
            frame = frame.copy()
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
            cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 255), -1)
            cv2.putText(frame, self._alert_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA)
            if self._box is not None:
                x1, y1, x2, y2 = (int(v) for v in self._box)
                cv2.rectangle(frame, (max(x1, 0), max(y1, 0)), (min(x2, w - 1), min(y2, h - 1)), (0, 0, 255), 4)
        h, w = frame.shape[:2]
        if w > self.width:
            frame = cv2.resize(frame, (self.width, int(h * self.width / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            self._jpg = buf.tobytes()

    def close(self):
        self._closed = True
        try:
            self.server.shutdown()
        except Exception:
            pass