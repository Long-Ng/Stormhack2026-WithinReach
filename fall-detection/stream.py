"""Dashboard + MJPEG server for fall detection.

Streamer(port, events_dir).update(frame) is called by main.py for every frame.
Endpoints:
  /               dashboard.html
  /video          live MJPEG stream
  /snapshot       latest frame as one JPEG
  /events.json    list of saved event snapshots, newest first
  /events/<file>  a saved event snapshot
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
DASHBOARD_CANDIDATES = [
    HERE.parent / "dashboard" / "index.html",
    HERE.parent / "dashboard" / "dashboard.html",
    HERE / "dashboard.html",
    HERE / "index.html",
]


class Streamer:
    def __init__(self, port: int = 5000, events_dir="events", quality: int = 80, max_fps: float = 20.0):
        self.port = port
        self.events_dir = Path(events_dir)
        self.quality = quality
        self.min_dt = 1.0 / max_fps
        self._jpg: bytes | None = None
        self._cond = threading.Condition()
        self._last = 0.0
        self._closed = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, code, ctype, body: bytes):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self):
                self._send(204, "text/plain", b"")

            def do_GET(self):
                path = self.path.split("?")[0]
                try:
                    if path in ("/", "/index.html", "/dashboard.html"):
                        for c in DASHBOARD_CANDIDATES:
                            if c.is_file():
                                return self._send(200, "text/html; charset=utf-8", c.read_bytes())
                        return self._send(404, "text/plain", b"dashboard.html not found")
                    if path == "/video":
                        return self._video()
                    if path == "/snapshot":
                        jpg = owner._jpg
                        if jpg is None:
                            return self._send(503, "text/plain", b"no frame yet")
                        return self._send(200, "image/jpeg", jpg)
                    if path == "/events.json":
                        return self._send(200, "application/json", owner._events_json())
                    if path.startswith("/events/"):
                        f = owner.events_dir / Path(path[len("/events/"):]).name
                        if f.is_file():
                            return self._send(200, "image/jpeg", f.read_bytes())
                    self._send(404, "text/plain", b"not found")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def _video(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                last = None
                while not owner._closed:
                    with owner._cond:
                        owner._cond.wait(timeout=1.0)
                        jpg = owner._jpg
                    if jpg is None or jpg is last:
                        continue
                    last = jpg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")

        self._server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"[stream] dashboard -> http://localhost:{port}/")
        print(f"[stream] events folder -> {self.events_dir.resolve()}")

    def update(self, frame) -> None:
        now = time.perf_counter()
        if now - self._last < self.min_dt:
            return
        self._last = now
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        with self._cond:
            self._jpg = buf.tobytes()
            self._cond.notify_all()

    def _events_json(self) -> bytes:
        items = []
        if self.events_dir.is_dir():
            for f in sorted(self.events_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
                items.append({"file": f.name, "url": f"/events/{f.name}", "time": f.stat().st_mtime})
        return json.dumps(items).encode()

    def close(self) -> None:
        self._closed = True
        with self._cond:
            self._cond.notify_all()
        self._server.shutdown()
        self._server.server_close()
