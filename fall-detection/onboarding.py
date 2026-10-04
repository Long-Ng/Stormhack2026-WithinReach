"""onboarding.py - stores the onboarding data (name, address, emergency voice message, family contacts).

Put next to stream.py. Streamer attaches it automatically (see stream.py); nothing else to wire.
Data lives in fall-detection/data/ (profile.json + the recording). Keep that folder out of git.
There is no login: anyone on the same network can read these endpoints, so use a trusted Wi-Fi.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

EXT = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/m4a": ".m4a",
       "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/3gpp": ".3gp", "audio/amr": ".amr",
       "audio/aac": ".aac", "video/webm": ".webm", "video/mp4": ".m4a"}
MAX_RECORDING = 20 * 1024 * 1024
JSON = "application/json; charset=utf-8"


def _json(obj, code=200):
    return code, JSON, json.dumps(obj, ensure_ascii=False).encode("utf-8")


class Onboarding:
    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "profile.json"
        self.lock = threading.Lock()
        self.data = {"name": "", "address": "", "onboarded": False, "recording": None, "contacts": []}
        if self.file.is_file():
            try:
                self.data.update(json.loads(self.file.read_text(encoding="utf-8")))
            except ValueError:
                pass

    # ---- helpers ----
    def _save(self):
        self.file.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def done(self) -> bool:
        return bool(self.data.get("onboarded"))

    def config_overrides(self) -> dict:
        """Merged into /config.json so the phone page shows the name/address entered during onboarding."""
        out = {}
        if self.data.get("name"):
            out["name"] = self.data["name"]
        if self.data.get("address"):
            out["address"] = self.data["address"]
        out["contacts"] = [{"name": c["name"], "relationship": c["relationship"], "phone": c["phone"]}
                           for c in self.data.get("contacts", [])]
        out["has_recording"] = bool(self.data.get("recording"))
        return out

    def attach(self, streamer):
        streamer.onboarding = self
        streamer.add_route("/api/profile", self.get_profile)
        streamer.add_route("/api/recording", self.get_recording)
        streamer.add_post_route("/api/profile", self.post_profile)
        streamer.add_post_route("/api/recording", self.post_recording)
        streamer.add_post_route("/api/contacts", self.post_contact)
        streamer.add_post_route("/api/contacts/delete", self.delete_contact)
        streamer.add_post_route("/api/contacts/invited", self.mark_invited)
        streamer.add_post_route("/api/finish", self.finish)
        streamer.add_post_route("/api/reset", self.reset)

    # ---- GET ----
    def get_profile(self, query):
        d = self.data
        rec = d.get("recording")
        return _json({"name": d.get("name", ""), "address": d.get("address", ""), "onboarded": bool(d.get("onboarded")),
                      "has_recording": bool(rec), "recording_seconds": (rec or {}).get("seconds", 0),
                      "contacts": d.get("contacts", [])})

    def get_recording(self, query):
        rec = self.data.get("recording")
        if not rec or not (self.dir / rec["file"]).is_file():
            return 404, "text/plain", b"no recording"
        return 200, rec.get("mime", "audio/webm"), (self.dir / rec["file"]).read_bytes()

    # ---- POST (query, body, headers) ----
    def post_profile(self, query, body, headers):
        try:
            p = json.loads(body.decode("utf-8") or "{}")
        except ValueError:
            return _json({"error": "bad json"}, 400)
        with self.lock:
            if "name" in p:
                self.data["name"] = str(p["name"]).strip()[:80]
            if "address" in p:
                self.data["address"] = str(p["address"]).strip()[:200]
            self._save()
        return _json({"ok": True})

    def post_recording(self, query, body, headers):
        if not body:
            return _json({"error": "empty"}, 400)
        if len(body) > MAX_RECORDING:
            return _json({"error": "too large"}, 413)
        mime = (headers.get("Content-Type") or "audio/webm").split(";")[0].strip().lower()
        if not (mime.startswith("audio/") or mime.startswith("video/")):
            return _json({"error": "not audio"}, 415)
        try:
            seconds = max(0, min(int(float(query.get("seconds", "0"))), 600))
        except ValueError:
            seconds = 0
        name = "emergency_message" + EXT.get(mime, ".bin")
        with self.lock:
            old = self.data.get("recording")
            if old and old["file"] != name and (self.dir / old["file"]).is_file():
                (self.dir / old["file"]).unlink()
            (self.dir / name).write_bytes(body)
            self.data["recording"] = {"file": name, "mime": mime, "seconds": seconds, "saved": time.time()}
            self._save()
        return _json({"ok": True, "seconds": seconds})

    def post_contact(self, query, body, headers):
        try:
            p = json.loads(body.decode("utf-8") or "{}")
        except ValueError:
            return _json({"error": "bad json"}, 400)
        name = str(p.get("name", "")).strip()[:80]
        rel = str(p.get("relationship", "")).strip()[:40]
        phone = re.sub(r"[^\d+]", "", str(p.get("phone", "")))[:20]
        if not name or not rel or len(re.sub(r"\D", "", phone)) < 6:
            return _json({"error": "missing fields"}, 400)
        contact = {"id": uuid.uuid4().hex[:8], "name": name, "relationship": rel, "phone": phone, "invited": False}
        with self.lock:
            self.data["contacts"].append(contact)
            self._save()
        return _json(contact)

    def delete_contact(self, query, body, headers):
        cid = query.get("id", "")
        with self.lock:
            self.data["contacts"] = [c for c in self.data["contacts"] if c["id"] != cid]
            self._save()
        return _json({"ok": True})

    def mark_invited(self, query, body, headers):
        cid = query.get("id", "")
        with self.lock:
            for c in self.data["contacts"]:
                if c["id"] == cid:
                    c["invited"] = True
            self._save()
        return _json({"ok": True})

    def finish(self, query, body, headers):
        with self.lock:
            if not self.data.get("recording") or not self.data.get("contacts"):
                return _json({"error": "need a recording and at least one contact"}, 400)
            self.data["onboarded"] = True
            self._save()
        return _json({"ok": True})

    def reset(self, query, body, headers):
        with self.lock:
            self.data["onboarded"] = False
            self._save()
        return _json({"ok": True})