"""notify.py - phone push notifications through ntfy (https://ntfy.sh).

NtfySink plugs into the sink list in main.py. On every FallEvent it sends an urgent push with
the saved snapshot attached and three action buttons: open live camera, push to talk, call.
"""
from __future__ import annotations

import base64
import socket
import sys
import threading
import urllib.request
from pathlib import Path


def _hdr(text: str) -> str:
    """HTTP headers must be latin-1 safe; encode anything else as RFC 2047 (chunked, whole characters)."""
    try:
        text.encode("ascii")
        return text
    except UnicodeEncodeError:
        pass
    words, chunk = [], ""
    for ch in text:
        if len((chunk + ch).encode("utf-8")) > 40:
            words.append(chunk)
            chunk = ""
        chunk += ch
    if chunk:
        words.append(chunk)
    return " ".join("=?UTF-8?B?" + base64.b64encode(w.encode("utf-8")).decode("ascii") + "?=" for w in words)


def lan_ip() -> str:
    """Best guess of this PC's address on the local network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class NtfySink:
    def __init__(self, topic: str, server: str = "https://ntfy.sh", port: int = 5000,
                 who: str = "Nguoi than", phone: str = "911"):
        self.topic = topic.strip().strip("/")
        self.server = server.rstrip("/")
        self.base = f"http://{lan_ip()}:{port}"
        self.who = who
        self.phone = phone
        print(f"[notify] ntfy topic '{self.topic}' on {self.server} | phone page {self.base}/phone", flush=True)

    def send(self, event) -> None:
        # Network call in a thread so a slow connection never stalls the camera loop.
        threading.Thread(target=self._send, args=(event,), daemon=True).start()

    def _send(self, event) -> None:
        if event.kind == "prolonged_lying":
            title = f"{self.who} nằm bất động quá lâu — hãy kiểm tra ngay"
        else:
            title = f"{self.who} không phản hồi — hãy kiểm tra ngay"
        msg = (f"Phát hiện lúc {event.time_str}. Xem camera trực tiếp để biết tình hình, "
               f"sau đó nói chuyện với họ hoặc gọi {self.phone}.")
        actions = (f"view, Mở camera trực tiếp, {self.base}/phone; "
                   f"view, Nói chuyện, {self.base}/phone#talk; "
                   f"view, Gọi {self.phone}, tel:{self.phone}")
        headers = {"Title": _hdr(title), "Priority": "urgent", "Tags": "rotating_light",
                   "Click": f"{self.base}/phone", "Actions": _hdr(actions)}
        url = f"{self.server}/{self.topic}"
        snap = Path(event.snapshot_path) if event.snapshot_path else None
        try:
            if snap and snap.is_file():
                headers.update({"Filename": snap.name, "Message": _hdr(msg)})
                req = urllib.request.Request(url, data=snap.read_bytes(), headers=headers, method="PUT")
            else:
                req = urllib.request.Request(url, data=msg.encode("utf-8"), headers=headers, method="POST")
            urllib.request.urlopen(req, timeout=15).read()
            print("[notify] push sent", flush=True)
        except Exception as e:
            print(f"[notify] push failed: {e!r}", file=sys.stderr, flush=True)
