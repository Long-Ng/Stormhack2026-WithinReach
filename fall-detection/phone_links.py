"""Easy ways for phones to open the pages: QR codes and a short local name.

    terminal at startup  -> QR codes for /monitor and /granny (scan with the phone camera)
    http://<pc>/connect  -> the same QR codes on a page, e.g. shown on the laptop screen
    within-reach.local   -> the PC's name on the Wi-Fi (mDNS), if zeroconf is installed

Everything stays on the local network: nothing is published outside the Wi-Fi.
mDNS (.local names) works on iPhones, Macs and Windows; Android support is patchy,
which is why the QR codes use the plain IP address.
"""

from __future__ import annotations

import html
import socket
import sys
import threading

import cv2
import numpy as np

PAGES = [("/monitor", "Family member's phone", "Live camera, alerts, what to do"),
         ("/granny", "Person being monitored", "\"Did you fall?\" with I'm OK / I need help")]


def qr_matrix(text: str) -> np.ndarray:
    """QR code as a bool array (True = dark), with a 4-module quiet zone."""
    m = cv2.QRCodeEncoder.create().encode(text)
    dark = m < 128
    return np.pad(dark, 4 - _border(dark), constant_values=False) if _border(dark) < 4 else dark


def _border(dark: np.ndarray) -> int:
    """Width of the empty margin OpenCV already put around the code."""
    rows = np.where(dark.any(axis=1))[0]
    return int(rows[0]) if len(rows) else 0


def qr_png(text: str, scale: int = 8) -> bytes:
    img = np.where(qr_matrix(text), 0, 255).astype(np.uint8)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return cv2.imencode(".png", img)[1].tobytes()


def qr_text(text: str) -> list[str]:
    """QR code as terminal lines, two modules per character (half blocks)."""
    dark = qr_matrix(text)
    if len(dark) % 2:
        dark = np.vstack([dark, np.zeros((1, dark.shape[1]), bool)])
    # Light background with dark modules, as phones expect.
    chars = {(False, False): "█", (True, False): "▄",
             (False, True): "▀", (True, True): " "}
    return ["".join(chars[(bool(top), bool(bot))] for top, bot in zip(dark[r], dark[r + 1]))
            for r in range(0, len(dark), 2)]


def print_qr_codes(base_url: str, local_url: str | None) -> None:
    """Both QR codes side by side, with the URLs under them."""
    blocks = [qr_text(base_url + path) for path, _, _ in PAGES]
    width = max(len(line) for line in blocks[0])
    lines = [a.ljust(width) + "    " + b for a, b in zip(*blocks)]
    lines.append("MONITOR (family)".ljust(width) + "    " + "GRANNY (the person)")
    for path, _, _ in PAGES:
        lines.append(f"  {base_url}{path}" + (f"   or   {local_url}{path}" if local_url else ""))
    try:
        print("[phones] scan with a phone camera:\n" + "\n".join(lines), flush=True)
    except UnicodeEncodeError:  # console without UTF-8: the URLs are enough
        print("[phones] open: " + ", ".join(base_url + p for p, _, _ in PAGES), flush=True)


def connect_page(base_url: str, local_url: str | None) -> bytes:
    cards = []
    for path, who, what in PAGES:
        alt = f'<div class="alt">or type <b>{html.escape(local_url + path)}</b></div>' if local_url else ""
        cards.append(f'''<div class="card"><h2>{html.escape(who)}</h2><p>{html.escape(what)}</p>
<img src="/qr.png?path={path}" alt="QR code for {path}">
<div class="url">{html.escape(base_url + path)}</div>{alt}</div>''')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Connect a phone</title>
<style>
:root{{--bg:#f5f5f1;--card:#fff;--ink:#14231f;--muted:#5d6b66}}
@media (prefers-color-scheme: dark){{:root{{--bg:#111816;--card:#1b2421;--ink:#eef2f0;--muted:#9fb1ab}}}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--ink);font-family:system-ui,"Segoe UI",sans-serif}}
h1{{text-align:center;margin:0 0 6px}}.sub{{text-align:center;color:var(--muted);margin:0 0 22px}}
.grid{{display:flex;flex-wrap:wrap;gap:18px;justify-content:center}}
.card{{background:var(--card);border-radius:20px;padding:20px;width:300px;max-width:100%;text-align:center}}
.card h2{{margin:0 0 4px;font-size:20px}}.card p{{margin:0 0 12px;color:var(--muted)}}
.card img{{width:240px;max-width:100%;image-rendering:pixelated;background:#fff;border-radius:8px}}
.url{{margin-top:10px;font-family:ui-monospace,Consolas,monospace;font-size:14px;word-break:break-all}}
.alt{{margin-top:6px;color:var(--muted);font-size:13px}}
</style></head><body><h1>Connect a phone</h1>
<p class="sub">Scan with the phone's camera. The phone must be on the same Wi-Fi.</p>
<div class="grid">{"".join(cards)}</div></body></html>'''.encode()


class LocalName:
    """Announce this PC as <name>.local on the Wi-Fi (mDNS), so phones can type a name
    instead of an IP. Needs the zeroconf package; without it this does nothing."""

    def __init__(self, name: str, ip: str, port: int):
        self.name, self.ip, self.port = name, ip, port
        self._zc = self._info = None

    @property
    def url(self) -> str:
        return f"http://{self.name}.local" + ("" if self.port == 80 else f":{self.port}")

    def start(self) -> bool:
        try:
            from zeroconf import IPVersion, ServiceInfo, Zeroconf
        except ImportError:
            print("[phones] for a short name like within-reach.local: pip install zeroconf",
                  file=sys.stderr)
            return False
        self._info = ServiceInfo(
            "_http._tcp.local.", f"Within Reach ({self.name})._http._tcp.local.",
            addresses=[socket.inet_aton(self.ip)], port=self.port,
            server=f"{self.name}.local.", properties={"path": "/monitor"})
        self._zc = Zeroconf(ip_version=IPVersion.V4Only, interfaces=[self.ip])
        # Registering probes the network for name clashes (a second or two): not on startup.
        threading.Thread(target=self._register, daemon=True).start()
        return True

    def _register(self) -> None:
        try:
            self._zc.register_service(self._info)
        except Exception as e:
            print(f"[phones] could not announce {self.name}.local: {e!r}", file=sys.stderr)

    def close(self) -> None:
        if self._zc is not None:
            try:
                self._zc.unregister_all_services()
                self._zc.close()
            except Exception:
                pass
