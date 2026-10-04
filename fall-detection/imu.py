"""Wearable accelerometer via Phyphox remote access, and impact detection.

Phone: Phyphox > "Acceleration (without g)" (or "with g") > menu > Allow remote access.

    python imu.py http://172.16.164.134:8080    # live readout + impacts, for drop tests

main.py feeds impacts and phone stillness to FallDetector (see `Wearable`). Times
are PC `time.perf_counter()` seconds, estimated from arrival time, so no clock
sync with the phone is needed.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass

from config import Config

G = 9.81


@dataclass
class Impact:
    t: float        # PC perf_counter seconds
    peak: float     # m/s^2 above rest


class ImpactDetector:
    """Fires when the acceleration departure from rest exceeds cfg.imu_impact_ms2,
    at most once per cfg.imu_refractory_s."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._last: float | None = None

    def update(self, t: float, accel: float) -> Impact | None:
        if accel < self.cfg.imu_impact_ms2:
            return None
        if self._last is not None and t - self._last < self.cfg.imu_refractory_s:
            return None
        self._last = t
        return Impact(t, accel)


class Wearable:
    """Turns a reader's samples into impacts and "still for N seconds"."""

    def __init__(self, reader: PhyphoxReader, cfg: Config):
        self.reader = reader
        self.cfg = cfg
        self.impacts = ImpactDetector(cfg)
        self.accel = 0.0                      # latest value, m/s^2 from rest
        self.last_impact: Impact | None = None
        self.impact_log: deque[Impact] = deque(maxlen=20)  # recent impacts, for the graph
        self._still_since: float | None = None
        self._last_t = time.perf_counter()

    @property
    def connected(self) -> bool:
        return self.reader.connected

    def poll(self) -> list[Impact]:
        """Process samples that arrived since the last call. Returns new impacts."""
        new = [(t, a) for t, a in list(self.reader.samples) if t > self._last_t]
        hits = []
        for t, a in new:
            self.accel = a
            if a < self.cfg.imu_still_ms2:
                if self._still_since is None:
                    self._still_since = t
            else:
                self._still_since = None
            hit = self.impacts.update(t, a)
            if hit is not None:
                hits.append(hit)
                self.last_impact = hit
                self.impact_log.append(hit)
        if new:
            self._last_t = new[-1][0]
        return hits

    def still_for(self, now: float) -> float:
        return 0.0 if self._still_since is None else max(now - self._still_since, 0.0)

    def graph(self, seconds: float = 10.0, now: float | None = None) -> dict:
        """The last `seconds` of acceleration for the dashboard graph, as ages in seconds."""
        now = time.perf_counter() if now is None else now
        if not self.connected:
            return {"available": False, "reason": "Phone sensor (Phyphox) offline"}
        samples = [[round(now - t, 3), round(a, 2)] for t, a in list(self.reader.samples) if now - t <= seconds]
        impacts = [[round(now - h.t, 3), round(h.peak, 2)] for h in list(self.impact_log) if now - h.t <= seconds]
        return {"available": True, "seconds": seconds, "threshold": self.cfg.imu_impact_ms2,
                "accel": round(self.accel, 2), "samples": samples, "impacts": impacts}

    def close(self) -> None:
        self.reader.stop()


def open_wearable(cfg: Config) -> Wearable | None:
    """Connect to cfg.imu_url. A missing or unreachable phone is not fatal."""
    if not cfg.imu_url:
        return None
    try:
        reader = PhyphoxReader(cfg.imu_url, cfg).start()
    except Exception as e:
        print(f"Phone sensor not available at {cfg.imu_url} ({e!r}); camera only")
        return None
    print(f"Phone sensor: Phyphox '{reader.title}' at {cfg.imu_url}")
    return Wearable(reader, cfg)


class PhyphoxReader:
    """Polls a Phyphox acceleration experiment in a background thread.

    `samples` holds (pc_t, accel) where accel is the departure from rest in m/s^2:
    |a| for the "without g" experiment, ||a| - g| for "with g".
    """

    def __init__(self, url: str, cfg: Config, maxlen: int = 3000):
        self.url = url.rstrip("/")
        self.cfg = cfg
        self.samples: deque[tuple[float, float]] = deque(maxlen=maxlen)
        self.connected = False
        self.error: str | None = None
        self._phone_t = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        config = self._get("/config")
        names = {b["name"] for b in config["buffers"]}
        if not {"acc_time", "accX", "accY", "accZ"} <= names:
            raise ValueError(f"{config.get('title')!r} is not a Phyphox acceleration experiment")
        self.title = config.get("title", "")
        self._has_abs = "acc" in names
        self._with_g = "without g" not in self.title.lower()

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=self.cfg.imu_timeout_s) as r:
            return json.load(r)

    def start(self) -> PhyphoxReader:
        status = self._get("/get")["status"]
        if not status.get("measuring"):
            self._get("/control?cmd=start")
        # Skip whatever was recorded before we connected.
        self._phone_t = self._latest_phone_t()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _latest_phone_t(self) -> float:
        buf = self._get("/get?acc_time")["buffer"]["acc_time"]["buffer"]
        return buf[-1] if buf else 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll()
                self.connected, self.error = True, None
            except Exception as e:
                self.connected, self.error = False, repr(e)
            self._stop.wait(self.cfg.imu_poll_s)

    def _poll(self) -> None:
        since = self._phone_t
        wanted = ["acc_time"] + (["acc"] if self._has_abs else ["accX", "accY", "accZ"])
        # Partial fetch: only values with acc_time > since ("%7C" is "|").
        query = "&".join(f"{n}={since}" + ("" if n == "acc_time" else "%7Cacc_time")
                         for n in wanted)
        data = self._get("/get?" + query)
        now = time.perf_counter()
        b = data["buffer"]
        times = b["acc_time"]["buffer"]
        if not times:
            return
        if self._has_abs:
            mags = b["acc"]["buffer"]
        else:
            mags = [math.sqrt(x * x + y * y + z * z) for x, y, z in
                    zip(b["accX"]["buffer"], b["accY"]["buffer"], b["accZ"]["buffer"])]
        latest = times[-1]
        for pt, m in zip(times, mags):
            if pt is None or m is None:
                continue
            accel = abs(m - G) if self._with_g else m
            # Newest sample arrived ~now; older ones by their phone-time offset.
            self.samples.append((now - (latest - pt), accel))
        self._phone_t = latest


def main() -> int:
    cfg = Config.load()
    url = sys.argv[1] if len(sys.argv) > 1 else cfg.imu_url
    if not url:
        print("usage: python imu.py http://<phone-ip>:8080  (or set imu_url in params.toml)")
        return 1
    reader = PhyphoxReader(url, cfg).start()
    detector = ImpactDetector(cfg)
    print(f"Connected to Phyphox '{reader.title}'. Impact threshold "
          f"{cfg.imu_impact_ms2:.0f} m/s^2 ({cfg.imu_impact_ms2 / G:.1f} g). Ctrl+C to stop.")
    last_t = time.perf_counter()
    t0 = last_t
    try:
        while True:
            time.sleep(0.1)
            new = [(t, a) for t, a in list(reader.samples) if t > last_t]
            if new:
                last_t = new[-1][0]
            for t, a in new:
                hit = detector.update(t, a)
                if hit is not None:
                    print(f"\n*** IMPACT {hit.peak:5.1f} m/s^2 ({hit.peak / G:.1f} g) "
                          f"at {hit.t - t0:6.2f}s")
            if new:
                peak = max(a for _, a in new)
                bar = "#" * min(int(peak), 60)
                print(f"\r{peak:5.1f} m/s^2 |{bar:<60}|", end="", flush=True)
            elif reader.error:
                print(f"\rphone not responding: {reader.error}", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        reader.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
