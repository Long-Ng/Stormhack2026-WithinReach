from config import Config
from imu import ImpactDetector


def test_impact_fires_once_per_refractory_window():
    det = ImpactDetector(Config(imu_impact_ms2=20.0, imu_refractory_s=1.0))
    readings = [(0.00, 0.5), (0.02, 25.0), (0.04, 30.0), (0.50, 22.0), (1.10, 21.0), (1.2, 0.3)]
    hits = [h for t, a in readings if (h := det.update(t, a)) is not None]
    assert [h.t for h in hits] == [0.02, 1.10]


def test_below_threshold_never_fires():
    det = ImpactDetector(Config(imu_impact_ms2=20.0))
    assert all(det.update(i / 50, 15.0) is None for i in range(200))


def test_phone_address_without_http_is_accepted(monkeypatch):
    import imu
    seen = []

    def fake_get(self, path):
        seen.append(self.url + path)
        return {"title": "Acceleration (without g)",
                "buffers": [{"name": n} for n in ("acc_time", "accX", "accY", "accZ", "acc")]}

    monkeypatch.setattr(imu.PhyphoxReader, "_get", fake_get)
    r = imu.PhyphoxReader("172.16.164.134:8080/", Config())
    assert r.url == "http://172.16.164.134:8080"
    assert seen == ["http://172.16.164.134:8080/config"]
