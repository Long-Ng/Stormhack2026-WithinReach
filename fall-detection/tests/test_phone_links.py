import cv2
import numpy as np

from phone_links import LocalName, connect_page, qr_matrix, qr_png, qr_text

URL = "http://192.168.1.23:5000/monitor"


def decode(img: np.ndarray) -> str:
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    return text


def test_png_decodes_back_to_the_url():
    img = cv2.imdecode(np.frombuffer(qr_png(URL), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert decode(img) == URL


def test_quiet_zone_is_four_modules():
    dark = qr_matrix(URL)
    assert not dark[:4].any() and not dark[-4:].any()
    assert not dark[:, :4].any() and not dark[:, -4:].any()
    assert dark[4:-4, 4:-4].any()


def test_terminal_version_is_the_same_code():
    lines = qr_text(URL)
    dark = qr_matrix(URL)
    assert len(lines) == (len(dark) + 1) // 2 and len(lines[0]) == dark.shape[1]
    # Rebuild the picture from the half blocks and scan it.
    # On a dark terminal a block is drawn light: "▄" = dark top, light bottom.
    top = {"█": 0, "▄": 1, "▀": 0, " ": 1}
    bottom = {"█": 0, "▄": 0, "▀": 1, " ": 1}
    rows = []
    for line in lines:
        rows.append([top[c] for c in line])
        rows.append([bottom[c] for c in line])
    img = np.where(np.array(rows, bool), 0, 255).astype(np.uint8)
    img = cv2.resize(img, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    assert decode(img) == URL


def test_connect_page_lists_both_pages_and_the_local_name():
    page = connect_page("http://192.168.1.23:5000", "http://within-reach.local:5000").decode()
    for path in ("/monitor", "/granny"):
        assert f"/qr.png?path={path}" in page and f"http://192.168.1.23:5000{path}" in page
        assert f"http://within-reach.local:5000{path}" in page
    assert "within-reach.local" not in connect_page("http://192.168.1.23:5000", None).decode()


def test_local_name_url_drops_port_80():
    assert LocalName("within-reach", "192.168.1.23", 80).url == "http://within-reach.local"
    assert LocalName("within-reach", "192.168.1.23", 5000).url == "http://within-reach.local:5000"
