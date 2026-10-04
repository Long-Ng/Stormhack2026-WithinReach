import json

from onboarding import Onboarding


def test_delete_profile_erases_data_and_recording(tmp_path):
    ob = Onboarding(tmp_path)
    ob.post_profile({}, json.dumps({"name": "Nick", "address": "1 Main St"}).encode(), {})
    ob.post_recording({"seconds": "40"}, b"audio", {"Content-Type": "audio/webm"})
    assert (tmp_path / "emergency_message.webm").is_file()

    code, _, body = ob.delete_profile({}, b"", {})
    assert code == 200 and json.loads(body)["ok"]
    assert not (tmp_path / "emergency_message.webm").exists()
    assert ob.data == Onboarding._empty()
    assert Onboarding(tmp_path).data == Onboarding._empty()  # also gone from disk
