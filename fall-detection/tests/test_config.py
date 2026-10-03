import pytest

from config import DEFAULT_PARAMS, Config


def write(tmp_path, text):
    p = tmp_path / "params.toml"
    p.write_text(text)
    return p


def test_shipped_params_file_loads():
    assert DEFAULT_PARAMS.exists()
    Config.load()


def test_overrides_and_keeps_defaults(tmp_path):
    cfg = Config.load(write(tmp_path, "[detector]\nconfirm_s = 2\nstill_motion = 0.5\n"))
    assert cfg.confirm_s == 2.0 and isinstance(cfg.confirm_s, float)
    assert cfg.still_motion == 0.5
    assert cfg.fall_vel == Config().fall_vel


def test_unknown_key_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="confirm_secs"):
        Config.load(write(tmp_path, "confirm_secs = 2.0\n"))


def test_wrong_type_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="camera_index"):
        Config.load(write(tmp_path, "camera_index = 1.5\n"))


def test_missing_explicit_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.load(tmp_path / "nope.toml")


def test_relative_model_path_resolves_next_to_params(tmp_path):
    cfg = Config.load(write(tmp_path, 'model_path = "models/x.task"\n'))
    assert cfg.model_path == str(tmp_path / "models" / "x.task")


def test_bool_params(tmp_path):
    assert Config.load(write(tmp_path, "prefer_droidcam = false\n")).prefer_droidcam is False
    with pytest.raises(ValueError, match="prefer_droidcam"):
        Config.load(write(tmp_path, "prefer_droidcam = 0\n"))


def test_local_params_override_main_file(tmp_path):
    main = write(tmp_path, "[alerts]\nreply_timeout_s = 30.0\n")
    (tmp_path / "params.local.toml").write_text('ntfy_person_topic = "secret"\nreply_timeout_s = 10\n')
    cfg = Config.load(main)
    assert cfg.ntfy_person_topic == "secret" and cfg.reply_timeout_s == 10.0


def test_local_params_unknown_key_is_an_error(tmp_path):
    main = write(tmp_path, "")
    (tmp_path / "params.local.toml").write_text('ntfy_topic = "x"\n')
    with pytest.raises(ValueError, match="ntfy_topic"):
        Config.load(main)
