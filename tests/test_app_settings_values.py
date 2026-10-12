import pytest

from src.app_settings import DEFAULTS, load_settings, sanitize


@pytest.mark.parametrize("raw,expected", [(500, 100), (-5, 0), (55.6, 56), (0, 0), (100, 100)])
def test_volume_is_clamped_and_rounded(raw, expected):
    assert sanitize({"volume": raw})["volume"] == expected


@pytest.mark.parametrize("raw", ["loud", None, True, False, float("nan"), float("inf"), [1]])
def test_volume_bad_types_fall_back_to_default(raw):
    assert sanitize({"volume": raw})["volume"] == DEFAULTS["volume"]


@pytest.mark.parametrize("raw,expected", [(250, 100), (-1, 0), ("x", 100)])
def test_intensity_rules(raw, expected):
    assert sanitize({"intensity": raw})["intensity"] == expected


@pytest.mark.parametrize("raw,expected", [("robot", "robot"), ("", "normal"), (123, "normal"), (None, "normal")])
def test_effect_rules(raw, expected):
    assert sanitize({"effect": raw})["effect"] == expected


@pytest.mark.parametrize("raw,expected", [(True, True), (False, False), ("yes", False), (1, False)])
def test_muted_only_accepts_real_bool(raw, expected):
    assert sanitize({"muted": raw})["muted"] == expected


@pytest.mark.parametrize("raw,expected", [("Microphone (USB)", "Microphone (USB)"), ("", None), (5, None), (None, None)])
def test_device_names(raw, expected):
    assert sanitize({"input": raw})["input"] == expected


def test_unknown_keys_are_dropped():
    assert "evil" not in sanitize({"evil": 1, "volume": 40})


def test_changing_result_does_not_change_defaults():
    s = sanitize(None)
    s["volume"] = 1
    assert DEFAULTS["volume"] == 70


def test_binary_garbage_file_gives_defaults(tmp_path):
    p = tmp_path / "settings.json"
    p.write_bytes(b"\xff\xfe\x00\x01")
    assert load_settings(p) == sanitize(None)