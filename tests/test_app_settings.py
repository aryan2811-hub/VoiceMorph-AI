import json

from src.app_settings import load_settings, sanitize, save_settings


def test_missing_file_gives_defaults(tmp_path):
    assert load_settings(tmp_path / "nope.json") == sanitize(None)


def test_broken_json_gives_defaults(tmp_path):
    bad = tmp_path / "settings.json"
    bad.write_text("{this is not json", encoding="utf-8")
    assert load_settings(bad) == sanitize(None)


def test_non_dict_json_still_returns_dict(tmp_path):
    odd = tmp_path / "settings.json"
    odd.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert isinstance(load_settings(odd), dict)


def test_save_then_load_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    data = sanitize(None)
    assert save_settings(path, data) is True
    assert load_settings(path) == data


def test_save_leaves_no_tmp_file(tmp_path):
    path = tmp_path / "settings.json"
    assert save_settings(path, sanitize(None)) is True
    assert path.is_file()
    assert not (tmp_path / "settings.json.tmp").exists()


def test_save_creates_missing_folders(tmp_path):
    path = tmp_path / "a" / "b" / "settings.json"
    assert save_settings(path, sanitize(None)) is True
    assert path.is_file()


def test_save_failure_returns_false(tmp_path):
    blocker = tmp_path / "afile"
    blocker.write_text("x", encoding="utf-8")
    # parent "folder" is really a file, so saving must fail without crashing
    assert save_settings(blocker / "settings.json", sanitize(None)) is False