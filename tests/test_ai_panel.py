from src.ai_panel import build_command, check_inputs


def test_build_command_order():
    prog, args = build_command("py.exe", "s.py", "a.wav", "b.wav", "o.wav")
    assert prog == "py.exe"
    assert args == ["s.py", "--source", "a.wav", "--ref", "b.wav", "--output", "o.wav"]


def test_check_inputs_needs_source_and_ref():
    assert check_inputs(None, "r.wav", "p", "s") is not None
    assert check_inputs("a.wav", None, "p", "s") is not None


def test_check_inputs_missing_files(tmp_path):
    msg = check_inputs(str(tmp_path / "x.wav"), str(tmp_path / "y.wav"), "p", "s")
    assert "nahi mili" in msg


def test_check_inputs_ok(tmp_path):
    for n in ("a.wav", "b.wav", "py.exe", "s.py"):
        (tmp_path / n).write_text("x")
    err = check_inputs(
        str(tmp_path / "a.wav"), str(tmp_path / "b.wav"),
        tmp_path / "py.exe", tmp_path / "s.py",
    )
    assert err is None