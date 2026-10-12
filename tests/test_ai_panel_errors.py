import numpy as np

from src.ai_panel import build_command, check_inputs, write_wav


def _files(tmp_path):
    src = tmp_path / "source.wav"
    ref = tmp_path / "ref.wav"
    py = tmp_path / "python.exe"
    script = tmp_path / "ai_convert.py"
    for f in (src, ref, py, script):
        f.write_text("x", encoding="utf-8")
    return src, ref, py, script


def test_all_present_gives_no_error(tmp_path):
    src, ref, py, script = _files(tmp_path)
    assert check_inputs(str(src), str(ref), py, script) is None


def test_no_source_chosen(tmp_path):
    _, ref, py, script = _files(tmp_path)
    err = check_inputs(None, str(ref), py, script)
    assert err is not None and "source" in err.lower()


def test_no_reference_chosen(tmp_path):
    src, _, py, script = _files(tmp_path)
    err = check_inputs(str(src), None, py, script)
    assert err is not None and "reference" in err.lower()


def test_source_checked_before_reference(tmp_path):
    _, _, py, script = _files(tmp_path)
    err = check_inputs(None, None, py, script)
    assert "source" in err.lower()


def test_source_file_missing(tmp_path):
    _, ref, py, script = _files(tmp_path)
    err = check_inputs(str(tmp_path / "gone.wav"), str(ref), py, script)
    assert err is not None and "source" in err.lower()


def test_reference_file_missing(tmp_path):
    src, _, py, script = _files(tmp_path)
    err = check_inputs(str(src), str(tmp_path / "gone.wav"), py, script)
    assert err is not None and "reference" in err.lower()


def test_ai_python_missing(tmp_path):
    src, ref, _, script = _files(tmp_path)
    err = check_inputs(str(src), str(ref), tmp_path / "no_python.exe", script)
    assert err is not None and "python" in err.lower()


def test_ai_script_missing(tmp_path):
    src, ref, py, _ = _files(tmp_path)
    err = check_inputs(str(src), str(ref), py, tmp_path / "no_script.py")
    assert err is not None and "ai_convert" in err


def test_build_command_arguments():
    program, args = build_command("py.exe", "s.py", "a.wav", "b.wav", "out.wav")
    assert program == "py.exe"
    assert args == ["s.py", "--source", "a.wav", "--ref", "b.wav", "--output", "out.wav"]


def test_write_wav_accepts_empty_recording(tmp_path):
    out = tmp_path / "empty.wav"
    write_wav(out, np.array([], dtype=np.float32), 16000)
    assert out.is_file()