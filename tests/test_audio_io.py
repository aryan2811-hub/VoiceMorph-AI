import numpy as np
import pytest

from src import audio_io
from src.audio_io import AudioIOError

SR = 44_100


def tone(seconds=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_wav_round_trip(tmp_path):
    x = tone()
    path = audio_io.save_wav(tmp_path / "out" / "a.wav", x, SR)
    y, rate = audio_io.load_audio(path)
    assert rate == SR
    assert len(y) == len(x)
    assert np.max(np.abs(x - y)) < 1e-3  # 16-bit rounding only


def test_save_rejects_non_wav(tmp_path):
    with pytest.raises(AudioIOError):
        audio_io.save_wav(tmp_path / "a.mp3", tone(), SR)


def test_save_rejects_empty_audio(tmp_path):
    with pytest.raises(ValueError):
        audio_io.save_wav(tmp_path / "a.wav", np.array([]), SR)


def test_save_clips_instead_of_wrapping(tmp_path):
    loud = np.full(1000, 3.0, dtype=np.float32)
    path = audio_io.save_wav(tmp_path / "loud.wav", loud, SR)
    y, _ = audio_io.load_audio(path)
    assert np.all(y > 0.9)


def test_load_missing_file(tmp_path):
    with pytest.raises(AudioIOError):
        audio_io.load_audio(tmp_path / "nope.wav")


def test_load_corrupt_file(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(AudioIOError):
        audio_io.load_audio(bad)


def test_record_rejects_bad_duration():
    with pytest.raises(AudioIOError):
        audio_io.record(0)
    with pytest.raises(AudioIOError):
        audio_io.record(float("nan"))


def test_stereo_file_loads_as_mono(tmp_path):
    import soundfile as sf
    stereo = np.stack([tone(), tone()], axis=1)
    sf.write(str(tmp_path / "s.wav"), stereo, SR)
    y, _ = audio_io.load_audio(tmp_path / "s.wav")
    assert y.ndim == 1
