import wave

import numpy as np

from src.ai_panel import write_wav


def test_write_wav_roundtrip(tmp_path):
    rate = 16000
    t = np.arange(rate) / rate
    samples = 0.5 * np.sin(2 * np.pi * 440 * t)
    out = tmp_path / "sub" / "rec.wav"
    write_wav(out, samples, rate)
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == rate
        assert w.getnframes() == rate


def test_write_wav_clips_loud_samples(tmp_path):
    out = tmp_path / "loud.wav"
    write_wav(out, np.array([5.0, -5.0, 0.0]), 8000)
    with wave.open(str(out), "rb") as w:
        data = np.frombuffer(w.readframes(3), dtype=np.int16)
    assert data[0] == 32767
    assert data[1] == -32767
    assert data[2] == 0