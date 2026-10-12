import numpy as np
import pytest

from src import stream_effects as se
from src import voice_effects as ve

SR = 48_000


def voice_like(seconds=1.0):
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * SR)) / SR
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t))
    return x.astype(np.float32)


def run_in_blocks(effect, x, block):
    pieces = [effect.process(x[i:i + block]) for i in range(0, len(x), block)]
    return np.concatenate(pieces)


@pytest.mark.parametrize("name", list(se.STREAM_EFFECTS))
@pytest.mark.parametrize("block", [64, 256, 512, 1000])
def test_block_size_does_not_change_result(name, block):
    x = voice_like()
    a = run_in_blocks(se.make_effect(name, SR, 1.0), x, block)
    b = run_in_blocks(se.make_effect(name, SR, 1.0), x, 2048)
    assert np.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("name", ["robot", "echo", "radio"])
def test_streaming_matches_offline_version(name):
    x = voice_like()
    live = run_in_blocks(se.make_effect(name, SR, 1.0), x, 512)
    offline = ve.EFFECTS[name](x, SR, 1.0)
    assert np.allclose(live, offline, atol=1e-3)


@pytest.mark.parametrize("name", list(se.STREAM_EFFECTS))
def test_output_is_float32_same_length_finite(name):
    x = voice_like(0.1)
    out = se.make_effect(name, SR).process(x[:512])
    assert out.dtype == np.float32 and len(out) == 512 and np.all(np.isfinite(out))


def test_unknown_effect_rejected_clearly():
    with pytest.raises(ValueError, match="not available in real time"):
        se.make_effect("nope", SR)


# ---- streaming pitch shifter ----
def dominant_freq(x):
    x = x[int(0.3 * SR):]  # skip start-up
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / SR)[np.argmax(spec)]


def stream_sine(freq, effect, block=512, seconds=1.5):
    t = np.arange(int(seconds * SR)) / SR
    x = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return run_in_blocks(effect, x, block)


@pytest.mark.parametrize("semitones,expected", [(12, 880), (-12, 220), (7, 659), (-5, 330)])
def test_streaming_pitch_accuracy(semitones, expected):
    out = stream_sine(440, se.PitchShift(SR, semitones))
    assert dominant_freq(out) == pytest.approx(expected, rel=0.035)


def test_pitch_zero_is_passthrough():
    x = voice_like(0.2)
    assert np.allclose(se.PitchShift(SR, 0.0).process(x), x)


def test_pitch_output_stays_in_range_and_continuous():
    out = stream_sine(300, se.PitchShift(SR, -6))
    assert np.max(np.abs(out)) <= 0.45
    jumps = np.abs(np.diff(out[int(0.3 * SR):]))
    assert np.max(jumps) < 0.2  # no clicks at block or window boundaries


def test_deep_is_lower_and_high_is_higher_live():
    deep = dominant_freq(stream_sine(300, se.make_effect("deep", SR)))
    high = dominant_freq(stream_sine(300, se.make_effect("high", SR)))
    assert deep < 300 < high


def test_pitch_processing_is_fast_enough():
    import time
    eff = se.PitchShift(SR, 6)
    block = voice_like(1.0)[:512]
    t0 = time.perf_counter()
    for _ in range(200):
        eff.process(block)
    per_block_ms = 1000 * (time.perf_counter() - t0) / 200
    assert per_block_ms < 0.5 * (1000 * 512 / SR)  # under half of the 10.7 ms budget


def test_echo_block_longer_than_delay_rejected():
    echo = se.Echo(SR, delay_s=0.001)
    with pytest.raises(ValueError):
        echo.process(np.zeros(512, dtype=np.float32))
