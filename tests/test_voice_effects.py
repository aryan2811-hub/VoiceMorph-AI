"""Tests for src/voice_effects.py. They use synthetic sounds, so no microphone is needed."""
import numpy as np
import pytest

from src import voice_effects as ve

SR = 16_000


def sine(freq=440.0, seconds=1.0, amp=0.5, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_freq(audio, sr=SR):
    window = np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(audio * window))
    return np.fft.rfftfreq(len(audio), 1 / sr)[np.argmax(spectrum)]


# ---- validation ----
def test_empty_audio_rejected():
    with pytest.raises(ValueError):
        ve.validate_audio(np.array([]), SR)


def test_nan_rejected():
    with pytest.raises(ValueError):
        ve.validate_audio(np.array([0.0, np.nan, 0.1]), SR)


def test_inf_rejected():
    with pytest.raises(ValueError):
        ve.validate_audio(np.array([0.0, np.inf]), SR)


@pytest.mark.parametrize("bad", [0, -1, 100, 500_000, 44100.5, "44100", True])
def test_invalid_sample_rate_rejected(bad):
    with pytest.raises(ValueError):
        ve.validate_audio(sine(), bad)


def test_stereo_is_mixed_to_mono():
    stereo = np.stack([sine(), sine()], axis=1)
    assert ve.validate_audio(stereo, SR).ndim == 1


def test_unknown_effect_rejected():
    with pytest.raises(ValueError):
        ve.process(sine(), SR, effect="nope")


# ---- every effect: same length, finite, float32 ----
@pytest.mark.parametrize("name", list(ve.EFFECTS))
def test_effect_keeps_length_and_is_finite(name):
    out = ve.process(sine(), SR, effect=name)
    assert out.shape == (SR,)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("name", list(ve.EFFECTS))
def test_silence_stays_silent_or_finite(name):
    out = ve.process(np.zeros(SR, dtype=np.float32), SR, effect=name)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) < 1e-3


@pytest.mark.parametrize("name", list(ve.EFFECTS))
def test_very_short_clip_does_not_crash(name):
    out = ve.process(sine(seconds=0.01), SR, effect=name)
    assert len(out) == int(0.01 * SR)


# ---- pitch accuracy ----
def test_pitch_up_one_octave():
    out = ve.pitch_shift(sine(440), SR, 12)
    assert dominant_freq(out) == pytest.approx(880, abs=20)


def test_pitch_down_one_octave():
    out = ve.pitch_shift(sine(440), SR, -12)
    assert dominant_freq(out) == pytest.approx(220, abs=20)


def test_zero_shift_unchanged():
    x = sine()
    assert np.allclose(ve.pitch_shift(x, SR, 0), x)


def test_deep_is_lower_and_high_is_higher():
    x = sine(300)
    assert dominant_freq(ve.effect_deep(x, SR)) < 300 < dominant_freq(ve.effect_high(x, SR))


# ---- other effects ----
def test_robot_changes_signal():
    x = sine()
    assert not np.allclose(ve.effect_robot(x, SR), x)


def test_echo_adds_late_energy():
    x = np.zeros(SR, dtype=np.float32)
    x[:100] = 0.5
    out = ve.effect_echo(x, SR)
    assert np.max(np.abs(out[int(0.28 * SR):])) > 0.1


def test_radio_removes_low_frequency():
    out = ve.effect_radio(sine(50, amp=0.5), SR, 1.0)
    assert np.sqrt(np.mean(out ** 2)) < 0.2 * np.sqrt(np.mean(sine(50, amp=0.5) ** 2))


def test_radio_works_at_low_sample_rate():
    out = ve.process(sine(sr=8000, seconds=0.5), 8000, effect="radio")
    assert np.all(np.isfinite(out))


# ---- volume and clipping ----
def test_volume_scales():
    out = ve.process(sine(amp=0.2), SR, volume=2.0)
    assert np.max(np.abs(out)) == pytest.approx(0.4, abs=0.01)


def test_negative_volume_rejected():
    with pytest.raises(ValueError):
        ve.process(sine(), SR, volume=-1)


def test_clipping_prevented():
    out = ve.process(sine(amp=0.9), SR, volume=10.0)
    assert np.max(np.abs(out)) <= 0.99 + 1e-6


def test_intensity_is_clamped():
    out = ve.process(sine(), SR, effect="robot", intensity=50)
    assert np.all(np.isfinite(out))
