"""Conventional (non-AI) voice effects.

Every effect takes mono float32 audio in the range [-1, 1] plus the sample
rate, and returns float32 audio of the SAME length. These are offline
(whole-clip) versions. Streaming versions come in the real-time phase.

Nothing in this file uses AI. These are classic signal-processing tricks.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Dict

import numpy as np
from scipy import signal

MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 192_000
PEAK_LIMIT = 0.99


# --------------------------------------------------------------------------
# Validation and small helpers
# --------------------------------------------------------------------------
def validate_audio(audio, sample_rate) -> np.ndarray:
    """Check the input and return clean mono float32 audio.

    Accepts a 1-D array, or a 2-D array shaped (samples, channels), which is
    mixed down to mono. Raises ValueError for bad input.
    """
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise ValueError(f"sample_rate must be an integer, got {sample_rate!r}")
    if not MIN_SAMPLE_RATE <= int(sample_rate) <= MAX_SAMPLE_RATE:
        raise ValueError(
            f"sample_rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz, "
            f"got {sample_rate}"
        )

    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    elif data.ndim != 1:
        raise ValueError(f"audio must be 1-D or 2-D, got {data.ndim}-D")

    if data.size == 0:
        raise ValueError("audio is empty")
    if not np.all(np.isfinite(data)):
        raise ValueError("audio contains NaN or infinite values")
    return np.ascontiguousarray(data, dtype=np.float32)


def _fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad so the result has exactly `length` samples."""
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)))


def _mix(dry: np.ndarray, wet: np.ndarray, amount: float) -> np.ndarray:
    """Blend the original (dry) and processed (wet) audio. amount is 0..1."""
    return (1.0 - amount) * dry + amount * wet


def apply_volume(audio: np.ndarray, gain: float) -> np.ndarray:
    """Multiply the audio by a gain (1.0 = unchanged, 0.0 = silence)."""
    if not np.isfinite(gain) or gain < 0:
        raise ValueError(f"gain must be a finite number >= 0, got {gain}")
    return (audio * np.float32(gain)).astype(np.float32)


def prevent_clipping(audio: np.ndarray) -> np.ndarray:
    """Scale the audio down only if it would go past full scale."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > PEAK_LIMIT:
        audio = audio * (PEAK_LIMIT / peak)
    return audio.astype(np.float32)


# --------------------------------------------------------------------------
# Pitch shifting (phase vocoder + resampling)
# --------------------------------------------------------------------------
def _phase_vocoder(stft: np.ndarray, rate: float, hop: int, n_fft: int) -> np.ndarray:
    """Time-stretch an STFT. rate < 1 makes the signal longer."""
    n_bins, n_frames = stft.shape
    steps = np.arange(0, n_frames - 1, rate)
    expected = 2.0 * np.pi * hop * np.arange(n_bins) / n_fft
    phase = np.angle(stft[:, 0])
    out = np.empty((n_bins, len(steps)), dtype=np.complex128)

    for i, step in enumerate(steps):
        idx = int(step)
        frac = step - idx
        a, b = stft[:, idx], stft[:, idx + 1]
        magnitude = (1.0 - frac) * np.abs(a) + frac * np.abs(b)
        out[:, i] = magnitude * np.exp(1j * phase)
        delta = np.angle(b) - np.angle(a) - expected
        delta -= 2.0 * np.pi * np.round(delta / (2.0 * np.pi))
        phase += expected + delta
    return out


def pitch_shift(audio, sample_rate: int, semitones: float) -> np.ndarray:
    """Shift pitch by `semitones` (+12 = one octave up) and keep the length.

    Method: stretch the sound in time without changing pitch (phase vocoder),
    then resample it back to the original length, which moves the pitch.
    """
    data = validate_audio(audio, sample_rate)
    if not np.isfinite(semitones):
        raise ValueError("semitones must be a finite number")
    if abs(semitones) < 1e-6:
        return data.copy()

    n = len(data)
    n_fft, hop = 2048, 512
    factor = 2.0 ** (semitones / 12.0)

    # Very short clips: pad with silence so the analysis window fits,
    # process, then cut back to the original length.
    min_len = n_fft * 4
    if n < min_len:
        padded = np.pad(data, (0, min_len - n))
        return pitch_shift(padded, sample_rate, semitones)[:n]

    _, _, spec = signal.stft(
        data, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros", padded=True
    )
    if spec.shape[1] < 3:  # clip too short to analyse
        return data.copy()
    stretched_spec = _phase_vocoder(spec, 1.0 / factor, hop, n_fft)
    _, stretched = signal.istft(
        stretched_spec, nperseg=n_fft, noverlap=n_fft - hop, boundary=True
    )

    ratio = Fraction(factor).limit_denominator(1000)
    shifted = signal.resample_poly(stretched, up=ratio.denominator, down=ratio.numerator)
    shifted = _fit_length(shifted.astype(np.float32), n)
    return np.nan_to_num(shifted).astype(np.float32)


# --------------------------------------------------------------------------
# Individual effects. Signature: (audio, sample_rate, intensity 0..1)
# --------------------------------------------------------------------------
def effect_normal(audio, sample_rate, intensity=1.0):
    return validate_audio(audio, sample_rate).copy()


def effect_deep(audio, sample_rate, intensity=1.0):
    return pitch_shift(audio, sample_rate, -6.0 * intensity)


def effect_high(audio, sample_rate, intensity=1.0):
    return pitch_shift(audio, sample_rate, 6.0 * intensity)


def effect_robot(audio, sample_rate, intensity=1.0):
    """Ring modulation: multiply the voice by a low-frequency sine wave."""
    data = validate_audio(audio, sample_rate)
    t = np.arange(len(data), dtype=np.float64) / sample_rate
    carrier = np.sin(2.0 * np.pi * 70.0 * t).astype(np.float32)
    return _mix(data, data * carrier, intensity).astype(np.float32)


def effect_echo(audio, sample_rate, intensity=1.0):
    """Add decaying delayed copies of the sound."""
    data = validate_audio(audio, sample_rate)
    out = data.copy()
    delay = int(0.28 * sample_rate)
    for k in range(1, 5):
        offset = delay * k
        if offset >= len(data):
            break
        out[offset:] += (0.5 ** k) * intensity * data[: len(data) - offset]
    return out.astype(np.float32)


def effect_radio(audio, sample_rate, intensity=1.0):
    """Narrow band-pass (like a small speaker) plus light distortion."""
    data = validate_audio(audio, sample_rate)
    nyquist = sample_rate / 2.0
    high_cut = min(3400.0, nyquist * 0.9)
    sos = signal.butter(4, [300.0, high_cut], btype="bandpass", fs=sample_rate, output="sos")
    wet = np.tanh(1.5 * signal.sosfilt(sos, data)).astype(np.float32)
    return _mix(data, wet, intensity).astype(np.float32)


def _vibrato(audio: np.ndarray, sample_rate: int, rate_hz: float, depth_s: float) -> np.ndarray:
    positions = np.arange(len(audio)) + depth_s * sample_rate * np.sin(
        2.0 * np.pi * rate_hz * np.arange(len(audio)) / sample_rate
    )
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def effect_alien(audio, sample_rate, intensity=1.0):
    """Higher pitch plus a fast wobble."""
    data = validate_audio(audio, sample_rate)
    shifted = pitch_shift(data, sample_rate, 4.0 * intensity)
    return _vibrato(shifted, sample_rate, rate_hz=6.0, depth_s=0.003 * intensity)


EFFECTS: Dict[str, Callable] = {
    "normal": effect_normal,
    "deep": effect_deep,
    "high": effect_high,
    "robot": effect_robot,
    "echo": effect_echo,
    "radio": effect_radio,
    "alien": effect_alien,
}


# --------------------------------------------------------------------------
# One entry point used by the app
# --------------------------------------------------------------------------
def process(
    audio,
    sample_rate: int,
    effect: str = "normal",
    intensity: float = 1.0,
    pitch_semitones: float = 0.0,
    volume: float = 1.0,
) -> np.ndarray:
    """Apply extra pitch, then an effect, then volume, then clipping protection."""
    if effect not in EFFECTS:
        raise ValueError(f"Unknown effect {effect!r}. Choose from: {', '.join(EFFECTS)}")
    if not np.isfinite(intensity):
        raise ValueError("intensity must be a finite number")
    intensity = float(np.clip(intensity, 0.0, 1.0))

    data = validate_audio(audio, sample_rate)
    if pitch_semitones:
        data = pitch_shift(data, sample_rate, pitch_semitones)
    data = EFFECTS[effect](data, sample_rate, intensity)
    data = apply_volume(data, volume)
    return prevent_clipping(data)
