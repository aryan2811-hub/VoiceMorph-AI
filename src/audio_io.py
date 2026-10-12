"""Recording, playback, device listing and WAV import/export.

sounddevice is imported inside the functions that need it. That way the file
functions and the tests still work on a computer with no sound hardware.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

from src.voice_effects import validate_audio


class AudioIOError(Exception):
    """Raised with a clear message when recording, playback or file I/O fails."""


def _sd():
    try:
        import sounddevice as sd
        return sd
    except Exception as exc:  # PortAudio missing, etc.
        raise AudioIOError(f"Audio system is not available: {exc}") from exc


def list_devices() -> List[dict]:
    """Return a list of dicts: index, name, inputs, outputs, default_rate, hostapi."""
    sd = _sd()
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as exc:
        raise AudioIOError(f"Could not list audio devices: {exc}") from exc
    result = []
    for index, d in enumerate(devices):
        result.append(
            {
                "index": index,
                "name": d["name"],
                "inputs": d["max_input_channels"],
                "outputs": d["max_output_channels"],
                "default_rate": int(d["default_samplerate"]),
                "hostapi": hostapis[d["hostapi"]]["name"],
            }
        )
    return result


def record(
    seconds: float,
    sample_rate: int = 44_100,
    device: Optional[int] = None,
) -> np.ndarray:
    """Record mono audio from the microphone. Blocks until finished."""
    if not np.isfinite(seconds) or not 0 < seconds <= 600:
        raise AudioIOError("seconds must be between 0 and 600")
    sd = _sd()
    try:
        frames = int(seconds * sample_rate)
        data = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32", device=device)
        sd.wait()
    except Exception as exc:
        raise AudioIOError(
            f"Recording failed (device={device}, rate={sample_rate}): {exc}"
        ) from exc
    return validate_audio(data[:, 0], sample_rate)


def play(audio, sample_rate: int, device: Optional[int] = None) -> None:
    """Play audio on a speaker or headphone. Blocks until finished."""
    data = validate_audio(audio, sample_rate)
    sd = _sd()
    try:
        sd.play(data, samplerate=sample_rate, device=device)
        sd.wait()
    except Exception as exc:
        raise AudioIOError(
            f"Playback failed (device={device}, rate={sample_rate}): {exc}"
        ) from exc


def save_wav(path, audio, sample_rate: int) -> Path:
    """Save mono audio as a 16-bit WAV file. Creates the folder if needed."""
    data = validate_audio(audio, sample_rate)
    path = Path(path)
    if path.suffix.lower() != ".wav":
        raise AudioIOError(f"Output must be a .wav file, got {path.name!r}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), np.clip(data, -1.0, 1.0), int(sample_rate), subtype="PCM_16")
    except Exception as exc:
        raise AudioIOError(f"Could not save {path}: {exc}") from exc
    return path


def load_audio(path) -> Tuple[np.ndarray, int]:
    """Load an audio file (WAV, FLAC, OGG...) as mono float32 plus its sample rate."""
    path = Path(path)
    if not path.is_file():
        raise AudioIOError(f"File not found: {path}")
    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioIOError(f"Could not read {path.name}: {exc}") from exc
    return validate_audio(data.mean(axis=1), rate), int(rate)
