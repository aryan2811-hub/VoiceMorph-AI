"""Streaming (block-by-block) voice effects for the real-time engine.

Each effect keeps its own state between blocks, so processing a sound in many
small pieces gives the same result as processing it all at once.

Rules for code in here (it runs inside the audio callback):
  * no file access, no printing, no logging
  * no waiting or locks
  * keep the work small
Pitch effects (deep, high, alien) use a simple low-delay streaming pitch
shifter. It is lower quality than the offline one; see PitchShift below.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


class StreamEffect:
    """Base class. process() takes a mono float32 block and returns one of the same length."""

    name = "normal"

    def process(self, block: np.ndarray) -> np.ndarray:
        return block


class PassThrough(StreamEffect):
    name = "normal"


class RingModRobot(StreamEffect):
    """Robot voice: multiply by a 70 Hz sine wave, blended with the original."""

    name = "robot"

    def __init__(self, sample_rate: int, intensity: float = 1.0, freq: float = 70.0):
        self.intensity = float(np.clip(intensity, 0.0, 1.0))
        self._step = 2.0 * np.pi * freq / sample_rate
        self._phase = 0.0

    def process(self, block):
        n = len(block)
        phases = self._phase + self._step * np.arange(n)
        self._phase = (self._phase + self._step * n) % (2.0 * np.pi)
        wet = block * np.sin(phases).astype(np.float32)
        return ((1.0 - self.intensity) * block + self.intensity * wet).astype(np.float32)


class Echo(StreamEffect):
    """Echo using a circular delay line: y[n] = x[n] + g * y[n - delay]."""

    name = "echo"

    def __init__(self, sample_rate: int, intensity: float = 1.0, delay_s: float = 0.28):
        self.delay = max(int(delay_s * sample_rate), 1)
        self.gain = 0.5 * float(np.clip(intensity, 0.0, 1.0))
        self._buf = np.zeros(self.delay, dtype=np.float32)
        self._pos = 0

    def process(self, block):
        n = len(block)
        if n > self.delay:
            raise ValueError("block is longer than the echo delay")
        idx = (self._pos + np.arange(n)) % self.delay
        out = block + self.gain * self._buf[idx]
        self._buf[idx] = out
        self._pos = (self._pos + n) % self.delay
        return out.astype(np.float32)


class Radio(StreamEffect):
    """Narrow band-pass filter plus light distortion (small-speaker sound)."""

    name = "radio"

    def __init__(self, sample_rate: int, intensity: float = 1.0):
        self.intensity = float(np.clip(intensity, 0.0, 1.0))
        high = min(3400.0, sample_rate / 2.0 * 0.9)
        self._sos = signal.butter(4, [300.0, high], btype="bandpass", fs=sample_rate, output="sos")
        self._zi = np.zeros((self._sos.shape[0], 2))

    def process(self, block):
        filtered, self._zi = signal.sosfilt(self._sos, block, zi=self._zi)
        wet = np.tanh(1.5 * filtered)
        return ((1.0 - self.intensity) * block + self.intensity * wet).astype(np.float32)


class PitchShift(StreamEffect):
    """Streaming pitch shifter (two crossfaded, sliding delay taps).

    How it works: reading old audio from a delay line whose delay slowly
    changes plays it faster or slower, which changes pitch. The delay must
    jump back now and then, so two taps half a window apart are crossfaded:
    one is always at full volume while the other jumps (silently).

    Honest limits: this is the cheap, low-delay method. It adds about half a
    window (~20 ms) of delay and sounds a little "warbly" compared with the
    offline phase-vocoder shifter. Pitch on pure tones can be off by about
    2%. A longer window is more accurate but adds more delay.
    """

    name = "pitch"

    def __init__(self, sample_rate: int, semitones: float, window_s: float = 0.04):
        self.ratio = 2.0 ** (float(semitones) / 12.0)
        self.window = float(max(int(window_s * sample_rate), 64))
        self._hist_len = int(self.window) + 3
        self._hist = np.zeros(self._hist_len, dtype=np.float32)
        self._phase = 0.0

    def process(self, block):
        if abs(self.ratio - 1.0) < 1e-9:
            return block
        n = len(block)
        w = self.window
        step = 1.0 - self.ratio
        i = np.arange(n)
        p1 = (self._phase + step * i) % w
        p2 = (p1 + 0.5 * w) % w
        buf = np.concatenate((self._hist, block))

        def tap(p):
            pos = self._hist_len + i - (p + 1.0)  # delay is always >= 1 sample
            lo = np.floor(pos).astype(np.int64)
            frac = (pos - lo).astype(np.float32)
            return buf[lo] * (1.0 - frac) + buf[lo + 1] * frac

        w1 = np.sin(np.pi * p1 / w) ** 2  # 0 when tap 1 jumps; tap 2 is then full
        out = w1 * tap(p1) + (1.0 - w1) * tap(p2)
        self._phase = (self._phase + step * n) % w
        self._hist = buf[-self._hist_len:]
        return out.astype(np.float32)


class Alien(StreamEffect):
    """Pitch up 4 semitones plus a 6 Hz wobble in volume (tremolo).

    This differs from the offline alien effect, which uses a pitch wobble.
    """

    name = "alien"

    def __init__(self, sample_rate: int, intensity: float = 1.0):
        self.intensity = float(np.clip(intensity, 0.0, 1.0))
        self._pitch = PitchShift(sample_rate, 4.0 * self.intensity)
        self._step = 2.0 * np.pi * 6.0 / sample_rate
        self._phase = 0.0

    def process(self, block):
        n = len(block)
        out = self._pitch.process(block)
        phases = self._phase + self._step * np.arange(n)
        self._phase = (self._phase + self._step * n) % (2.0 * np.pi)
        tremolo = 1.0 - 0.4 * self.intensity * (0.5 + 0.5 * np.sin(phases))
        return (out * tremolo).astype(np.float32)


STREAM_EFFECTS = {
    "normal": lambda sr, i: PassThrough(),
    "deep": lambda sr, i: PitchShift(sr, -6.0 * i),
    "high": lambda sr, i: PitchShift(sr, 6.0 * i),
    "robot": lambda sr, i: RingModRobot(sr, i),
    "echo": lambda sr, i: Echo(sr, i),
    "radio": lambda sr, i: Radio(sr, i),
    "alien": lambda sr, i: Alien(sr, i),
}


def make_effect(name: str, sample_rate: int, intensity: float = 1.0) -> StreamEffect:
    if name not in STREAM_EFFECTS:
        raise ValueError(
            f"Effect {name!r} is not available in real time yet. "
            f"Available: {', '.join(STREAM_EFFECTS)}"
        )
    return STREAM_EFFECTS[name](sample_rate, intensity)
