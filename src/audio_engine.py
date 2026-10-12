"""Real-time audio engine: microphone -> effect -> speakers/headphones.

Design:
  * One duplex sounddevice stream. PortAudio calls _callback for every block.
  * The callback only runs the effect, applies volume, clips and writes output.
    It never touches the disk, the GUI or blocking functions.
  * Control methods (set_effect, set_volume...) run in your own thread and only
    swap simple values or object references, which is safe in Python.
  * Problems found in the callback are counted and stored. They are reported
    from start/stop/status, never printed from inside the callback.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.stream_effects import StreamEffect, make_effect

log = logging.getLogger(__name__)

VALID_RATES = (16_000, 22_050, 32_000, 44_100, 48_000, 96_000)


class EngineError(Exception):
    """Raised with a clear message when the engine cannot start or run."""


class AudioEngine:
    def __init__(self, sample_rate: int = 48_000, blocksize: int = 512):
        if sample_rate not in VALID_RATES:
            raise EngineError(f"Unsupported sample rate {sample_rate}. Use one of {VALID_RATES}")
        if not 64 <= blocksize <= 8192:
            raise EngineError("blocksize must be between 64 and 8192 samples")
        self.sample_rate = sample_rate
        self.blocksize = blocksize

        self._effect_name = "normal"
        self._intensity = 1.0
        self._effect: StreamEffect = make_effect("normal", sample_rate)
        self._volume = 1.0
        self._muted = False

        self._stream = None
        self._running = False

        # Written by the callback, read by the control thread.
        self.input_peak = 0.0
        self.output_peak = 0.0
        self.underflow_count = 0
        self.overflow_count = 0
        self.callback_errors = 0
        self.last_error: Optional[str] = None
        self.callback_count = 0
        self.max_callback_seconds = 0.0

    # ---------------- control (not real-time) ----------------
    def set_effect(self, name: str, intensity: float = 1.0) -> None:
        """Switch effect. A new effect object is built here, then swapped in."""
        new_effect = make_effect(name, self.sample_rate, intensity)  # raises if unknown
        self._effect = new_effect
        self._effect_name = name
        self._intensity = float(np.clip(intensity, 0.0, 1.0))

    def set_volume(self, volume: float) -> None:
        if not np.isfinite(volume) or volume < 0:
            raise EngineError("volume must be a number >= 0")
        self._volume = float(min(volume, 4.0))

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def effect_name(self) -> str:
        return self._effect_name

    # ---------------- the real-time callback ----------------
    def _callback(self, indata, outdata, frames, time_info, status) -> None:
        import time

        t0 = time.perf_counter()
        try:
            if status:
                if getattr(status, "input_overflow", False):
                    self.overflow_count += 1
                if getattr(status, "output_underflow", False):
                    self.underflow_count += 1

            block = indata[:, 0]
            self.input_peak = float(np.max(np.abs(block))) if frames else 0.0

            if self._muted:
                outdata.fill(0.0)
                self.output_peak = 0.0
            else:
                out = self._effect.process(block)
                out = np.clip(out * self._volume, -1.0, 1.0)
                outdata[:, 0] = out
                self.output_peak = float(np.max(np.abs(out))) if frames else 0.0
        except Exception as exc:  # never let an exception escape into PortAudio
            outdata.fill(0.0)
            self.callback_errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self.callback_count += 1
            elapsed = time.perf_counter() - t0
            if elapsed > self.max_callback_seconds:
                self.max_callback_seconds = elapsed

    # ---------------- start / stop ----------------
    def start(self, input_device: Optional[int] = None, output_device: Optional[int] = None) -> None:
        if self._running:
            raise EngineError("The engine is already running")
        try:
            import sounddevice as sd
        except Exception as exc:
            raise EngineError(f"Audio system is not available: {exc}") from exc

        try:
            sd.check_input_settings(
                device=input_device, channels=1, dtype="float32", samplerate=self.sample_rate
            )
            sd.check_output_settings(
                device=output_device, channels=1, dtype="float32", samplerate=self.sample_rate
            )
        except Exception as exc:
            raise EngineError(
                f"These devices do not support {self.sample_rate} Hz mono: {exc}\n"
                "Try the other sample rate (44100 or 48000), or other device numbers."
            ) from exc

        self._reset_stats()
        self._effect = make_effect(self._effect_name, self.sample_rate, self._intensity)
        try:
            self._stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                device=(input_device, output_device),
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise EngineError(
                f"Could not open the audio stream: {exc}\n"
                "Tip: choose an input and an output device from the SAME host API "
                "(for example both 'MME')."
            ) from exc
        self._running = True
        log.info("Engine started at %d Hz, block %d", self.sample_rate, self.blocksize)

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        self._running = False
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                log.warning("Error while closing stream: %s", exc)
        log.info("Engine stopped")

    def _reset_stats(self) -> None:
        self.input_peak = self.output_peak = 0.0
        self.underflow_count = self.overflow_count = 0
        self.callback_errors = self.callback_count = 0
        self.max_callback_seconds = 0.0
        self.last_error = None

    # ---------------- information ----------------
    def block_duration_ms(self) -> float:
        return 1000.0 * self.blocksize / self.sample_rate

    def reported_latency_ms(self) -> Optional[float]:
        """Input + output latency as reported by the driver. This is an estimate, not a measurement."""
        if self._stream is None:
            return None
        lat_in, lat_out = self._stream.latency
        return 1000.0 * (float(lat_in) + float(lat_out))

    def check_stream_alive(self) -> bool:
        """False if the stream stopped by itself (for example a device was unplugged)."""
        if not self._running or self._stream is None:
            return False
        try:
            alive = bool(self._stream.active)
        except Exception:
            alive = False
        if not alive:
            self._running = False
            self.last_error = self.last_error or "The audio stream stopped (device disconnected?)"
        return alive

    def status(self) -> dict:
        return {
            "running": self._running,
            "effect": self._effect_name,
            "input_peak": self.input_peak,
            "output_peak": self.output_peak,
            "underflows": self.underflow_count,
            "overflows": self.overflow_count,
            "callback_errors": self.callback_errors,
            "last_error": self.last_error,
            "block_ms": self.block_duration_ms(),
            "reported_latency_ms": self.reported_latency_ms(),
            "max_callback_ms": 1000.0 * self.max_callback_seconds,
            "callbacks": self.callback_count,
        }
