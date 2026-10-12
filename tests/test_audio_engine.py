"""Engine tests that need NO sound hardware: we call the callback by hand."""
import numpy as np
import pytest

from src.audio_engine import AudioEngine, EngineError

SR = 48_000


class FakeStatus:
    def __init__(self, under=False, over=False):
        self.output_underflow = under
        self.input_overflow = over

    def __bool__(self):
        return self.output_underflow or self.input_overflow


def blocks(engine, x, status=None):
    out = np.zeros((len(x), 1), dtype=np.float32)
    engine._callback(x.reshape(-1, 1), out, len(x), None, status or FakeStatus())
    return out[:, 0]


def tone(n=512, amp=0.3):
    return (amp * np.sin(2 * np.pi * 440 * np.arange(n) / SR)).astype(np.float32)


def test_pass_through_copies_input():
    e = AudioEngine(SR, 512)
    x = tone()
    assert np.allclose(blocks(e, x), x)
    assert e.input_peak == pytest.approx(0.3, abs=0.01)


def test_volume_and_clipping():
    e = AudioEngine(SR, 512)
    e.set_volume(2.0)
    assert np.max(np.abs(blocks(e, tone(amp=0.2)))) == pytest.approx(0.4, abs=0.01)
    e.set_volume(4.0)
    assert np.max(np.abs(blocks(e, tone(amp=0.9)))) <= 1.0


def test_mute_gives_silence():
    e = AudioEngine(SR, 512)
    e.set_muted(True)
    assert np.all(blocks(e, tone()) == 0.0)


def test_switch_effect_while_running_callbacks():
    e = AudioEngine(SR, 512)
    for name in ["robot", "echo", "radio", "deep", "high", "alien", "normal"]:
        e.set_effect(name, 0.8)
        out = blocks(e, tone())
        assert np.all(np.isfinite(out))
    assert e.callback_errors == 0


def test_unknown_effect_leaves_engine_unchanged():
    e = AudioEngine(SR, 512)
    with pytest.raises(ValueError):
        e.set_effect("nope")
    assert e.effect_name == "normal"


def test_underflow_and_overflow_counted():
    e = AudioEngine(SR, 512)
    blocks(e, tone(), FakeStatus(under=True))
    blocks(e, tone(), FakeStatus(over=True))
    assert e.underflow_count == 1 and e.overflow_count == 1


def test_exception_in_effect_gives_silence_not_crash():
    e = AudioEngine(SR, 512)

    class Broken:
        def process(self, block):
            raise RuntimeError("boom")

    e._effect = Broken()
    out = blocks(e, tone())
    assert np.all(out == 0.0)
    assert e.callback_errors == 1 and "boom" in e.last_error


def test_invalid_settings_rejected():
    with pytest.raises(EngineError):
        AudioEngine(12345, 512)
    with pytest.raises(EngineError):
        AudioEngine(SR, 10)
    with pytest.raises(EngineError):
        AudioEngine(SR, 512).set_volume(-1)


def test_stop_without_start_is_safe():
    e = AudioEngine(SR, 512)
    e.stop()
    e.stop()
    assert not e.is_running
    assert e.reported_latency_ms() is None
    assert e.check_stream_alive() is False


def test_status_and_block_duration():
    e = AudioEngine(SR, 480)
    assert e.block_duration_ms() == pytest.approx(10.0)
    assert e.status()["callbacks"] == 0
