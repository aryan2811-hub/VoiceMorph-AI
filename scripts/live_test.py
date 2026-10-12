"""Live voice test: microphone -> effect -> headphones.

USE HEADPHONES. With speakers the microphone hears them and you get a loud howl.

Examples (run from the project folder, with the venv active):
    python scripts\\live_test.py --input 9 --output 8 --effect robot
    python scripts\\live_test.py --input 9 --output 8 --effect echo --seconds 20
Press Ctrl+C to stop early.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio_engine import AudioEngine, EngineError  # noqa: E402
from src.stream_effects import STREAM_EFFECTS  # noqa: E402


def bar(level: float, width: int = 20) -> str:
    filled = min(width, int(level * width * 2))
    return "#" * filled + "-" * (width - filled)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=int, default=None)
    p.add_argument("--output", type=int, default=None)
    p.add_argument("--effect", default="normal", choices=list(STREAM_EFFECTS))
    p.add_argument("--intensity", type=float, default=1.0)
    p.add_argument("--volume", type=float, default=1.0)
    p.add_argument("--rate", type=int, default=48_000)
    p.add_argument("--block", type=int, default=512)
    p.add_argument("--seconds", type=float, default=15.0)
    a = p.parse_args()

    engine = AudioEngine(a.rate, a.block)
    engine.set_effect(a.effect, a.intensity)
    engine.set_volume(a.volume)

    try:
        engine.start(a.input, a.output)
    except EngineError as exc:
        print(f"\nCould not start: {exc}")
        return 1

    s = engine.status()
    print(f"Running: effect={a.effect}, {a.rate} Hz, block={a.block} samples ({s['block_ms']:.1f} ms)")
    print(f"Driver-reported latency (estimate): {s['reported_latency_ms']:.0f} ms")
    print("Speak now. Ctrl+C to stop.\n")

    end = time.time() + a.seconds
    try:
        while time.time() < end:
            time.sleep(0.2)
            if not engine.check_stream_alive():
                print("\nThe audio stream stopped. Was a device unplugged?")
                break
            s = engine.status()
            print(
                f"\rIN [{bar(s['input_peak'])}]  OUT [{bar(s['output_peak'])}]  "
                f"underflow={s['underflows']} overflow={s['overflows']} errors={s['callback_errors']}   ",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()

    s = engine.status()
    print("\n\n--- Summary ---")
    print(f"Callbacks run          : {s['callbacks']}")
    print(f"Slowest callback       : {s['max_callback_ms']:.2f} ms (must stay well under {s['block_ms']:.1f} ms)")
    print(f"Underflows / overflows : {s['underflows']} / {s['overflows']} (audio dropouts)")
    print(f"Callback errors        : {s['callback_errors']}  {s['last_error'] or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
