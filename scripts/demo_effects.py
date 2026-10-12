"""Record your voice, then play it back with each effect and save WAV files.

Run from the project folder:
    python scripts\\demo_effects.py
Optional: choose devices by index (see check_env.py):
    python scripts\\demo_effects.py --input 9 --output 8
"""
import argparse
import sys
from pathlib import Path

# Make "src" importable when running from the scripts folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import audio_io, voice_effects  # noqa: E402

SAMPLE_RATE = 44_100


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=int, default=None, help="microphone device index")
    parser.add_argument("--output", type=int, default=None, help="speaker device index")
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "recordings"

    try:
        input("Press Enter, then speak for %.0f seconds... " % args.seconds)
        print("Recording...")
        voice = audio_io.record(args.seconds, SAMPLE_RATE, args.input)
        peak = float(abs(voice).max())
        print(f"Done. Peak level: {peak:.3f}")
        if peak < 0.01:
            print("WARNING: the recording is almost silent. Check the microphone or device index.")

        audio_io.save_wav(out_dir / "original.wav", voice, SAMPLE_RATE)

        for name in voice_effects.EFFECTS:
            print(f"Playing: {name}")
            processed = voice_effects.process(voice, SAMPLE_RATE, effect=name)
            audio_io.save_wav(out_dir / f"{name}.wav", processed, SAMPLE_RATE)
            audio_io.play(processed, SAMPLE_RATE, args.output)
    except audio_io.AudioIOError as exc:
        print(f"\nAudio problem: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0

    print(f"\nSaved WAV files in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
