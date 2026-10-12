"""AI voice conversion (kNN-VC). Runs in .venv-ai (Python 3.11).

Usage:
    python src/ai_convert.py --source a.wav --ref b.wav --output out.wav
"""
import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

import imageio_ffmpeg
import soundfile as sf
import torch
import torchaudio


def _load(path, normalize=True, **kwargs):
    # torchaudio.load needs torchcodec/FFmpeg DLLs; soundfile does not.
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load


def status(msg: str) -> None:
    print("STATUS: " + msg, flush=True)


def to_wav16k(src: Path, dst: Path) -> None:
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run(
        [ff, "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError("ffmpeg could not read: " + str(src))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--topk", type=int, default=4)
    a = p.parse_args()

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / "_tmp"
    tmp.mkdir(exist_ok=True)
    src16, ref16 = tmp / "source16k.wav", tmp / "ref16k.wav"

    status("Audio files taiyaar ho rahi hain...")
    to_wav16k(Path(a.source), src16)
    to_wav16k(Path(a.ref), ref16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    status("Model load ho raha hai (%s)..." % device)
    knn_vc = torch.hub.load(
        "bshall/knn-vc", "knn_vc",
        prematched=True, trust_repo=True, pretrained=True, device=device,
    )

    status("Reference voice process ho rahi hai (kuch second)...")
    matching_set = knn_vc.get_matching_set([str(ref16)])

    status("Convert ho raha hai...")
    t0 = time.time()
    query = knn_vc.get_features(str(src16))
    wav = knn_vc.match(query, matching_set, topk=a.topk)
    sf.write(str(out), wav.cpu().numpy(), 16000)
    status("Ho gaya (%.1f second). Saved: %s" % (time.time() - t0, out))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)