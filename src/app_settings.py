"""Load and save the desktop app's settings (remembered between runs).

Stored as a small JSON file. A missing, broken or hand-edited file never
crashes the app: every value is checked and falls back to a safe default.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

DEFAULTS = {
    "host": None,        # audio system name, e.g. "MME"
    "input": None,       # microphone device name
    "output": None,      # output device name
    "effect": "normal",
    "intensity": 100,    # percent, 0..100
    "volume": 70,        # percent, 0..100 (capped at 100 on purpose)
    "muted": False,
}


def _clamp_int(value, low: int, high: int, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    if not math.isfinite(value):
        return default
    return int(max(low, min(high, round(value))))


def _opt_str(value):
    return value if isinstance(value, str) and value else None


def sanitize(raw) -> dict:
    """Return a complete, valid settings dict, whatever `raw` contains."""
    out = dict(DEFAULTS)
    if not isinstance(raw, dict):
        return out
    out["host"] = _opt_str(raw.get("host"))
    out["input"] = _opt_str(raw.get("input"))
    out["output"] = _opt_str(raw.get("output"))
    effect = raw.get("effect")
    if isinstance(effect, str) and effect:
        out["effect"] = effect
    out["intensity"] = _clamp_int(raw.get("intensity"), 0, 100, DEFAULTS["intensity"])
    out["volume"] = _clamp_int(raw.get("volume"), 0, 100, DEFAULTS["volume"])
    muted = raw.get("muted")
    if isinstance(muted, bool):
        out["muted"] = muted
    return out


def load_settings(path) -> dict:
    """Read settings from `path`. Missing or broken file -> defaults."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return sanitize(None)
    return sanitize(raw)


def save_settings(path, data: dict) -> bool:
    """Write settings safely (temp file, then replace). Returns False on failure."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(sanitize(data), indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False