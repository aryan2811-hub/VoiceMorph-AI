"""Prints environment info so we can verify the setup."""
import platform
import sys

import numpy as np
import scipy
import sounddevice as sd
import soundfile as sf


def main() -> None:
    print("Python      :", sys.version.split()[0])
    print("Windows     :", platform.platform())
    print("NumPy       :", np.__version__)
    print("SciPy       :", scipy.__version__)
    print("SoundFile   :", sf.__version__)
    print("sounddevice :", sd.__version__)
    print()
    print("Default input/output device indexes:", sd.default.device)
    print()
    print("Audio devices found:")
    for index, device in enumerate(sd.query_devices()):
        print(
            f"  [{index}] {device['name']} | "
            f"in={device['max_input_channels']} "
            f"out={device['max_output_channels']} | "
            f"default rate={int(device['default_samplerate'])} Hz"
        )


if __name__ == "__main__":
    main()