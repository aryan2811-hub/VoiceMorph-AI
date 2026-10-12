\# VoiceMorph-AI



A real-time voice changer for Windows. Speak into your microphone and hear your

voice with an effect applied, live. It also has an optional AI mode that records

your voice and converts it to the voice of a reference recording.



\## Features



\- \*\*Real-time effects:\*\* normal, deep, high, robot, echo, radio, alien

\- \*\*Desktop app\*\* (PySide6) with microphone and output selection, intensity, volume, mute and live input/output level meters

\- \*\*Live health status:\*\* block size, slowest callback, underflow and overflow counts

\- \*\*Settings are remembered\*\* between runs (and a broken settings file never crashes the app)

\- \*\*AI voice (optional):\*\* record or pick a file, convert it to a reference voice with kNN-VC, and play the result. It runs in a separate process so the window never freezes.

\- \*\*164 automated tests\*\* (pytest)



\## Download (Windows, no Python needed)



1\. Go to the \*\*Releases\*\* page: https://github.com/YOUR-USERNAME/VoiceMorph-AI/releases

2\. Download `VoiceMorph-win.zip` and extract it.

3\. Open the extracted folder and run `VoiceMorph.exe`.



Notes:

\- Windows may show "Windows protected your PC" because the app is not code-signed. Click \*\*More info\*\*, then \*\*Run anyway\*\*.

\- Use \*\*headphones\*\*. With speakers, your output goes back into the microphone and you will hear feedback.

\- The downloaded app includes the real-time effects only. The AI voice mode is not included (see below).



\## Run from source



Requires Python 3.13 on Windows.



```powershell

git clone https://github.com/YOUR-USERNAME/VoiceMorph-AI.git

cd VoiceMorph-AI

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

python app.py

```



Run the tests:



```powershell

pip install -r requirements-dev.txt

python -m pytest tests -q

```



Build the `.exe` yourself:



```powershell

pyinstaller --noconfirm --windowed --name VoiceMorph --exclude-module torch --exclude-module torchaudio --exclude-module torchvision app.py

```



\## AI voice mode (optional, advanced)



The AI mode uses PyTorch and kNN-VC. It runs in a \*\*separate Python environment\*\*

(`.venv-ai` in the project root) so the real-time app stays small and fast. It is

\*\*not\*\* part of the downloadable `.exe`. A GPU is strongly recommended; on the

author's RTX 3050 (6 GB) the model used up to about 5.5 GB of GPU memory.



Do not run live effects and AI conversion at the same time.



\## Project structure



```

app.py              desktop app (window, controls, meters)

src/audio\_engine.py real-time audio engine

src/audio\_io.py     reading and writing audio files

src/stream\_effects.py  streaming (chunk by chunk) effects used live

src/voice\_effects.py   effect building blocks

src/app\_settings.py    safe load/save of settings

src/ai\_panel.py     AI section: record, convert, play

src/ai\_convert.py   AI conversion script (runs in .venv-ai)

tests/              pytest tests

scripts/            small helper scripts (environment check, demos, live test)

```



\## Honest limitations



\- Windows only, and tested on the author's machine. If something does not work on yours, please open an issue.

\- Real-time AI voice conversion is \*\*not\*\* supported. The AI mode is "record, convert, play": you record, press Stop, and the converted voice plays after processing.

\- The first AI conversion is slower because the model has to load.

\- The `.exe` is unsigned, so Windows SmartScreen may warn about it.

