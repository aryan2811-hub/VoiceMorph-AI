"""AI voice section: pick files or record from the mic, convert in .venv-ai, play the result."""
import os
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_PYTHON = PROJECT_ROOT / ".venv-ai" / "Scripts" / "python.exe"
AI_SCRIPT = PROJECT_ROOT / "src" / "ai_convert.py"
OUTPUT_FILE = PROJECT_ROOT / "ai_output" / "converted.wav"
RECORDING_FILE = PROJECT_ROOT / "ai_output" / "recording.wav"
AUDIO_FILTER = "Audio (*.wav *.mp3 *.m4a *.flac *.ogg);;All files (*.*)"
MIN_RECORD_SECONDS = 0.5


def build_command(python_exe, script, source, ref, output):
    """Return (program, arguments) for the AI process."""
    args = [str(script), "--source", str(source), "--ref", str(ref), "--output", str(output)]
    return str(python_exe), args


def check_inputs(source, ref, python_exe, script):
    """Return an error message, or None if everything needed exists."""
    if not source:
        return "Pehle source audio chuno (jo awaaz convert karni hai)."
    if not ref:
        return "Pehle reference audio chuno (jis awaaz mein convert karna hai)."
    if not Path(source).is_file():
        return f"Source file nahi mili: {source}"
    if not Path(ref).is_file():
        return f"Reference file nahi mili: {ref}"
    if not Path(python_exe).is_file():
        return f"AI Python nahi mila: {python_exe}"
    if not Path(script).is_file():
        return f"ai_convert.py nahi mili: {script}"
    return None


def write_wav(path, samples, rate) -> None:
    """Save mono float samples (-1..1) as a 16-bit WAV file."""
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm.tobytes())


class AIPanel(QGroupBox):
    def __init__(self, get_mic=None, is_engine_running=None) -> None:
        super().__init__("AI voice (record / file, then convert)")
        self.get_mic = get_mic or (lambda: None)
        self.is_engine_running = is_engine_running or (lambda: False)
        self.source = None
        self.ref = None
        self.process = None
        self.log = ""
        self.auto_play = False
        self.recorder = None
        self.chunks = []
        self.rate = 48000
        self.rec_seconds = 0

        self.ref_btn = QPushButton("1. Reference voice chuno...")
        self.ref_label = QLabel("(koi file nahi)")
        self.record_btn = QPushButton("2. Record (bolna shuru karo)")
        self.source_btn = QPushButton("Ya: Source audio file chuno...")
        self.source_label = QLabel("(koi file nahi)")
        self.convert_btn = QPushButton("Convert")
        self.play_btn = QPushButton("Converted awaaz chalao")
        self.play_btn.setEnabled(False)
        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        for lab in (self.source_label, self.ref_label):
            lab.setWordWrap(True)

        lay = QVBoxLayout(self)
        for w in (self.ref_btn, self.ref_label, self.record_btn, self.source_btn,
                  self.source_label, self.convert_btn, self.play_btn, self.status):
            lay.addWidget(w)

        self.rec_timer = QTimer(self)
        self.rec_timer.timeout.connect(self.tick_record)

        self.ref_btn.clicked.connect(self.pick_ref)
        self.record_btn.clicked.connect(self.toggle_record)
        self.source_btn.clicked.connect(self.pick_source)
        self.convert_btn.clicked.connect(lambda: self.convert())
        self.play_btn.clicked.connect(self.play)

    # ---------- file picking ----------
    def pick_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Source audio", "", AUDIO_FILTER)
        if path:
            self.source = path
            self.source_label.setText(path)

    def pick_ref(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Reference voice", "", AUDIO_FILTER)
        if path:
            self.ref = path
            self.ref_label.setText(path)

    # ---------- recording ----------
    def _on_audio(self, indata, frames, time_info, status) -> None:
        self.chunks.append(indata[:, 0].copy())

    def toggle_record(self) -> None:
        if self.recorder is not None:
            self.stop_record()
            return
        if self.process is not None:
            self.status.setText("Convert chal raha hai, thoda ruko.")
            return
        if self.is_engine_running():
            self.status.setText("Pehle upar wala live effect Stop karo, phir record karo.")
            return
        if not self.ref:
            self.status.setText("Pehle reference voice chuno (step 1).")
            return
        dev = self.get_mic()
        try:
            if dev is None:
                info = sd.query_devices(kind="input")
            else:
                info = sd.query_devices(dev, "input")
            self.rate = int(info["default_samplerate"])
            self.chunks = []
            self.recorder = sd.InputStream(
                device=dev, channels=1, samplerate=self.rate,
                dtype="float32", callback=self._on_audio,
            )
            self.recorder.start()
        except Exception as exc:  # sounddevice raises many error types
            self.recorder = None
            self.status.setText(f"Mic shuru nahi hua: {exc}")
            return
        self.rec_seconds = 0
        self.rec_timer.start(1000)
        self.record_btn.setText("Stop (recording chal rahi hai: 0 s)")
        self.convert_btn.setEnabled(False)
        self.status.setText("Bolo... jab ho jaye toh Stop dabao.")

    def tick_record(self) -> None:
        self.rec_seconds += 1
        self.record_btn.setText(f"Stop (recording chal rahi hai: {self.rec_seconds} s)")

    def stop_record(self) -> None:
        self.rec_timer.stop()
        try:
            self.recorder.stop()
            self.recorder.close()
        except Exception:
            pass
        self.recorder = None
        self.record_btn.setText("2. Record (bolna shuru karo)")
        self.convert_btn.setEnabled(True)
        if not self.chunks:
            self.status.setText("Kuch record nahi hua. Mic check karo.")
            return
        samples = np.concatenate(self.chunks)
        if len(samples) < MIN_RECORD_SECONDS * self.rate:
            self.status.setText("Recording bahut chhoti thi, dobara try karo.")
            return
        write_wav(RECORDING_FILE, samples, self.rate)
        self.source = str(RECORDING_FILE)
        self.source_label.setText(f"{RECORDING_FILE} ({len(samples) / self.rate:.1f} s)")
        self.auto_play = True
        self.convert()

    # ---------- convert / play ----------
    def convert(self) -> None:
        if self.process is not None or self.recorder is not None:
            return
        err = check_inputs(self.source, self.ref, AI_PYTHON, AI_SCRIPT)
        if err:
            self.auto_play = False
            self.status.setText(err)
            return
        program, args = build_command(AI_PYTHON, AI_SCRIPT, self.source, self.ref, OUTPUT_FILE)

        env = QProcessEnvironment.systemEnvironment()
        for d in (r"D:\temp\torch_home", r"D:\temp\pip_tmp"):
            os.makedirs(d, exist_ok=True)
        env.insert("TORCH_HOME", r"D:\temp\torch_home")
        env.insert("TEMP", r"D:\temp\pip_tmp")
        env.insert("TMP", r"D:\temp\pip_tmp")
        env.insert("PYTHONIOENCODING", "utf-8")

        self.log = ""
        self.process = QProcess(self)
        self.process.setProcessEnvironment(env)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.on_output)
        self.process.finished.connect(self.on_finished)
        self.convert_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.status.setText("Shuru ho raha hai...")
        self.process.start(program, args)

    def on_output(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.log += text
        for line in text.splitlines():
            if line.startswith("STATUS: "):
                self.status.setText(line[len("STATUS: "):])

    def on_finished(self, code, _status) -> None:
        self.convert_btn.setEnabled(True)
        self.record_btn.setEnabled(True)
        ok = code == 0 and OUTPUT_FILE.is_file()
        if ok:
            self.play_btn.setEnabled(True)
            if self.auto_play:
                self.play()
        else:
            self.status.setText("Convert fail hua:\n" + self.log[-500:])
        self.auto_play = False
        self.process.deleteLater()
        self.process = None

    def play(self) -> None:
        if OUTPUT_FILE.is_file():
            os.startfile(str(OUTPUT_FILE))  # opens in the default Windows player

    def shutdown(self) -> None:
        self.rec_timer.stop()
        if self.recorder is not None:
            try:
                self.recorder.stop()
                self.recorder.close()
            except Exception:
                pass
            self.recorder = None
        if self.process is not None:
            self.process.kill()
            self.process.waitForFinished(2000)