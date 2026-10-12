"""VoiceMorph-AI - Phase 4b: desktop window that remembers your settings.

Run from the project folder (see run.ps1):
    powershell -ExecutionPolicy Bypass -File .\\run.ps1

USE HEADPHONES. With speakers the microphone hears them and you get a loud howl.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sounddevice as sd  # noqa: E402
from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.app_settings import load_settings, save_settings  # noqa: E402
from src.audio_engine import AudioEngine, EngineError  # noqa: E402
from src.stream_effects import STREAM_EFFECTS  # noqa: E402
from src.ai_panel import AIPanel  # noqa: E402

SETTINGS_PATH = (
    Path.home() / 'AppData' / 'Roaming' / 'VoiceMorph' / 'settings.json'
    if getattr(sys, 'frozen', False)
    else Path(__file__).resolve().parent / 'configs' / 'settings.json'
)


def list_devices() -> list[dict]:
    """All audio devices with their host API name (MME, WASAPI, ...)."""
    hostapis = sd.query_hostapis()
    devices = []
    for i, d in enumerate(sd.query_devices()):
        devices.append(
            {
                "index": i,
                "name": d["name"],
                "in": d["max_input_channels"],
                "out": d["max_output_channels"],
                "host": hostapis[d["hostapi"]]["name"],
            }
        )
    return devices


def meter_value(level: float) -> int:
    """Turn a 0..1 peak into 0..100 for the bar (same scaling as live_test.py)."""
    return max(0, min(100, int(level * 200)))


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VoiceMorph-AI")
        self.setMinimumWidth(460)

        self.engine = AudioEngine(48_000, 512)
        self.devices = list_devices()
        self.devices_by_index = {d["index"]: d for d in self.devices}

        # ---- widgets ----
        self.host_box = QComboBox()
        self.in_box = QComboBox()
        self.out_box = QComboBox()

        self.effect_box = QComboBox()
        self.effect_box.addItems(list(STREAM_EFFECTS))

        self.intensity = QSlider(Qt.Horizontal)
        self.intensity.setRange(0, 100)
        self.intensity_label = QLabel()

        # Volume is capped at 100% on purpose: above that the sound clips and gets harsh.
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume_label = QLabel()

        self.mute_box = QCheckBox("Mute")

        self.in_meter = QProgressBar()
        self.out_meter = QProgressBar()
        for m in (self.in_meter, self.out_meter):
            m.setRange(0, 100)
            m.setTextVisible(False)

        self.start_btn = QPushButton("Start")
        self.status_label = QLabel("Stopped. Use headphones to avoid howling.")
        self.status_label.setWordWrap(True)

        # ---- layout ----
        form = QFormLayout()
        form.addRow("Audio system", self.host_box)
        form.addRow("Microphone", self.in_box)
        form.addRow("Output", self.out_box)
        form.addRow("Effect", self.effect_box)
        form.addRow("Intensity", self.intensity)
        form.addRow("", self.intensity_label)
        form.addRow("Volume", self.volume)
        form.addRow("", self.volume_label)
        form.addRow("", self.mute_box)
        form.addRow("IN level", self.in_meter)
        form.addRow("OUT level", self.out_meter)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.status_label)
        self.ai_panel = AIPanel(self.in_box.currentData, lambda: self.engine.is_running)
        layout.addWidget(self.ai_panel)    

        # ---- fill device lists, then restore the last settings ----
        hosts = []
        for d in self.devices:
            if d["host"] not in hosts:
                hosts.append(d["host"])
        hosts.sort(key=lambda h: (h != "MME", h))  # MME first, it worked for you
        self.host_box.addItems(hosts)

        saved = load_settings(SETTINGS_PATH)
        if saved["host"]:
            i = self.host_box.findText(saved["host"])
            if i >= 0:
                self.host_box.setCurrentIndex(i)
        self.fill_devices()  # fills the lists with defaults for this audio system
        self.select_device(self.in_box, saved["input"])
        self.select_device(self.out_box, saved["output"])
        i = self.effect_box.findText(saved["effect"])
        if i >= 0:
            self.effect_box.setCurrentIndex(i)
        self.intensity.setValue(saved["intensity"])
        self.volume.setValue(saved["volume"])
        self.mute_box.setChecked(saved["muted"])
        self.apply_settings()

        # ---- signals (connected after restoring, so restoring does not trigger them) ----
        self.host_box.currentIndexChanged.connect(self.fill_devices)
        self.effect_box.currentIndexChanged.connect(self.apply_settings)
        self.intensity.valueChanged.connect(self.apply_settings)
        self.volume.valueChanged.connect(self.apply_settings)
        self.mute_box.toggled.connect(self.apply_settings)
        self.start_btn.clicked.connect(self.toggle)

        # ---- timer: refresh meters 10 times per second ----
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)

    # ---------------- helpers ----------------
    def fill_devices(self) -> None:
        host = self.host_box.currentText()
        self.in_box.clear()
        self.out_box.clear()
        default_in, default_out = sd.default.device
        for d in self.devices:
            if d["host"] != host:
                continue
            if d["in"] > 0:
                self.in_box.addItem(f"[{d['index']}] {d['name']}", d["index"])
            if d["out"] > 0:
                self.out_box.addItem(f"[{d['index']}] {d['name']}", d["index"])
        i = self.in_box.findData(default_in)
        if i >= 0:
            self.in_box.setCurrentIndex(i)
        o = self.out_box.findData(default_out)
        if o >= 0:
            self.out_box.setCurrentIndex(o)

    def select_device(self, box: QComboBox, name) -> None:
        """Select the entry whose device name matches `name` (if it still exists)."""
        if not name:
            return
        for i in range(box.count()):
            d = self.devices_by_index.get(box.itemData(i))
            if d and d["name"] == name:
                box.setCurrentIndex(i)
                return

    def device_name(self, box: QComboBox):
        d = self.devices_by_index.get(box.currentData())
        return d["name"] if d else None

    def current_settings(self) -> dict:
        return {
            "host": self.host_box.currentText() or None,
            "input": self.device_name(self.in_box),
            "output": self.device_name(self.out_box),
            "effect": self.effect_box.currentText(),
            "intensity": self.intensity.value(),
            "volume": self.volume.value(),
            "muted": self.mute_box.isChecked(),
        }

    def apply_settings(self) -> None:
        self.intensity_label.setText(f"{self.intensity.value()}%")
        self.volume_label.setText(f"{self.volume.value()}%")
        try:
            self.engine.set_effect(self.effect_box.currentText(), self.intensity.value() / 100.0)
            self.engine.set_volume(self.volume.value() / 100.0)
            self.engine.set_muted(self.mute_box.isChecked())
        except EngineError as exc:
            QMessageBox.warning(self, "Setting problem", str(exc))

    def set_running_ui(self, running: bool) -> None:
        self.start_btn.setText("Stop" if running else "Start")
        for w in (self.host_box, self.in_box, self.out_box):
            w.setEnabled(not running)
        if not running:
            self.in_meter.setValue(0)
            self.out_meter.setValue(0)

    # ---------------- actions ----------------
    def toggle(self) -> None:
        if self.engine.is_running:
            self.engine.stop()
            self.set_running_ui(False)
            self.status_label.setText("Stopped.")
            return

        in_dev = self.in_box.currentData()
        out_dev = self.out_box.currentData()
        if in_dev is None or out_dev is None:
            QMessageBox.warning(
                self, "No device", "Pick a microphone and an output device from the lists."
            )
            return
        self.apply_settings()
        try:
            self.engine.start(in_dev, out_dev)
        except EngineError as exc:
            QMessageBox.critical(self, "Could not start", str(exc))
            return
        self.set_running_ui(True)
        save_settings(SETTINGS_PATH, self.current_settings())

    def refresh(self) -> None:
        if not self.engine.is_running:
            return
        if not self.engine.check_stream_alive():
            self.engine.stop()
            self.set_running_ui(False)
            msg = self.engine.last_error or "The audio stream stopped."
            self.status_label.setText(msg)
            QMessageBox.warning(self, "Audio stopped", msg)
            return
        s = self.engine.status()
        self.in_meter.setValue(meter_value(s["input_peak"]))
        self.out_meter.setValue(meter_value(s["output_peak"]))
        self.status_label.setText(
            f"Running: {s['effect']} | block {s['block_ms']:.1f} ms | "
            f"slowest callback {s['max_callback_ms']:.2f} ms | "
            f"underflow {s['underflows']} overflow {s['overflows']} errors {s['callback_errors']}"
        )

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt name)
        self.timer.stop()
        self.ai_panel.shutdown()
        self.engine.stop()
        save_settings(SETTINGS_PATH, self.current_settings())
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())