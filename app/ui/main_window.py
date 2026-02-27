import os
import traceback
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout,
    QFileDialog, QMessageBox, QLabel, QSlider, QComboBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from .tab_view import TabView, NoteEvent
from .score_view import ScoreView
from .note_player import SynthNote, synth_notes_to_wav_path

from app.pipeline.run import run_pipeline
from app.live.audio_stream import AudioStreamConfig, LiveAudioInput
from app.transcription.live_basic_pitch import live_loop_basic_pitch
from app.transcription.basic_pitch_poly import BasicPitchConfig
from app.transcription.guitar_mapper import map_notes_to_guitar

# Optional filter (if you want it). If you remove this feature, delete these two lines.
from app.transcription.guitar_filter import guitar_only_pipeline


def _fmt_ms(ms: int) -> str:
    s = max(0, ms) // 1000
    m = s // 60
    s = s % 60
    return f"{m}:{s:02d}"


class Worker(QThread):
    done = pyqtSignal(list)
    fail = pyqtSignal(str)

    def __init__(self, audio_path: str, out_dir: str):
        super().__init__()
        self.audio_path = audio_path
        self.out_dir = out_dir

    def run(self):
        try:
            notes = run_pipeline(self.audio_path, out_dir=self.out_dir)
            self.done.emit(notes)
        except Exception:
            self.fail.emit(traceback.format_exc())


class LiveWorker(QThread):
    new_notes = pyqtSignal(list)   # list[dict]
    fail = pyqtSignal(str)

    def __init__(self, device_index: int | None):
        super().__init__()
        self.device_index = device_index
        self._stop = False
        self._audio: LiveAudioInput | None = None

    def request_stop(self):
        self._stop = True
        try:
            if self._audio:
                self._audio.stop()
        except Exception:
            pass

    def run(self):
        try:
            import sounddevice as sd
            import time

            cfg = AudioStreamConfig(sample_rate=44100, channels=1, buffer_seconds=12.0)
            self._audio = LiveAudioInput(cfg)

            # If None: use sounddevice default input
            dev = int(sd.default.device[0]) if self.device_index is None else int(self.device_index)

            self._audio.start(device=dev)
            capture_sr = int(self._audio.cfg.sample_rate)
            print(f"[LIVE] device={dev} capture_sr={capture_sr}")

            time.sleep(0.5)

            def should_stop() -> bool:
                return bool(self._stop)

            def on_new(events):
                # events: list[app.models.note_event.NoteEvent]
                # Optional: guitar-only filtering (remove if you don't want it)
                events = guitar_only_pipeline(events)

                mapped = map_notes_to_guitar(events, max_fret=24)

                payload = []
                for n in mapped:
                    if n.string is None or n.fret is None:
                        continue
                    f = int(n.fret)
                    if not (0 <= f <= 24):
                        continue
                    payload.append({
                        "pitch": int(n.pitch),
                        "onset_s": float(n.onset),
                        "dur_s": float(n.duration),
                        "string": int(n.string),
                        "fret": f,
                    })

                if payload:
                    self.new_notes.emit(payload)

            # Basic Pitch live settings tuned for better accuracy
            live_loop_basic_pitch(
                get_audio_window=lambda sec: self._audio.ring.get_last_seconds(sec),
                get_stream_time=lambda: self._audio.ring.stream_time(),
                capture_sr=capture_sr,
                on_new=on_new,
                should_stop=should_stop,
                window_sec=3.0,
                hop_sec=0.5,
                cfg=BasicPitchConfig(prefer_cpu=True),
                rms_gate=0.0003,
            )

        except Exception:
            self.fail.emit(traceback.format_exc())
        finally:
            try:
                if self._audio:
                    self._audio.stop()
            except Exception:
                pass
            self._audio = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Notecryption Integrated")

        # Views
        self.tab = TabView()
        self.score = ScoreView()
        self.score.hide()  # show tab by default

        # ---- device selector row ----
        self.mic_combo = QComboBox()
        btn_refresh_mics = QPushButton("Refresh Mics")
        btn_refresh_mics.clicked.connect(self.refresh_mics)

        mic_row = QHBoxLayout()
        mic_row.addWidget(QLabel("Input device:"))
        mic_row.addWidget(self.mic_combo, 1)
        mic_row.addWidget(btn_refresh_mics)
        self.refresh_mics()

        # Top buttons
        btn_choose = QPushButton("Choose Audio")
        btn_transcribe = QPushButton("Transcribe")
        btn_live_start = QPushButton("Start Live")
        btn_live_stop = QPushButton("Stop Live")

        btn_choose.clicked.connect(self.choose_audio)
        btn_transcribe.clicked.connect(self.transcribe)
        btn_live_start.clicked.connect(self.start_live)
        btn_live_stop.clicked.connect(self.stop_live)

        top_row = QHBoxLayout()
        top_row.addWidget(btn_choose)
        top_row.addWidget(btn_transcribe)
        top_row.addWidget(btn_live_start)
        top_row.addWidget(btn_live_stop)

        # View toggle buttons
        btn_show_score = QPushButton("Show Score")
        btn_show_tab = QPushButton("Show Tab")
        btn_show_score.clicked.connect(self.show_score_view)
        btn_show_tab.clicked.connect(self.show_tab_view)

        view_row = QHBoxLayout()
        view_row.addWidget(btn_show_score)
        view_row.addWidget(btn_show_tab)
        view_row.addStretch(1)

        # Notes player UI
        self.note_player = QMediaPlayer(self)
        self.note_audio_out = QAudioOutput(self)
        self.note_player.setAudioOutput(self.note_audio_out)

        btn_notes_play = QPushButton("Play Notes")
        btn_notes_pause = QPushButton("Pause")
        btn_notes_stop = QPushButton("Stop")

        btn_notes_play.clicked.connect(self.play_notes)
        btn_notes_pause.clicked.connect(self.note_player.pause)
        btn_notes_stop.clicked.connect(lambda: self.stop_notes(cleanup=True))

        self.notes_seek = QSlider(Qt.Orientation.Horizontal)
        self.notes_seek.setMinimum(0)
        self.notes_seek.setMaximum(0)
        self.notes_seek.sliderMoved.connect(self.on_notes_seek)

        self.notes_time = QLabel("0:00 / 0:00")

        self.notes_vol = QSlider(Qt.Orientation.Horizontal)
        self.notes_vol.setRange(0, 100)
        self.notes_vol.setValue(70)
        self.notes_vol.valueChanged.connect(self.on_notes_volume)
        self.note_audio_out.setVolume(self.notes_vol.value() / 100.0)

        player_row = QHBoxLayout()
        player_row.addWidget(btn_notes_play)
        player_row.addWidget(btn_notes_pause)
        player_row.addWidget(btn_notes_stop)
        player_row.addWidget(self.notes_time)
        player_row.addWidget(QLabel("Vol"))
        player_row.addWidget(self.notes_vol)

        root = QVBoxLayout()
        root.addLayout(mic_row)
        root.addLayout(top_row)
        root.addLayout(view_row)
        root.addLayout(player_row)
        root.addWidget(self.notes_seek)
        root.addWidget(self.score)
        root.addWidget(self.tab)

        w = QWidget()
        w.setLayout(root)
        self.setCentralWidget(w)

        self.note_player.durationChanged.connect(self.on_notes_duration)
        self.note_player.positionChanged.connect(self.on_notes_position)

        self.audio_path = None
        self._last_notes: list[NoteEvent] = []

        self._live_worker: LiveWorker | None = None
        self._live_notes: list[NoteEvent] = []
        self._live_start_time = None

        self._notes_wav_path = None

        # Output folder for pipeline artifacts (notes.json, out.musicxml)
        self._out_dir = str(Path("data/output").resolve())

    # ---------- mic list ----------

    def refresh_mics(self):
        """
        Clean mic list:
        - keep Default input first
        - filter out Windows junk endpoints (mapper/primary/loopback/stereo mix)
        - dedupe by visible name (keep best host API)
        """
        self.mic_combo.clear()
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            hostapis = sd.query_hostapis()

            def hostapi_name(d: dict) -> str:
                try:
                    return str(hostapis[int(d.get("hostapi", 0))]["name"])
                except Exception:
                    return ""

            def hostapi_rank(h: str) -> int:
                hl = (h or "").lower()
                if "wasapi" in hl:
                    return 0
                if "wdm-ks" in hl or "wdm" in hl:
                    return 1
                if "directsound" in hl:
                    return 2
                if "mme" in hl:
                    return 3
                return 4

            def is_useful_input(d: dict) -> bool:
                if int(d.get("max_input_channels", 0)) <= 0:
                    return False

                name = str(d.get("name", "")).strip()
                low = name.lower()

                bad_tokens = [
                    "microsoft sound mapper",
                    "primary sound",
                    "mapper",
                    "stereo mix",
                    "loopback",
                ]
                if any(t in low for t in bad_tokens):
                    return False

                return True

            # Default option
            default_in = sd.default.device[0]
            self.mic_combo.addItem(f"Default input (index {default_in})", None)

            # Collect candidates
            candidates: list[tuple[int, dict, str]] = []
            for i, d in enumerate(devices):
                if is_useful_input(d):
                    h = hostapi_name(d)
                    candidates.append((i, d, h))

            # Dedupe: keep best hostapi for same name
            best_by_name: dict[str, tuple[int, dict, str]] = {}
            for i, d, h in candidates:
                name = str(d.get("name", f"Device {i}")).strip()
                key = name.lower()
                cur = best_by_name.get(key)
                if cur is None:
                    best_by_name[key] = (i, d, h)
                else:
                    _, _, h2 = cur
                    if hostapi_rank(h) < hostapi_rank(h2):
                        best_by_name[key] = (i, d, h)

            final = list(best_by_name.values())
            final.sort(key=lambda t: (hostapi_rank(t[2]), str(t[1].get("name", "")).lower()))

            for i, d, h in final:
                name = str(d.get("name", f"Device {i}")).strip()
                sr = int(float(d.get("default_samplerate", 0) or 0))
                ch = int(d.get("max_input_channels", 0))
                self.mic_combo.addItem(f"{name}  |  {h}  |  ch={ch} sr={sr}  (idx {i})", i)

            self.mic_combo.setCurrentIndex(0)

        except Exception as e:
            self.mic_combo.addItem(f"(sounddevice error: {e})", None)

    def _selected_device_index(self) -> int | None:
        return self.mic_combo.currentData()

    # ---------- lifecycle ----------
    def closeEvent(self, event):
        try:
            self.stop_live()
            if self._live_worker:
                self._live_worker.wait(1500)
        except Exception:
            pass
        super().closeEvent(event)

    # ---------- file transcription ----------
    def choose_audio(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select audio file", "", "Audio Files (*.wav *.mp3)"
        )
        if fp:
            self.audio_path = fp

    def transcribe(self):
        if not self.audio_path:
            QMessageBox.warning(self, "Error", "Select audio first.")
            return

        Path(self._out_dir).mkdir(parents=True, exist_ok=True)

        self.worker = Worker(self.audio_path, self._out_dir)
        self.worker.done.connect(self.on_transcribe_done)
        self.worker.fail.connect(self.on_fail)
        self.worker.start()

    def on_transcribe_done(self, raw):
        notes: list[NoteEvent] = []
        for r in raw:
            notes.append(NoteEvent(
                pitch=int(r.get("pitch", 60)),
                onset_s=float(r["onset_s"]),
                dur_s=float(r["dur_s"]),
                string=int(r["string"]),
                fret=int(r["fret"]),
            ))
        self._last_notes = notes

        # Always update tab view (fallback/debug)
        self.tab.render_tab(notes)

        # Render engraved score (staff + TAB) via OSMD
        ok = self.render_score_view()
        if ok:
            self.show_score_view()
        else:
            self.show_tab_view()

    def render_score_view(self) -> bool:
        """Loads data/output/out.musicxml and renders it in ScoreView (OSMD)."""
        xml_path = Path(self._out_dir) / "out.musicxml"
        if not xml_path.exists():
            return False

        try:
            xml_text = xml_path.read_text(encoding="utf-8")
        except Exception:
            return False

        self.score.render_musicxml_text(xml_text)
        return True

    def show_score_view(self):
        self.tab.hide()
        self.score.show()

    def show_tab_view(self):
        self.score.hide()
        self.tab.show()

    # ---------- live ----------
    def start_live(self):
        if self._live_worker and self._live_worker.isRunning():
            QMessageBox.information(self, "Live", "Live is already running.")
            return

        self._live_notes = []
        self._live_start_time = None
        self.tab.render_tab(self._live_notes)
        self.show_tab_view()  # live uses tab grid (low latency)

        dev = self._selected_device_index()

        # Validate device index (if device list changed, fall back to default)
        try:
            if dev is not None:
                import sounddevice as sd
                sd.query_devices(int(dev))
        except Exception:
            dev = None

        self._live_worker = LiveWorker(dev)
        self._live_worker.new_notes.connect(self.on_live_notes)
        self._live_worker.fail.connect(self.on_fail)
        self._live_worker.start()

    def stop_live(self):
        if self._live_worker and self._live_worker.isRunning():
            self._live_worker.request_stop()

    def on_live_notes(self, raw):
        if self._live_start_time is None and raw:
            self._live_start_time = float(raw[0]["onset_s"])
        base = float(self._live_start_time or 0.0)

        for r in raw:
            rel = float(r["onset_s"]) - base
            self._live_notes.append(NoteEvent(
                pitch=int(r.get("pitch", 60)),
                onset_s=rel,
                dur_s=float(r["dur_s"]),
                string=int(r["string"]),
                fret=int(r["fret"]),
            ))

        self._live_notes.sort(key=lambda n: (n.onset_s, n.string))
        self.tab.render_tab(self._live_notes)

        # auto-scroll to most recent note
        if self._live_notes:
            last = self._live_notes[-1]
            col = int(round(float(last.onset_s) / float(self.tab.dt)))
            x = self.tab.left_pad + col * self.tab.col_w
            y = self.tab.top_pad + (max(1, min(6, int(last.string))) - 1) * self.tab.line_gap
            self.tab.centerOn(x, y)

    # ---------- notes player ----------
    def _current_notes(self) -> list[NoteEvent]:
        return self._live_notes if self._live_notes else self._last_notes

    def play_notes(self):
        notes = self._current_notes()
        if not notes:
            QMessageBox.information(self, "Notes", "No transcribed notes to play.")
            return

        self.stop_notes(cleanup=True)

        synth = [SynthNote(pitch=n.pitch, onset_s=n.onset_s, dur_s=n.dur_s) for n in notes]
        path = synth_notes_to_wav_path(synth, sr=44100, gain=0.25)
        if not path:
            QMessageBox.information(self, "Notes", "Could not synthesize notes.")
            return

        self._notes_wav_path = path
        url = QUrl.fromLocalFile(str(Path(path).resolve()))
        self.note_player.setSource(url)
        self.note_player.play()

    def stop_notes(self, cleanup: bool = False):
        self.note_player.stop()
        self.notes_seek.setValue(0)
        dur = int(self.note_player.duration())
        self.notes_time.setText(f"{_fmt_ms(0)} / {_fmt_ms(dur)}")

        if cleanup and self._notes_wav_path:
            try:
                os.remove(self._notes_wav_path)
            except Exception:
                pass
            self._notes_wav_path = None

    def on_notes_seek(self, pos: int):
        self.note_player.setPosition(int(pos))

    def on_notes_volume(self, v: int):
        self.note_audio_out.setVolume(max(0.0, min(1.0, v / 100.0)))

    def on_notes_duration(self, dur: int):
        self.notes_seek.setMaximum(max(0, int(dur)))
        self.notes_time.setText(f"{_fmt_ms(int(self.note_player.position()))} / {_fmt_ms(int(dur))}")

    def on_notes_position(self, pos: int):
        self.notes_seek.setValue(int(pos))
        dur = int(self.note_player.duration())
        self.notes_time.setText(f"{_fmt_ms(int(pos))} / {_fmt_ms(dur)}")
        # Drive the tab playhead cursor (ms -> seconds)
        self.tab.set_playhead(pos / 1000.0)

    # ---------- errors ----------
    def on_fail(self, msg: str):
        QMessageBox.critical(self, "Error", msg)