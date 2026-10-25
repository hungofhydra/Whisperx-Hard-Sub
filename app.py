"""
WhisperX Hardsub Studio
========================
GUI application: pick one or more Japanese audio files + a background
image, transcribe with WhisperX, burn styled hardsubs onto a static-image
video, merge multi-file inputs into chapters (named after each filename),
and export a small mp4.

Run with:  python app.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QFileDialog,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QPlainTextEdit,
    QProgressBar,
    QMessageBox,
    QSplitter,
)

from core.pipeline import JobConfig, Pipeline, PipelineCancelled, RenderSettings
from core.subtitles import SubtitleStyle
from core.transcribe import ENGINE_FASTER_WHISPER, ENGINE_WHISPERX

MEDIA_EXTS = (
    "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus *.wma *.alac *.aiff *.aif "
    "*.mp4 *.m4v *.mkv *.mov *.avi *.webm *.wmv *.flv *.ts *.mts"
)
# Every input goes through ffmpeg, so anything it can decode works; "All files" covers the rest
AUDIO_FILTER = f"Audio/video files ({MEDIA_EXTS});;All files (*)"
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp)"


class PipelineWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int, str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, config: JobConfig):
        super().__init__()
        self.config = config
        self._pipeline: Optional[Pipeline] = None

    def run(self):
        try:
            self._pipeline = Pipeline(
                self.config,
                log=lambda s: self.log.emit(s),
                progress=lambda cur, total, label: self.progress.emit(cur, total, label),
            )
            out = self._pipeline.run()
            self.finished.emit(str(out))
        except PipelineCancelled:
            self.log.emit("Cancelled.")
            self.failed.emit("Cancelled by user.")
        except Exception as e:  # noqa: BLE001
            self.log.emit(traceback.format_exc())
            self.failed.emit(str(e))

    def cancel(self):
        if self._pipeline:
            self._pipeline.cancel()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WhisperX Hardsub Studio")
        self.resize(980, 720)

        self.audio_files: List[Path] = []
        self.thread: Optional[QThread] = None
        self.worker: Optional[PipelineWorker] = None

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QSplitter(Qt.Vertical)
        self.setCentralWidget(root)

        top = QWidget()
        top_layout = QHBoxLayout(top)

        top_layout.addWidget(self._build_files_group(), 2)
        top_layout.addWidget(self._build_settings_group(), 3)

        root.addWidget(top)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Idle")
        bottom_layout.addWidget(self.progress_label)
        bottom_layout.addWidget(self.progress_bar)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        bottom_layout.addWidget(self.log_view, 1)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.on_start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.on_cancel)
        btn_row.addStretch(1)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.start_btn)
        bottom_layout.addLayout(btn_row)

        root.addWidget(bottom)
        root.setStretchFactor(0, 3)
        root.setStretchFactor(1, 2)

    def _build_files_group(self) -> QGroupBox:
        box = QGroupBox("1. Audio/video files (same format, in play order)")
        layout = QVBoxLayout(box)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.file_list, 1)

        row = QHBoxLayout()
        add_btn = QPushButton("Add files…")
        add_btn.clicked.connect(self.on_add_files)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.on_remove_files)
        up_btn = QPushButton("Move up")
        up_btn.clicked.connect(lambda: self.on_move(-1))
        down_btn = QPushButton("Move down")
        down_btn.clicked.connect(lambda: self.on_move(1))
        for b in (add_btn, remove_btn, up_btn, down_btn):
            row.addWidget(b)
        layout.addLayout(row)

        note = QLabel(
            "Each file's name (without extension) becomes a chapter title "
            "in the final output, in the order shown above."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        layout.addWidget(note)

        layout.addWidget(QLabel("Output type:"))
        self.output_type_combo = QComboBox()
        self.output_type_combo.addItems([
            "Video with hardsub (.mp4)",
            "Subtitle file only (.srt)",
            "Merged video, no subtitles - skip transcription (.mp4)",
        ])
        self.output_type_combo.currentIndexChanged.connect(self.on_output_type_changed)
        layout.addWidget(self.output_type_combo)

        self.image_label = QLabel("Background image (shown for the whole video):")
        layout.addWidget(self.image_label)
        img_row = QHBoxLayout()
        self.image_edit = QLineEdit()
        self.image_edit.setReadOnly(True)
        self.image_browse_btn = QPushButton("Browse…")
        self.image_browse_btn.clicked.connect(self.on_pick_image)
        img_row.addWidget(self.image_edit, 1)
        img_row.addWidget(self.image_browse_btn)
        layout.addLayout(img_row)

        self.image_preview = QLabel("(no image selected)")
        self.image_preview.setFixedHeight(140)
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setStyleSheet("border: 1px solid #999; color: #888;")
        layout.addWidget(self.image_preview)

        self.output_label = QLabel("Output video (.mp4):")
        layout.addWidget(self.output_label)
        out_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self.on_pick_output)
        out_row.addWidget(self.output_edit, 1)
        out_row.addWidget(out_browse)
        layout.addLayout(out_row)

        return box

    def _build_settings_group(self) -> QGroupBox:
        box = QGroupBox("2. Settings")
        layout = QVBoxLayout(box)

        # --- Transcription group ---
        self.whisper_box = QGroupBox("Transcription (Japanese)")
        whisper_box = self.whisper_box
        g = QGridLayout(whisper_box)
        g.addWidget(QLabel("Engine:"), 0, 0)
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("WhisperX (batched, fast)", ENGINE_WHISPERX)
        self.engine_combo.addItem("faster-whisper (sequential)", ENGINE_FASTER_WHISPER)
        self.engine_combo.setToolTip(
            "WhisperX: batched VAD chunks, much faster on GPU.\n"
            "faster-whisper (SYSTRAN): decodes the audio in order using the\n"
            "previous text as context. Slower, but lines often read more naturally.\n"
            "Both use the same downloaded models."
        )
        self.engine_combo.currentIndexChanged.connect(self.on_engine_changed)
        g.addWidget(self.engine_combo, 0, 1)

        g.addWidget(QLabel("Model size:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["large-v3", "large-v3-turbo", "large-v2", "medium", "small"])
        self.model_combo.setCurrentText("medium")
        g.addWidget(self.model_combo, 1, 1)

        g.addWidget(QLabel("Device:"), 2, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        self.device_combo.setCurrentText("cpu")
        g.addWidget(self.device_combo, 2, 1)

        g.addWidget(QLabel("Compute type:"), 3, 0)
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["float16", "int8", "float32"])
        self.compute_combo.setCurrentText("float32")
        g.addWidget(self.compute_combo, 3, 1)

        self.batch_label = QLabel("Batch size:")
        g.addWidget(self.batch_label, 4, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(16)
        g.addWidget(self.batch_spin, 4, 1)

        self.align_check = QCheckBox("Precise alignment (downloads a wav2vec2 model, needs internet)")
        self.align_check.setChecked(False)
        self.align_check.setToolTip(
            "Re-times each subtitle line against the audio for tighter sync.\n"
            "Downloads ~1GB the first time from Hugging Face.\n"
            "Uncheck to use Whisper's own (slightly less precise) timestamps\n"
            "instead — useful if you're offline or the download keeps failing."
        )
        g.addWidget(self.align_check, 5, 0, 1, 2)
        layout.addWidget(whisper_box)

        # --- Subtitle style ---
        self.sub_box = QGroupBox("Subtitle style (styled box)")
        sub_box = self.sub_box
        g2 = QGridLayout(sub_box)
        g2.addWidget(QLabel("Font:"), 0, 0)
        self.font_edit = QLineEdit("Noto Sans JP")
        g2.addWidget(self.font_edit, 0, 1)

        g2.addWidget(QLabel("Font size:"), 1, 0)
        self.fontsize_spin = QSpinBox()
        self.fontsize_spin.setRange(10, 120)
        self.fontsize_spin.setValue(42)
        g2.addWidget(self.fontsize_spin, 1, 1)

        g2.addWidget(QLabel("Box opacity (0=solid, 100=transparent):"), 2, 0)
        self.box_opacity_spin = QSpinBox()
        self.box_opacity_spin.setRange(0, 100)
        self.box_opacity_spin.setValue(25)
        g2.addWidget(self.box_opacity_spin, 2, 1)
        layout.addWidget(sub_box)

        # --- Render / size settings ---
        self.render_box = QGroupBox("Video output (kept small)")
        render_box = self.render_box
        g3 = QGridLayout(render_box)
        g3.addWidget(QLabel("Resolution:"), 0, 0)
        self.res_combo = QComboBox()
        self.res_combo.addItems(["1280x720", "854x480", "640x360", "1920x1080"])
        g3.addWidget(self.res_combo, 0, 1)

        g3.addWidget(QLabel("Frame rate (fps):"), 1, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(5)
        g3.addWidget(self.fps_spin, 1, 1)

        g3.addWidget(QLabel("Video CRF (higher = smaller/lower quality):"), 2, 0)
        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(18, 40)
        self.crf_spin.setValue(30)
        g3.addWidget(self.crf_spin, 2, 1)

        g3.addWidget(QLabel("Encode preset:"), 3, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["veryslow", "slower", "slow", "medium", "fast"])
        g3.addWidget(self.preset_combo, 3, 1)

        g3.addWidget(QLabel("Audio bitrate (kbps):"), 4, 0)
        self.abitrate_spin = QSpinBox()
        self.abitrate_spin.setRange(16, 192)
        self.abitrate_spin.setValue(40)
        g3.addWidget(self.abitrate_spin, 4, 1)

        self.mono_check = QCheckBox("Mono audio (smaller file)")
        self.mono_check.setChecked(True)
        g3.addWidget(self.mono_check, 5, 0, 1, 2)

        self.normalize_check = QCheckBox("Boost/normalize volume (recommended for quiet source audio)")
        self.normalize_check.setChecked(True)
        g3.addWidget(self.normalize_check, 6, 0, 1, 2)

        layout.addWidget(render_box)
        layout.addStretch(1)
        return box

    # ------------------------------------------------------------------
    def on_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select audio files", "", AUDIO_FILTER)
        if not files:
            return
        for f in files:
            p = Path(f)
            self.audio_files.append(p)
            self.file_list.addItem(QListWidgetItem(p.name))
        self._validate_same_extension()

    def on_remove_files(self):
        rows = sorted((i.row() for i in self.file_list.selectedIndexes()), reverse=True)
        for r in rows:
            self.file_list.takeItem(r)
            del self.audio_files[r]

    def on_move(self, delta: int):
        row = self.file_list.currentRow()
        new_row = row + delta
        if row < 0 or not (0 <= new_row < self.file_list.count()):
            return
        item = self.file_list.takeItem(row)
        self.file_list.insertItem(new_row, item)
        self.file_list.setCurrentRow(new_row)
        self.audio_files[row], self.audio_files[new_row] = self.audio_files[new_row], self.audio_files[row]

    def _validate_same_extension(self):
        exts = {p.suffix.lower() for p in self.audio_files}
        if len(exts) > 1:
            QMessageBox.warning(
                self,
                "Mixed formats",
                f"You added files with different extensions ({sorted(exts)}). "
                "All input audio files must share the same format.",
            )

    def is_srt_only(self) -> bool:
        return self.output_type_combo.currentIndex() == 1

    def is_merge_only(self) -> bool:
        return self.output_type_combo.currentIndex() == 2

    def on_engine_changed(self):
        # faster-whisper decodes sequentially, so batch size doesn't apply
        batched = self.engine_combo.currentData() == ENGINE_WHISPERX
        self.batch_label.setEnabled(batched)
        self.batch_spin.setEnabled(batched)

    def on_output_type_changed(self):
        srt_only = self.is_srt_only()
        merge_only = self.is_merge_only()
        self.image_label.setEnabled(not srt_only)
        self.image_edit.setEnabled(not srt_only)
        self.image_browse_btn.setEnabled(not srt_only)
        self.image_preview.setEnabled(not srt_only)
        self.whisper_box.setEnabled(not merge_only)
        self.sub_box.setEnabled(not srt_only and not merge_only)
        self.render_box.setEnabled(not srt_only)
        self.output_label.setText("Output subtitle (.srt):" if srt_only else "Output video (.mp4):")
        # Clear a stale path from the other mode so the extension prompt below stays accurate
        current = self.output_edit.text()
        if current:
            if srt_only and current.lower().endswith(".mp4"):
                self.output_edit.setText(str(Path(current).with_suffix(".srt")))
            elif not srt_only and current.lower().endswith(".srt"):
                self.output_edit.setText(str(Path(current).with_suffix(".mp4")))

    def on_pick_image(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select background image", "", IMAGE_FILTER)
        if not f:
            return
        self.image_edit.setText(f)
        pix = QPixmap(f)
        if not pix.isNull():
            self.image_preview.setPixmap(
                pix.scaled(self.image_preview.width(), self.image_preview.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def on_pick_output(self):
        if self.is_srt_only():
            f, _ = QFileDialog.getSaveFileName(self, "Save subtitle file as", "output.srt", "SubRip subtitle (*.srt)")
            if f:
                if not f.lower().endswith(".srt"):
                    f += ".srt"
                self.output_edit.setText(f)
        else:
            f, _ = QFileDialog.getSaveFileName(self, "Save output video as", "output.mp4", "MP4 video (*.mp4)")
            if f:
                if not f.lower().endswith(".mp4"):
                    f += ".mp4"
                self.output_edit.setText(f)

    # ------------------------------------------------------------------
    def on_start(self):
        srt_only = self.is_srt_only()
        if not self.audio_files:
            QMessageBox.warning(self, "Missing input", "Add at least one audio file.")
            return
        if not srt_only and not self.image_edit.text():
            QMessageBox.warning(self, "Missing input", "Pick a background image.")
            return
        if not self.output_edit.text():
            QMessageBox.warning(self, "Missing input", "Pick an output file path.")
            return

        exts = {p.suffix.lower() for p in self.audio_files}
        if len(exts) > 1:
            QMessageBox.critical(self, "Mixed formats", "All audio files must share the same format.")
            return

        style = SubtitleStyle(
            font=self.font_edit.text() or "Noto Sans JP",
            font_size=self.fontsize_spin.value(),
            box_alpha_hex=f"{int(self.box_opacity_spin.value() * 255 / 100):02X}",
        )
        render = RenderSettings(
            fps=self.fps_spin.value(),
            crf=self.crf_spin.value(),
            preset=self.preset_combo.currentText(),
            resolution=self.res_combo.currentText(),
            audio_bitrate_kbps=self.abitrate_spin.value(),
            audio_mono=self.mono_check.isChecked(),
            normalize_audio=self.normalize_check.isChecked(),
        )
        config = JobConfig(
            audio_files=list(self.audio_files),
            output_path=Path(self.output_edit.text()),
            background_image=Path(self.image_edit.text()) if self.image_edit.text() else None,
            srt_only=srt_only,
            skip_transcription=self.is_merge_only(),
            engine=self.engine_combo.currentData(),
            model_size=self.model_combo.currentText(),
            device=self.device_combo.currentText(),
            compute_type=self.compute_combo.currentText(),
            batch_size=self.batch_spin.value(),
            align=self.align_check.isChecked(),
            style=style,
            render=render,
        )

        self.log_view.clear()
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self.thread = QThread(self)
        self.worker = PipelineWorker(config)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self._on_log)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def on_cancel(self):
        if self.worker:
            self.worker.cancel()
        self.cancel_btn.setEnabled(False)

    def _on_log(self, text: str):
        self.log_view.appendPlainText(text)

    def _on_progress(self, current: int, total: int, label: str):
        self.progress_label.setText(f"{label} ({current}/{total})")
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(current)

    def _on_finished(self, output_path: str):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText("Done")
        QMessageBox.information(self, "Done", f"Created:\n{output_path}")

    def _on_failed(self, message: str):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText("Failed")
        QMessageBox.critical(self, "Failed", message)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("WhisperX Hardsub Studio")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
