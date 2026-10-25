"""
Thin wrapper around the ASR engines: loads the model once, transcribes each
audio file, then (optionally) runs forced alignment (Japanese) so segment
timings are tighter for hardsubbing.

Two engines are supported:
- "whisperx": WhisperX's batched pipeline (VAD-chunked, fast on GPU).
- "faster-whisper": SYSTRAN/faster-whisper used directly, sequential
  decoding with previous-text context (slower, but often reads more
  naturally across sentence boundaries).

Alignment downloads a separate wav2vec2 model on first use and can be
skipped (align=False) to fall back to Whisper's own (slightly less precise)
segment timestamps, e.g. when offline or the alignment model host is
unreachable.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable, List, TypedDict

LogFn = Callable[[str], None]

ENGINE_WHISPERX = "whisperx"
ENGINE_FASTER_WHISPER = "faster-whisper"


class Segment(TypedDict):
    start: float
    end: float
    text: str


def _clean(raw_segments) -> List[Segment]:
    return [
        {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": str(seg["text"]).strip(),
        }
        for seg in raw_segments
        if str(seg.get("text", "")).strip()
    ]


class Transcriber:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "ja",
        align: bool = True,
        engine: str = ENGINE_WHISPERX,
        log: LogFn = print,
    ):
        self.log = log
        self.device = device
        self.language = language
        self.align = align
        self.engine = engine
        self._whisperx = None  # imported lazily so the GUI can start without torch loaded

        label = "faster-whisper" if engine == ENGINE_FASTER_WHISPER else "WhisperX"
        self.log(f"Loading {label} model '{model_size}' on {device} ({compute_type})...")
        try:
            self.model = self._load_model(model_size, device, compute_type)
        except ValueError as e:
            # float16 isn't supported on CPU - fall back gracefully
            if compute_type == "float16":
                self.log("float16 not supported on this device, retrying with float32...")
                self.model = self._load_model(model_size, device, "float32")
            else:
                raise e

        self._align_model = None
        self._align_metadata = None

    def _load_model(self, model_size: str, device: str, compute_type: str):
        if self.engine == ENGINE_FASTER_WHISPER:
            from faster_whisper import WhisperModel

            return WhisperModel(model_size, device=device, compute_type=compute_type)

        whisperx = self._get_whisperx()
        return whisperx.load_model(model_size, device, compute_type=compute_type, language=self.language)

    def _get_whisperx(self):
        if self._whisperx is None:
            import whisperx

            self._whisperx = whisperx
        return self._whisperx

    def _ensure_align_model(self):
        if self._align_model is None:
            self.log("Loading Japanese alignment model...")
            self._align_model, self._align_metadata = self._get_whisperx().load_align_model(
                language_code=self.language, device=self.device
            )

    def transcribe_file(self, audio_path: Path, batch_size: int = 16) -> List[Segment]:
        self.log(f"Transcribing: {audio_path.name}")
        audio = None
        if self.engine == ENGINE_FASTER_WHISPER:
            raw_segments = self._transcribe_faster_whisper(audio_path)
        else:
            whisperx = self._get_whisperx()
            audio = whisperx.load_audio(str(audio_path))
            result = self.model.transcribe(audio, batch_size=batch_size, language=self.language)
            raw_segments = result.get("segments") or []

        if not raw_segments:
            self.log(f"  (no speech detected in {audio_path.name})")
            return []

        if not self.align:
            segments = _clean(raw_segments)
            self.log(f"  -> {len(segments)} subtitle lines (unaligned, using Whisper's own timestamps)")
            return segments

        whisperx = self._get_whisperx()
        if audio is None:
            audio = whisperx.load_audio(str(audio_path))
        self._ensure_align_model()
        aligned = whisperx.align(
            raw_segments,
            self._align_model,
            self._align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        segments = _clean(aligned["segments"])
        self.log(f"  -> {len(segments)} subtitle lines")
        return segments

    def _transcribe_faster_whisper(self, audio_path: Path) -> List[dict]:
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )
        # The iterator decodes lazily, so log as we go to show progress on long files
        raw = []
        last_pct = -10
        for seg in segments_iter:
            raw.append({"start": seg.start, "end": seg.end, "text": seg.text})
            if info.duration:
                pct = int(seg.end * 100 / info.duration)
                if pct >= last_pct + 10:
                    last_pct = pct - pct % 10
                    self.log(f"  {pct}% ({seg.end:.0f}s / {info.duration:.0f}s)")
        return raw

    def unload(self):
        del self.model
        if self._align_model is not None:
            del self._align_model
        gc.collect()
        try:
            import torch

            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
