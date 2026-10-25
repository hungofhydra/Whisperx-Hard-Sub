from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .chapters import write_ffmetadata
from .subtitles import SubtitleStyle, write_ass, write_srt
from .transcribe import ENGINE_WHISPERX, Segment, Transcriber
from .utils import ffmpeg_filter_path, find_ffmpeg, probe_duration_seconds, run_ffmpeg

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, str], None]  # (current_step, total_steps, label)


@dataclass
class RenderSettings:
    # Encoding tuned for a mostly-static image + speech: low fps + high CRF
    # gives dramatic size savings since consecutive frames are near-identical.
    fps: int = 5
    crf: int = 30
    preset: str = "veryslow"
    resolution: Optional[str] = None  # e.g. "1280x720"; None = keep image's native size
    audio_bitrate_kbps: int = 40
    audio_mono: bool = True
    normalize_audio: bool = True


@dataclass
class JobConfig:
    audio_files: List[Path]
    output_path: Path
    background_image: Optional[Path] = None
    srt_only: bool = False
    skip_transcription: bool = False  # merge audio + chapters into a video without subtitles
    engine: str = ENGINE_WHISPERX
    model_size: str = "medium"
    device: str = "cpu"
    compute_type: str = "float32"
    batch_size: int = 16
    align: bool = False
    style: SubtitleStyle = field(default_factory=SubtitleStyle)
    render: RenderSettings = field(default_factory=RenderSettings)
    keep_temp: bool = False


class PipelineCancelled(Exception):
    pass


class Pipeline:
    def __init__(self, config: JobConfig, log: LogFn = print, progress: Optional[ProgressFn] = None):
        self.cfg = config
        self.log = log
        self.progress = progress or (lambda cur, total, label: None)
        self._cancelled = False
        self.ffmpeg = find_ffmpeg()
        self.log(f"Using ffmpeg: {self.ffmpeg}")

    def cancel(self):
        self._cancelled = True

    def _check_cancel(self):
        if self._cancelled:
            raise PipelineCancelled("Cancelled by user")

    # ------------------------------------------------------------------
    def run(self) -> Path:
        cfg = self.cfg
        if not cfg.audio_files:
            raise ValueError("No audio files provided")

        exts = {p.suffix.lower() for p in cfg.audio_files}
        if len(exts) > 1:
            raise ValueError(f"All audio files must share the same format, got: {sorted(exts)}")

        if not cfg.srt_only and cfg.background_image is None:
            raise ValueError("A background image is required unless extracting subtitles only")

        if cfg.srt_only and cfg.skip_transcription:
            raise ValueError("Subtitle-only output needs transcription")

        if cfg.srt_only:
            return self._run_srt_only()
        return self._run_video()

    def _probe_and_transcribe(self, total_steps: int, step: int):
        """Shared first phase for all modes: probe durations (for chapter
        offsets), then transcribe each file unless transcription is skipped
        (all_segments is then None). Returns
        (step, durations, offsets, ends, titles, all_segments)."""
        cfg = self.cfg

        self.progress(step, total_steps, "Probing audio durations")
        durations = []
        for p in cfg.audio_files:
            self._check_cancel()
            d = probe_duration_seconds(p, self.ffmpeg)
            durations.append(d)
            self.log(f"  {p.name}: {d:.2f}s")
        step += 1

        offsets = []
        acc = 0.0
        for d in durations:
            offsets.append(acc)
            acc += d
        ends = [o + d for o, d in zip(offsets, durations)]
        titles = [p.stem for p in cfg.audio_files]

        if cfg.skip_transcription:
            self.log("Skipping transcription (merge only)")
            return step, durations, offsets, ends, titles, None

        transcriber = Transcriber(
            model_size=cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
            language="ja",
            align=cfg.align,
            engine=cfg.engine,
            log=self.log,
        )
        all_segments: List[List[Segment]] = []
        try:
            for p in cfg.audio_files:
                self._check_cancel()
                self.progress(step, total_steps, f"Transcribing {p.name}")
                segs = transcriber.transcribe_file(p, batch_size=cfg.batch_size)
                all_segments.append(segs)
                step += 1
        finally:
            transcriber.unload()

        return step, durations, offsets, ends, titles, all_segments

    def _run_srt_only(self) -> Path:
        cfg = self.cfg
        total_steps = 2 + len(cfg.audio_files)  # probe, transcribe(n), write srt
        step = 0

        step, durations, offsets, ends, titles, all_segments = self._probe_and_transcribe(total_steps, step)

        self._check_cancel()
        self.progress(step, total_steps, "Writing subtitle file")
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        out_path = cfg.output_path
        if out_path.suffix.lower() != ".srt":
            out_path = out_path.with_suffix(".srt")
        write_srt(out_path, all_segments, offsets, chapter_titles=titles)
        step += 1

        self.progress(total_steps, total_steps, "Done")
        self.log(f"Finished: {out_path}")
        return out_path

    def _run_video(self) -> Path:
        cfg = self.cfg
        n_transcribe = 0 if cfg.skip_transcription else len(cfg.audio_files)
        total_steps = 4 + n_transcribe  # probe, transcribe(n), subs+chapters, concat, render
        step = 0

        tmp_dir = Path(tempfile.mkdtemp(prefix="whisperx_hardsub_"))
        self.log(f"Working directory: {tmp_dir}")

        try:
            step, durations, offsets, ends, titles, all_segments = self._probe_and_transcribe(total_steps, step)

            # Build subtitle (.ass) and chapters (.ffmeta) files
            self._check_cancel()
            self.progress(step, total_steps, "Building subtitles and chapters")
            res_x, res_y = self._resolve_resolution()
            ass_path: Optional[Path] = None
            if all_segments is not None:
                style = cfg.style
                style.res_x, style.res_y = res_x, res_y
                ass_path = tmp_dir / "subs.ass"
                write_ass(ass_path, all_segments, offsets, style)

            chapters_path = tmp_dir / "chapters.ffmeta"
            write_ffmetadata(chapters_path, titles, offsets, ends)
            step += 1

            # Normalize + concatenate audio into one track
            self._check_cancel()
            self.progress(step, total_steps, "Merging audio")
            merged_audio = tmp_dir / "merged.wav"
            self._concat_audio(cfg.audio_files, tmp_dir, merged_audio)
            step += 1

            # Render final hardsubbed video
            self._check_cancel()
            self.progress(step, total_steps, "Rendering final video")
            cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._render_video(merged_audio, ass_path, chapters_path, res_x, res_y, cfg.output_path)
            step += 1

            self.progress(total_steps, total_steps, "Done")
            self.log(f"Finished: {cfg.output_path} ({cfg.output_path.stat().st_size / 1_000_000:.2f} MB)")
            return cfg.output_path
        finally:
            if not cfg.keep_temp:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                self.log(f"Kept temp files at {tmp_dir}")

    # ------------------------------------------------------------------
    def _resolve_resolution(self) -> tuple[int, int]:
        r = self.cfg.render.resolution
        if r:
            w, h = r.lower().split("x")
            return int(w), int(h)
        # Probe the image's native size via ffmpeg banner parsing would need
        # another regex; simplest robust route is to let ffmpeg scale later
        # with -2 to preserve aspect ratio from a default width.
        return 1280, 720

    def _concat_audio(self, files: List[Path], tmp_dir: Path, out_path: Path) -> None:
        normalized = []
        for i, f in enumerate(files):
            self._check_cancel()
            norm = tmp_dir / f"norm_{i:03d}.wav"
            run_ffmpeg(
                ["-i", str(f), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(norm)],
                log=self.log,
                ffmpeg_path=self.ffmpeg,
            )
            normalized.append(norm)

        if len(normalized) == 1:
            shutil.copy(normalized[0], out_path)
            return

        list_path = tmp_dir / "concat_list.txt"
        with list_path.open("w", encoding="utf-8") as fh:
            for n in normalized:
                escaped = str(n.resolve()).replace("'", "'\\''")
                fh.write(f"file '{escaped}'\n")

        run_ffmpeg(
            ["-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(out_path)],
            log=self.log,
            ffmpeg_path=self.ffmpeg,
        )

    def _render_video(
        self,
        audio_path: Path,
        ass_path: Optional[Path],
        chapters_path: Path,
        res_x: int,
        res_y: int,
        output_path: Path,
    ) -> None:
        render = self.cfg.render
        image_path = self.cfg.background_image

        vf_parts = [f"scale={res_x}:{res_y}:force_original_aspect_ratio=decrease,pad={res_x}:{res_y}:(ow-iw)/2:(oh-ih)/2"]
        if ass_path is not None:
            vf_parts.append(f"ass='{ffmpeg_filter_path(ass_path)}'")
        vf = ",".join(vf_parts)

        audio_channels = "1" if render.audio_mono else "2"

        args = [
            "-loop", "1",
            "-framerate", str(render.fps),
            "-i", str(image_path),
            "-i", str(audio_path),
            "-i", str(chapters_path),
            "-map_metadata", "2",
            "-map", "0:v",
            "-map", "1:a",
            "-vf", vf,
        ]
        if render.normalize_audio:
            # dynaudnorm boosts quiet passages (common with softly-spoken
            # source audio) without needing to hand-tune a target level,
            # and runs in a single pass unlike loudnorm's two-pass mode.
            args += ["-af", "dynaudnorm=f=150:g=15"]
        args += [
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-preset", render.preset,
            "-crf", str(render.crf),
            "-r", str(render.fps),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", f"{render.audio_bitrate_kbps}k",
            "-ac", audio_channels,
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]
        run_ffmpeg(args, log=self.log, ffmpeg_path=self.ffmpeg)
