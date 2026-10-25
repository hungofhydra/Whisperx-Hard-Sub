"""
Builds an .ass subtitle file (needed rather than plain .srt because we want
a styled semi-transparent background box behind the text) from a list of
per-file segment lists, applying a cumulative time offset per file so
everything lines up after the audio files are concatenated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .transcribe import Segment
from .utils import format_ass_time

ASS_HEADER = """[Script Info]
Title: Auto-generated hardsub
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601
PlayResX: {res_x}
PlayResY: {res_y}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,{back_colour},0,0,0,0,100,100,0,0,3,1,0,2,{margin_l},{margin_r},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


@dataclass
class SubtitleStyle:
    font: str = "Noto Sans JP"
    font_size: int = 42
    margin_l: int = 40
    margin_r: int = 40
    margin_v: int = 40
    # ASS colours are &HAABBGGRR. Alpha 00 = opaque, FF = fully transparent.
    box_alpha_hex: str = "40"  # ~75% opaque black box behind the text
    res_x: int = 1280
    res_y: int = 720


def _escape_ass_text(text: str) -> str:
    text = text.replace("\\", "\\\\").replace("{", "").replace("}", "")
    text = text.replace("\n", "\\N")
    return text


_BREAK_CHARS = "。、！？…」』】、,.!?"


def _wrap_line(text: str, max_chars: int) -> List[str]:
    """Split text into lines no longer than max_chars.

    ASS/libass wraps automatically at spaces, but Japanese text has no
    spaces between words - so without this, long CJK lines run straight
    off both edges of the frame instead of wrapping. This does simple
    character-count wrapping (CJK full-width glyphs are roughly as wide
    as they are tall, so char count is a decent proxy for pixel width),
    preferring to break right after punctuation near the limit so it
    doesn't cut mid-word when avoidable.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    lines = []
    remaining = text
    while len(remaining) > max_chars:
        cut = max_chars
        for i in range(max_chars, max(0, max_chars - 8), -1):
            if remaining[i - 1] in _BREAK_CHARS:
                cut = i
                break
        lines.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        lines.append(remaining)
    return lines


def build_ass(
    file_segments: List[List[Segment]],
    file_offsets: List[float],
    style: SubtitleStyle,
) -> str:
    """file_segments[i] are the aligned segments for input file i (in its
    own local time base); file_offsets[i] is how far into the merged
    timeline that file starts."""
    back_colour = f"&H{style.box_alpha_hex}000000"
    header = ASS_HEADER.format(
        res_x=style.res_x,
        res_y=style.res_y,
        font=style.font,
        fontsize=style.font_size,
        back_colour=back_colour,
        margin_l=style.margin_l,
        margin_r=style.margin_r,
        margin_v=style.margin_v,
    )

    lines = [header]
    # Conservative estimate: full-width CJK glyphs are roughly as wide as
    # they are tall, so treat each character as ~font_size px wide when
    # deciding where the text would overflow the frame.
    max_width_px = max(50, style.res_x - style.margin_l - style.margin_r)
    max_chars_per_line = max(4, int(max_width_px / style.font_size))

    for segments, offset in zip(file_segments, file_offsets):
        for seg in segments:
            start = format_ass_time(seg["start"] + offset)
            end = format_ass_time(seg["end"] + offset)
            raw_text = seg["text"].strip()
            if not raw_text:
                continue
            wrapped = _wrap_line(raw_text, max_chars_per_line)
            text = "\\N".join(_escape_ass_text(line) for line in wrapped)
            lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    return "\n".join(lines) + "\n"


def write_ass(path: Path, file_segments: List[List[Segment]], file_offsets: List[float], style: SubtitleStyle) -> None:
    content = build_ass(file_segments, file_offsets, style)
    path.write_text(content, encoding="utf-8-sig")  # BOM helps libass on Windows detect UTF-8


def _format_srt_time(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        ms = 0
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(
    file_segments: List[List[Segment]],
    file_offsets: List[float],
    chapter_titles: Optional[List[str]] = None,
    max_chars_per_line: int = 24,
) -> str:
    """Plain .srt with no hardsub styling. Since .srt has no native chapter
    metadata, each chapter boundary is inserted as its own short cue
    (filename as text) so chapter breaks are still visible when reading
    the subtitle file or in players that show it as regular dialogue."""
    entries = []  # (start, end, text)

    if chapter_titles:
        for title, offset in zip(chapter_titles, file_offsets):
            entries.append((offset, offset + 2.0, f"\u2014 {title} \u2014"))

    for segments, offset in zip(file_segments, file_offsets):
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            entries.append((seg["start"] + offset, seg["end"] + offset, text))

    entries.sort(key=lambda e: e[0])

    lines = []
    for i, (start, end, text) in enumerate(entries, 1):
        wrapped = "\n".join(_wrap_line(text, max_chars_per_line))
        lines.append(str(i))
        lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
        lines.append(wrapped)
        lines.append("")
    return "\n".join(lines) + "\n"


def write_srt(
    path: Path,
    file_segments: List[List[Segment]],
    file_offsets: List[float],
    chapter_titles: Optional[List[str]] = None,
) -> None:
    content = build_srt(file_segments, file_offsets, chapter_titles)
    path.write_text(content, encoding="utf-8-sig")
