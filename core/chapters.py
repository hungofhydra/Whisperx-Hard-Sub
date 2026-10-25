"""
Builds an ffmpeg FFMETADATA1 file so each source audio file becomes a
named chapter (its filename, without extension) embedded in the output
mp4 - visible in players like VLC / mpv / most phone video apps.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .utils import format_ffmetadata_time_ms


def build_ffmetadata(titles: List[str], starts: List[float], ends: List[float]) -> str:
    assert len(titles) == len(starts) == len(ends)
    out = [";FFMETADATA1"]
    for title, start, end in zip(titles, starts, ends):
        start_ms = format_ffmetadata_time_ms(start)
        end_ms = format_ffmetadata_time_ms(end)
        safe_title = title.replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#").replace("\n", " ")
        out.append("")
        out.append("[CHAPTER]")
        out.append("TIMEBASE=1/1000")
        out.append(f"START={start_ms}")
        out.append(f"END={end_ms}")
        out.append(f"title={safe_title}")
    return "\n".join(out) + "\n"


def write_ffmetadata(path: Path, titles: List[str], starts: List[float], ends: List[float]) -> None:
    path.write_text(build_ffmetadata(titles, starts, ends), encoding="utf-8")
