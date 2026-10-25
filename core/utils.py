"""
Low level helpers shared by the pipeline: locating ffmpeg, running it while
streaming progress back to the UI, probing media duration, and escaping
paths for use inside ffmpeg filter graphs (which is notoriously picky on
Windows because of drive letter colons and backslashes).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional

LogFn = Callable[[str], None]

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2})\.(\d{2})")


class FFmpegNotFoundError(RuntimeError):
    pass


class FFmpegError(RuntimeError):
    pass


def find_ffmpeg() -> str:
    """Resolve an ffmpeg executable path.

    Order of preference:
    1. ffmpeg already on PATH (best: user controls the build/version)
    2. bundled binary shipped by the `imageio-ffmpeg` package (fallback so
       the app works even if the user never installed ffmpeg manually)
    """
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path

    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    raise FFmpegNotFoundError(
        "Could not find an ffmpeg executable. Install ffmpeg and make sure "
        "it's on your PATH, or `pip install imageio-ffmpeg`."
    )


def run_ffmpeg(args: List[str], log: LogFn = print, ffmpeg_path: Optional[str] = None) -> None:
    """Run ffmpeg with the given args (excluding the 'ffmpeg' executable
    itself), streaming stderr lines to `log`. Raises FFmpegError on non-zero
    exit."""
    exe = ffmpeg_path or find_ffmpeg()
    cmd = [exe, "-hide_banner", "-y", *args]
    log("$ " + " ".join(_quote(a) for a in cmd))

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    assert proc.stdout is not None
    lines: List[str] = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        if line.strip():
            log(line)
    proc.wait()
    if proc.returncode != 0:
        tail = "\n".join(lines[-30:])
        raise FFmpegError(f"ffmpeg exited with code {proc.returncode}:\n{tail}")


def probe_duration_seconds(path: Path, ffmpeg_path: Optional[str] = None) -> float:
    """Get media duration in seconds by parsing ffmpeg's stderr banner.
    Avoids requiring ffprobe to be installed separately."""
    exe = ffmpeg_path or find_ffmpeg()
    cmd = [exe, "-hide_banner", "-i", str(path)]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    m = _DURATION_RE.search(proc.stdout)
    if not m:
        raise FFmpegError(f"Could not determine duration of {path}:\n{proc.stdout[-800:]}")
    h, mi, s, cs = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s + cs / 100.0


def _quote(arg: str) -> str:
    if " " in arg or "" == arg:
        return f'"{arg}"'
    return arg


def ffmpeg_filter_path(path: Path) -> str:
    """Escape a filesystem path for embedding inside an ffmpeg -vf filter
    argument (e.g. subtitles=..., ass=...). ffmpeg filter syntax treats
    ':', '\\', ''' and '[' ']' specially, and Windows drive-letter colons
    make this extra annoying."""
    p = str(path.resolve())
    p = p.replace("\\", "/")
    p = p.replace(":", "\\:")
    p = p.replace("'", "\\'")
    return p


def format_ass_time(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs == 100:
        cs = 0
        s += 1
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def format_ffmetadata_time_ms(t: float) -> int:
    return int(round(t * 1000))
