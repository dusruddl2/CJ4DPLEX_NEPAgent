"""Locate ffmpeg and read basic media facts from it, so that ffprobe is not required."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

PROBE_TIMEOUT_SEC = 60
_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_AUDIO_STREAM = re.compile(r"Stream #\S+.*Audio:")


def ffmpeg_binary() -> str:
    """Prefer a system ffmpeg; fall back to the binary bundled with imageio-ffmpeg."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _describe(video: str | Path) -> str:
    """Return ffmpeg's stream description, which it prints to stderr when given no output."""
    if not Path(video).exists():
        raise FileNotFoundError(video)
    result = subprocess.run(
        [ffmpeg_binary(), "-hide_banner", "-i", str(video)],
        capture_output=True, text=True, timeout=PROBE_TIMEOUT_SEC, check=False,
    )
    return result.stderr


def probe_duration(video: str | Path) -> float:
    """Return the container duration in seconds."""
    match = _DURATION.search(_describe(video))
    if match is None:
        raise ValueError(f"could not read the duration of {video}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def has_audio_stream(video: str | Path) -> bool:
    """Return True when the file has at least one audio stream."""
    return _AUDIO_STREAM.search(_describe(video)) is not None
