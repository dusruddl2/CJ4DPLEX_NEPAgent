"""Cut a video at t_split into prefix.mp4 (agent input) and future.mp4 (evaluation only).

Source: NEP [1], prefix/future task definition.
Changed from source: re-encodes for frame-accurate cuts; future.mp4 is optional. `observed_clip` is the single-video entry:
it never writes the future part and accepts a clip that already ends at the split point.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from src.utils.media import ffmpeg_binary, probe_duration

logger = logging.getLogger(__name__)

FFMPEG_TIMEOUT_SEC = 600


def _cut(source: Path, target: Path, start: float | None, end: float | None) -> None:
    command = [ffmpeg_binary(), "-y", "-v", "error", "-i", str(source)]
    if start is not None:
        command += ["-ss", f"{start:.3f}"]
    if end is not None:
        command += ["-to", f"{end:.3f}"]
    command += ["-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-movflags", "+faststart", str(target)]
    subprocess.run(command, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SEC, check=True)


def split_video(
    source: str | Path, t_split: float, output_dir: str | Path, write_future: bool = True
) -> tuple[Path, Path | None]:
    """Write prefix.mp4 (0..t_split) and optionally future.mp4 (t_split..end)."""
    source_path = Path(source)
    if t_split <= 0:
        raise ValueError(f"t_split must be positive, got {t_split}")
    duration = probe_duration(source_path)
    if t_split >= duration:
        raise ValueError(f"t_split {t_split} is not before the video end {duration:.2f}")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    prefix = target / "prefix.mp4"
    _cut(source_path, prefix, None, t_split)
    future: Path | None = None
    if write_future:
        future = target / "future.mp4"
        _cut(source_path, future, t_split, None)
    return prefix, future


def observed_clip(source: str | Path, t_split: float | None, output_dir: str | Path) -> Path:
    """The clip the agent may see. Without a split point, or with one at or beyond the end of the file, the whole file is
    the observed clip and nothing is written; otherwise only prefix.mp4 (0..t_split) is cut."""
    source_path = Path(source)
    if t_split is not None and t_split <= 0:
        raise ValueError(f"t_split must be positive, got {t_split}")
    if t_split is None or t_split >= probe_duration(source_path):
        return source_path
    prefix, _ = split_video(source_path, t_split, output_dir, write_future=False)
    return prefix
