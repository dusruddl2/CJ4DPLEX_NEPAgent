"""Observer: sample frames from the observed clip and label them with fixed ids (frame-number visual prompts, after ViKey).

The frame id is the index that every later citation refers to; ids never change.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import logging
import os
import numpy as np
from PIL import Image

from src.agent import frame_ids
from src.agent.vp import render_number
from src.schemas import STATUS_VIDEO_ERROR, AgentInput, RunConfig, SampleState

# decord gives up on the last frames of some clips on slow NFS ("takes too long to retrieve last few frames");
# its default EOF retry budget (1024) is far too small there. Set before decord is imported.
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "40960")
logger = logging.getLogger(__name__)
MAX_STORED_SIDE = 720
MIN_FRAMES_FLAG = 3


@dataclass(frozen=True)
class FrameIO:
    cache_root: Path

    def raw_path(self, sample_id: str, milliseconds: int) -> Path:
        return self.cache_root / sample_id / "raw" / f"{milliseconds:08d}.jpg"

    def labeled_path(self, sample_id: str, frame_id: str, milliseconds: int) -> Path:
        from src.agent.vp import VP_STYLE
        folder = "vp" if VP_STYLE == "number" else f"vp_{VP_STYLE}"
        return self.cache_root / sample_id / folder / f"{frame_ids.file_stem(frame_id)}_{milliseconds:08d}.jpg"


def uniform_indices(total_frames: int, count: int) -> list[int]:
    """Evenly spaced decoded-frame indices, always including the last observed frame."""
    if total_frames <= 0:
        raise ValueError("video has no frames")
    return sorted({int(round(i)) for i in np.linspace(0, total_frames - 1, count)})


DECODE_RETRIES = 3


def _decode_batch(video: str, reader, indices: list[int]):
    """decord raises on slow NFS reads near EOF ("takes too long to retrieve last few frames"); retry, then decode one by one."""
    from decord import DECORDError, VideoReader, cpu

    fps = float(reader.get_avg_fps())
    for attempt in range(DECODE_RETRIES):
        try:
            return reader.get_batch(indices).asnumpy()
        except DECORDError as exc:
            logger.warning("decord batch read failed (%s/%s) for %s: %s", attempt + 1, DECODE_RETRIES, video, str(exc)[:80])
            reader = VideoReader(video, ctx=cpu(0), num_threads=1)
    try:
        return np.stack([reader[i].asnumpy() for i in indices])
    except DECORDError as exc:
        logger.warning("decord per-frame read failed for %s (%s); falling back to ffmpeg", video, str(exc)[:60])
    return _ffmpeg_frames(video, indices, fps)


def _ffmpeg_frames(video: str, indices: list[int], fps: float):
    """Last resort: decode the requested frames with ffmpeg by timestamp. Slower, but immune to decord's EOF handling."""
    import io
    import subprocess

    from src.utils.media import ffmpeg_binary

    frames = []
    for index in indices:
        t_sec = max(0.0, index / fps)
        cmd = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-ss", f"{t_sec:.3f}", "-i", video,
               "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
        out = subprocess.run(cmd, capture_output=True, timeout=120).stdout
        if not out:  # beyond the last decodable timestamp: step back a little
            cmd[5] = f"{max(0.0, t_sec - 0.5):.3f}"
            out = subprocess.run(cmd, capture_output=True, timeout=120).stdout
        if not out:
            raise ValueError(f"ffmpeg could not decode frame {index} of {video}")
        frames.append(np.asarray(Image.open(io.BytesIO(out)).convert("RGB")))
    return np.stack(frames)


def _store_frames(sample_id: str, video: str, indices: Sequence[int], io: FrameIO) -> list[tuple[float, str]]:
    """Decode the indices once and return (t_sec, raw_path) in temporal order."""
    from decord import VideoReader, cpu

    reader = VideoReader(video, ctx=cpu(0), num_threads=2)
    fps = float(reader.get_avg_fps())
    frames_np = _decode_batch(video, reader, list(indices))
    stored: list[tuple[float, str]] = []
    for index, array in zip(indices, frames_np):
        t_sec = index / fps
        path = io.raw_path(sample_id, int(round(t_sec * 1000)))
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.fromarray(array)
            image.thumbnail((MAX_STORED_SIDE, MAX_STORED_SIDE))
            image.save(path, quality=92)
        stored.append((t_sec, str(path)))
    return stored


def _store_at_times(sample: AgentInput, times: Sequence[float], exclude: Sequence[float], io: FrameIO) -> list[tuple[float, str]]:
    """Save frames nearest to `times`, skipping decoded indices already present."""
    from decord import VideoReader, cpu

    reader = VideoReader(sample.prefix_path, ctx=cpu(0), num_threads=1)
    fps = float(reader.get_avg_fps())
    taken = {int(round(t * fps)) for t in exclude}
    indices = sorted({min(int(round(t * fps)), len(reader) - 1) for t in times} - taken)
    return _store_frames(sample.sample_id, sample.prefix_path, indices, io) if indices else []


def _labeled(sample_id: str, frame: dict[str, Any], use_vp: bool, io: FrameIO) -> dict[str, Any]:
    """Return the frame with its display path: the VP-labelled copy when enabled."""
    path = frame["raw_path"]
    if use_vp:
        target = io.labeled_path(sample_id, frame["frame_id"], int(round(frame["t_sec"] * 1000)))
        if not target.exists():
            render_number(frame["raw_path"], target, frame["frame_id"])
        path = str(target)
    return {**frame, "path": path}


def number_frames(
    sample_id: str, frames: Sequence[dict[str, Any]], use_vp: bool, io: FrameIO
) -> tuple[dict[str, Any], ...]:
    """Give the initial frames uniform ids "#01".. in time order and render the visual prompt."""
    ordered = sorted(frames, key=lambda frame: frame["t_sec"])
    return tuple(
        _labeled(
            sample_id,
            {**frame, "frame_id": frame_ids.uniform_id(position), "source": frame_ids.SOURCE_UNIFORM},
            use_vp,
            io,
        )
        for position, frame in enumerate(ordered, 1)
    )


def sample_frames(sample: AgentInput, cfg: RunConfig, io: FrameIO) -> SampleState:
    """Create the initial state, or a video_error state if the clip cannot be decoded."""
    from decord import VideoReader, cpu

    try:
        total = len(VideoReader(sample.prefix_path, ctx=cpu(0), num_threads=1))
        if getattr(cfg, "frame_select", "uniform") == "last":
            indices = list(range(max(0, total - cfg.num_frames), total))
        else:
            indices = uniform_indices(total, cfg.num_frames)
        stored = _store_frames(sample.sample_id, sample.prefix_path, indices, io)
    except (RuntimeError, ValueError, OSError) as exc:
        return SampleState(input=sample, status=STATUS_VIDEO_ERROR, flags=(f"video_error: {exc}",))
    frames = [{"t_sec": t_sec, "raw_path": path, "caption": ""} for t_sec, path in stored]
    flags = ("short_prefix",) if len(frames) < MIN_FRAMES_FLAG else ()
    numbered = number_frames(sample.sample_id, frames, cfg.use_vp, io)
    if getattr(cfg, "frame_order", "temporal") == "shuffle":
        # Temporal control: ids/labels keep their true time order, only the presentation order is permuted.
        import random
        numbered = tuple(random.Random(f"shuffle:{sample.sample_id}").sample(list(numbered), len(numbered)))
        flags = flags + ("shuffled_order",)
    return SampleState(input=sample, frames=numbered, flags=flags)


def add_frames(
    state: SampleState, times: Sequence[float], source: str, use_vp: bool, io: FrameIO
) -> tuple[SampleState, list[str]]:
    """Insert frames at `times` with sub-numbered ids; existing ids and times are untouched."""
    stored = _store_at_times(state.input, times, [frame["t_sec"] for frame in state.frames], io)
    if not stored:
        return state, []
    assigned = frame_ids.next_sub_ids(state.frames, [t_sec for t_sec, _ in stored])
    by_time = {t_sec: path for t_sec, path in stored}
    added = [
        _labeled(
            state.input.sample_id,
            {"frame_id": frame_id, "t_sec": t_sec, "source": source, "raw_path": by_time[t_sec], "caption": ""},
            use_vp,
            io,
        )
        for frame_id, t_sec in assigned
    ]
    frames = tuple(sorted([*state.frames, *added], key=lambda frame: frame["t_sec"]))
    new_ids = [frame["frame_id"] for frame in added]
    return replace(state, frames=frames, extra_frames=state.extra_frames + len(added)), new_ids


def frame_ids_of(state: SampleState) -> list[str]:
    return [frame["frame_id"] for frame in state.frames]


def range_midpoints(frames: Sequence[dict[str, Any]], first: str, last: str, limit: int) -> list[float]:
    """Midpoints of adjacent frames in [first, last] (ids), thinned evenly to at most `limit`."""
    low, high = frame_ids.sort_key(first), frame_ids.sort_key(last)
    selected = [f["t_sec"] for f in frames if low <= frame_ids.sort_key(f["frame_id"]) <= high]
    times = [(a + b) / 2 for a, b in zip(selected, selected[1:])]
    if len(times) <= limit:
        return times
    return [times[int(round(i))] for i in np.linspace(0, len(times) - 1, limit)]
