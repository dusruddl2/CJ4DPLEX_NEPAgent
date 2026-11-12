"""Frame identifiers that never change once assigned.

Uniform frames are "#01".."#16". A frame added between "#02" and "#03" becomes "#02.1", "#02.2";
a frame added after the last frame becomes "#16.1". A zoomed crop of "#07" is "#07z".
A later insertion that lands before an existing sub-frame gets a deeper id ("#02.0.1" lies between
"#02" and "#02.1"), so ids never change and sorting by `sort_key` always equals sorting by time.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

_ID = re.compile(r"^#(\d{2,})((?:\.\d+)*)(z)?$")
SOURCE_UNIFORM = "uniform"
SOURCE_DENSE = "dense_resample"
SOURCE_TERMINAL = "terminal_state"
SOURCE_ZOOM = "zoom"


def uniform_id(position: int) -> str:
    """1-based position among the initial uniform frames -> "#01"."""
    return f"#{position:02d}"


def parse_id(frame_id: Any) -> tuple[tuple[int, ...], bool] | None:
    """(components, is_zoom) or None when the value is not a well-formed frame id."""
    if not isinstance(frame_id, str):
        return None
    match = _ID.match(frame_id.strip())
    if match is None:
        return None
    components = (int(match.group(1)), *(int(part) for part in match.group(2).split(".") if part))
    return components, match.group(3) is not None


def sort_key(frame_id: str) -> tuple[Any, ...]:
    parsed = parse_id(frame_id)
    if parsed is None:
        raise ValueError(f"not a frame id: {frame_id!r}")
    components, is_zoom = parsed
    return (*components, int(is_zoom))


def _format(components: Sequence[int]) -> str:
    return f"#{components[0]:02d}" + "".join(f".{part}" for part in components[1:])


def zoom_id(frame_id: str) -> str:
    return f"{frame_id}z"


def vp_label(frame_id: str) -> str:
    return f"frame {frame_id}"


def file_stem(frame_id: str) -> str:
    """Filesystem-safe form: "#02.1" -> "02_1", "#07z" -> "07z"."""
    return frame_id.lstrip("#").replace(".", "_")


def _between(anchor: tuple[int, ...], successor: tuple[int, ...] | None, used: set[tuple[int, ...]]) -> tuple[int, ...]:
    """Shortest unused id path strictly after `anchor` and strictly before `successor`."""

    def fits(candidate: tuple[int, ...]) -> bool:
        return candidate not in used and candidate > anchor and (successor is None or candidate < successor)

    for length in range(1, len(anchor) + 1):
        candidate = anchor[:length] + (anchor[length] + 1,) if length < len(anchor) else anchor + (1,)
        if fits(candidate):
            return candidate
    candidate = anchor + (0, 1)
    while not fits(candidate):
        candidate = candidate[:-1] + (0, 1)
    return candidate


def next_sub_ids(frames: Sequence[dict[str, Any]], times: Sequence[float]) -> list[tuple[str, float]]:
    """Assign sub-numbered ids for new frames at `times` without touching existing ids.

    Each new frame gets the shortest id that sorts after the latest existing frame at or before its
    time and before the next one, normally "<anchor>.k". Returns (frame_id, t_sec) pairs sorted by time.
    """
    timeline: list[tuple[float, tuple[int, ...]]] = sorted(
        (frame["t_sec"], parse_id(frame["frame_id"])[0]) for frame in frames  # type: ignore[index]
    )
    used = {components for _, components in timeline}
    assigned: list[tuple[str, float]] = []
    for t_sec in sorted(times):
        anchor = timeline[0][1]
        successor: tuple[int, ...] | None = None
        for time, components in timeline:
            if time <= t_sec:
                anchor = components
            elif successor is None:
                successor = components
        candidate = _between(anchor, successor, used)
        used.add(candidate)
        timeline = sorted([*timeline, (t_sec, candidate)])
        assigned.append((_format(candidate), t_sec))
    return assigned


def valid_ids(values: Any, known: Sequence[str]) -> tuple[list[str], int]:
    """Keep ids that exist among `known`, sorted by time order; also return how many were rejected."""
    if not isinstance(values, list):
        return [], 0
    known_set = set(known)
    ordered = sorted(known, key=sort_key)
    kept: set[str] = set()
    for value in values:
        if isinstance(value, bool):
            continue
        position = value if isinstance(value, int) else (int(value) if isinstance(value, str) and value.strip().isdigit() else None)
        if position is not None:
            # Without visual labels the model refers to frames by position; map it to the k-th frame in time order.
            if 1 <= position <= len(ordered):
                kept.add(ordered[position - 1])
        elif isinstance(value, str) and value.strip() in known_set:
            kept.add(value.strip())
    return sorted(kept, key=sort_key), len(values) - len(kept)


def check_order(frames: Sequence[dict[str, Any]]) -> bool:
    """True when sorting by id gives the same order as sorting by time."""
    by_id = [frame["frame_id"] for frame in sorted(frames, key=lambda f: sort_key(f["frame_id"]))]
    by_time = [frame["frame_id"] for frame in sorted(frames, key=lambda f: f["t_sec"])]
    return by_id == by_time
