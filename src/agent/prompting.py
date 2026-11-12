"""Render the prompt blocks shared by every stage: frame preamble, audio text, question and extra evidence."""
from __future__ import annotations

from typing import Any, Sequence

from src.agent.frame_ids import sort_key, uniform_id
from src.schemas import AgentInput

MAX_ASR_LINES = 40
MAX_TAG_LINES = 30


def frame_preamble(frames: int | Sequence[str], use_vp: bool, shuffled: bool = False) -> str:
    """ViKey's ordered-frame instruction when labels are burned in; a plain one otherwise.

    `frames` is the list of frame ids, or an int meaning that many uniform frames.
    """
    ids = [uniform_id(i) for i in range(1, frames + 1)] if isinstance(frames, int) else list(frames)
    ordered = sorted(ids, key=sort_key)
    count = len(ordered)
    from src.agent.vp import VP_STYLE
    if shuffled:
        if use_vp and VP_STYLE == "number":
            return (f"You are given {count} frames from the clip in random order. Each frame shows its label in the "
                    f"bottom-left corner; labels run from Frame {ordered[0]} (earliest) to Frame {ordered[-1]} (latest).")
        return f"You are given {count} frames from the clip in random order; their time order is not given."
    if use_vp and VP_STYLE == "number":
        text = (
            f"Please examine frames from Frame {ordered[0]} to Frame {ordered[-1]} in sequential order. "
            "Each frame shows its label in the bottom-left corner."
        )
        if any("." in frame_id for frame_id in ordered):
            text += " A label with a sub-number, such as #02.1, marks a frame that lies between #02 and #03."
        return text
    return (
        f"You are given {count} frames in temporal order. "
        f"Refer to a frame by its position: 1 for the first frame and {count} for the last."
    )


def frame_for_time(frames: Sequence[dict[str, Any]], t_sec: float) -> str:
    """Id of the latest observed frame at or before t_sec (first frame if none)."""
    frame_id = frames[0]["frame_id"]
    for frame in frames:
        if frame["t_sec"] <= t_sec:
            frame_id = frame["frame_id"]
    return frame_id


def audio_block(audio: dict[str, Any] | None, frames: Sequence[dict[str, Any]]) -> str:
    """One line per speech segment and sound event, aligned to frame ids."""
    if not audio:
        return "Audio: not available."
    lines: list[str] = []
    for item in audio.get("asr", [])[-MAX_ASR_LINES:]:
        lines.append(f'[frame {frame_for_time(frames, item["t_start"])}] speech: "{item["text"]}"')
    for item in audio.get("tags", [])[-MAX_TAG_LINES:]:
        lines.append(f'[frame {frame_for_time(frames, item["t_sec"])}] sound: {item["label"]}')
    return "Audio:\n" + "\n".join(lines) if lines else "Audio: no speech or distinct sound detected."


def question_block(sample: AgentInput) -> str:
    return f"Question: {sample.question}\nOptions:\n{sample.options_text()}"


def extra_block(extras: Sequence[dict[str, Any]]) -> str:
    """Describe re-observed evidence; zoom images follow the labelled frames in image order."""
    lines: list[str] = []
    for entry in extras:
        if entry["kind"] == "zoom":
            lines.append(
                f'- image {entry["frame_id"]} (after the labelled frames) is an enlarged {entry["region"]} '
                f'region of frame {entry["parent"]}'
            )
        elif entry["kind"] == "audio_segment":
            lines.append(f'- audio in frames {entry["where"][0]} to {entry["where"][1]}: {entry["text"]}')
        else:
            lines.append(f'- at the end of the clip, {entry["text"]}')
    return "\nAdditional observations from re-examining the clip:\n" + "\n".join(lines) + "\n" if lines else ""
