"""Re-observation tools, always inside the observed clip (after VepAgent's evidence re-search, region zoom and
terminal-state tools). Each tool returns evidence that can be cited by a fixed frame id: new frames get sub-numbered ids,
zoom crops get "<id>z".
"""
from __future__ import annotations

import json

from dataclasses import replace
from typing import Any, Sequence

from PIL import Image

from src.agent import frame_ids, observer
from src.agent.observer import FrameIO
from src.schemas import RunConfig, SampleState

DENSE_RESAMPLE = "dense_resample"
ZOOM = "zoom"
TERMINAL_STATE = "terminal_state"
AUDIO_SEGMENT = "audio_segment"
TOOL_DESCRIPTIONS = {
    DENSE_RESAMPLE: "dense_resample: view more frames between the first and last frame of the range",
    ZOOM: "zoom: enlarge one region of the first and last frame of the range",
    TERMINAL_STATE: "terminal_state: re-inspect the last seconds for the action in progress and object states",
    AUDIO_SEGMENT: "audio_segment: read all speech and sounds within the range",
}
MAX_NEW_FRAMES = 6
TERMINAL_WINDOW_SEC = 3.0
TERMINAL_FRAMES = 3


def available_tools(state: SampleState, cfg: RunConfig) -> tuple[str, ...]:
    tools = (DENSE_RESAMPLE, ZOOM, TERMINAL_STATE)
    tools = tools + (AUDIO_SEGMENT,) if cfg.use_audio and state.audio else tools
    allowed = {t.strip() for t in cfg.tool_menu.split(",") if t.strip()}  # cfg.tool_menu restricts the menu
    return tuple(t for t in tools if t in allowed) if allowed else tools


def _crop_box(width: int, height: int, region: str) -> tuple[int, int, int, int]:
    """Half-size crop anchored at the named region."""
    crop_w, crop_h = width // 2, height // 2
    left = {"center": width // 4, "top-left": 0, "bottom-left": 0}.get(region, width - crop_w)
    top = {"center": height // 4, "top-left": 0, "top-right": 0}.get(region, height - crop_h)
    return left, top, left + crop_w, top + crop_h


def _time_range(state: SampleState, where: Sequence[str]) -> tuple[float, float]:
    by_id = {frame["frame_id"]: frame["t_sec"] for frame in state.frames}
    return by_id[where[0]], by_id[where[1]]


def apply_zoom(state: SampleState, gap: dict[str, Any], io: FrameIO, mode: str = "region") -> SampleState:
    """Add enlarged crops of the two range endpoints as extra citable images ("#07z").

    mode "region": half-size crop at the named region. mode "grounded": crop around the object that Grounding DINO finds for the
    request's cue (gap["missing"]); when nothing is found the half-size region crop is used and the entry records the fallback.
    """
    import hashlib
    by_id = {frame["frame_id"]: frame for frame in state.frames}
    already = {entry["frame_id"] for entry in state.extra_evidence if entry["kind"] == ZOOM}
    cue = str(gap.get("missing") or "")
    tag = gap["region"] if mode != "grounded" else "g" + hashlib.sha1(cue.encode("utf-8")).hexdigest()[:10]
    added: list[dict[str, Any]] = []
    for parent in sorted(set(gap["where"]), key=frame_ids.sort_key):
        zoom = frame_ids.zoom_id(parent)
        if zoom in already:
            continue
        frame = by_id[parent]
        target = io.cache_root / state.input.sample_id / "zoom" / f"{frame_ids.file_stem(zoom)}_{tag}.jpg"
        meta_path = target.with_suffix(".json")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(frame["raw_path"]) as opened:
                image = opened.convert("RGB")
            box, meta = _crop_box(*image.size, gap["region"]), {"region": gap["region"], "grounded": False}
            if mode == "grounded":
                from src.agent import grounding
                found = grounding.detect(image, cue)
                if found is not None:
                    box = grounding.expand_box(found["box"], *image.size)
                    meta = {"region": "grounded", "grounded": True, "detected_box": [round(v, 1) for v in found["box"]],
                            "detector_score": round(found["score"], 3), "detector_label": found["label"], "query": grounding.query_text(cue)}
                else:
                    meta = {**meta, "query": grounding.query_text(cue), "fallback": "no_detection"}
            meta["crop_box"] = list(box)
            image.crop(box).resize(image.size, Image.Resampling.BICUBIC).save(target, quality=92)
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"region": gap["region"]}
        added.append({"kind": ZOOM, "frame_id": zoom, "parent": parent, "t_sec": frame["t_sec"], "path": str(target), **meta})
    return replace(state, extra_evidence=state.extra_evidence + tuple(added))


def apply_audio_segment(state: SampleState, gap: dict[str, Any]) -> SampleState:
    """Attach every cached speech and sound line inside the frame range, without prompt caps."""
    start, end = _time_range(state, gap["where"])
    audio = state.audio or {}
    lines = [f'speech: "{a["text"]}"' for a in audio.get("asr", []) if a["t_end"] >= start and a["t_start"] <= end]
    lines += [f'sound: {t["label"]}' for t in audio.get("tags", []) if start <= t["t_sec"] <= end]
    text = "; ".join(lines) if lines else "no speech or distinct sound in this range"
    entry = {"kind": AUDIO_SEGMENT, "where": list(gap["where"]), "text": text}
    return replace(state, extra_evidence=state.extra_evidence + (entry,))


def apply_dense_resample(
    state: SampleState, gap: dict[str, Any], cfg: RunConfig, io: FrameIO
) -> tuple[SampleState, list[str]]:
    """Insert midpoint frames inside the range; existing ids stay valid, new ids are sub-numbered."""
    times = observer.range_midpoints(state.frames, gap["where"][0], gap["where"][1], MAX_NEW_FRAMES)
    return observer.add_frames(state, times, frame_ids.SOURCE_DENSE, cfg.use_vp, io)


def apply_terminal_frames(state: SampleState, cfg: RunConfig, io: FrameIO) -> tuple[SampleState, list[str]]:
    """Add the last seconds of the clip as sub-numbered frames of the final uniform frame."""
    last_time = state.frames[-1]["t_sec"]
    times = [
        max(0.0, last_time - TERMINAL_WINDOW_SEC * (1 - i / (TERMINAL_FRAMES - 1))) for i in range(TERMINAL_FRAMES)
    ]
    return observer.add_frames(state, times, frame_ids.SOURCE_TERMINAL, cfg.use_vp, io)
