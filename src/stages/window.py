"""Re-observation window: normalise and validate a requested (tool, frame window), then run the tool on the observed clip."""
from __future__ import annotations

import re
from typing import Any, Sequence

from src.agent import frame_ids, observer, prompting, tools
from src.schemas import RunConfig
from src.stages.evidence import evidence_registry
from src.stages.state import ToolState

MAX_WINDOW_FRAMES = 8
PLACEHOLDER_FACTS = {"observable fact to check", "the specific fact", "<the specific fact>", ""}
ID_PATTERN = re.compile(r"#\d+(?:\.\d+)?z?")


def normalize_where(payload: dict[str, Any]) -> dict[str, Any]:
    """Tolerate common `where` spellings before validation.

    Observed replies: a single frame (["#32"]), "frame #32" prefixes, or an explicit list of every frame in the window.
    The window becomes [first, last] of the frame ids found, in the order given. Width and existence are still checked
    by parse_window_plan/validate_plan, so a whole-clip window or an unknown id is still rejected.
    """
    where = payload.get("where")
    if isinstance(where, str):
        where = [where]
    if not isinstance(where, list) or not where:
        return payload
    ids: list[str] = []
    for item in where:
        found = ID_PATTERN.findall(str(item))
        ids.extend(found if found else [str(item).strip()])
    if len(ids) == 1:
        ids = [ids[0], ids[0]]
    return {**payload, "where": [ids[0], ids[-1]]}


def clamp_window(payload: Any, ids: Sequence[str]) -> Any:
    """A requested window wider than MAX_WINDOW_FRAMES keeps its last MAX_WINDOW_FRAMES frames (the end of the clip is
    what matters for the next event) instead of being rejected; the whole clip therefore never passes as is."""
    where = payload.get("where") if isinstance(payload, dict) else None
    if not (isinstance(where, list) and len(where) == 2 and where[0] in ids and where[1] in ids):
        return payload
    first, last = ids.index(where[0]), ids.index(where[1])
    if last - first + 1 <= MAX_WINDOW_FRAMES:
        return payload
    return {**payload, "where": [ids[last - MAX_WINDOW_FRAMES + 1], ids[last]], "clamped_from": list(where)}


def parse_window_plan(payload: Any, ids: Sequence[str], menu: Sequence[str]) -> dict[str, Any] | None:
    """Window-selection plan: it must always name a cue and a window; no 'unobservable' exit."""
    if not isinstance(payload, dict):
        return None
    cue = payload.get("cue")
    if not isinstance(cue, str) or not cue.strip() or cue.strip().lower() in PLACEHOLDER_FACTS:
        return None
    where = payload.get("where")
    if not isinstance(where, list) or len(where) != 2 or payload.get("tool") not in menu:
        return None
    # Without visual labels frames are referred to by position ("1".."N"); map both forms to ids.
    mapped = [frame_ids.valid_ids([w], ids)[0] for w in where]
    if any(len(m) != 1 for m in mapped):
        return None
    where = [mapped[0][0], mapped[1][0]]
    if ids.index(where[0]) > ids.index(where[1]):
        return None
    return {"sufficient": False, "resolvable": True, "gap_scope": "observed", "missing": cue,
            "where": where, "tool": payload["tool"], "region": payload.get("region", "center")}


def validate_plan(plan: dict[str, Any] | None, state: ToolState) -> dict[str, Any] | None:
    """Reject plans that echo the template or ask for a window wider than allowed."""
    if plan is None:
        return None
    if plan.get("sufficient") or not plan.get("resolvable"):
        return plan
    if plan.get("missing", "").strip().lower() in PLACEHOLDER_FACTS:
        return None
    ids = observer.frame_ids_of(state)
    first, last = plan["where"]
    if first not in ids or last not in ids or ids.index(last) - ids.index(first) + 1 > MAX_WINDOW_FRAMES:
        return None
    return plan


def execute(state: ToolState, plan: dict[str, Any], cfg: RunConfig, io) -> tuple[ToolState, list[str]]:
    """Acquire pixels/audio only: no VLM call."""
    if plan["tool"] == tools.DENSE_RESAMPLE:
        return tools.apply_dense_resample(state, plan, cfg, io)
    if plan["tool"] == tools.TERMINAL_STATE:
        return tools.apply_terminal_frames(state, cfg, io)
    if plan["tool"] == tools.ZOOM:
        before = set(evidence_registry(state))
        updated = tools.apply_zoom(state, plan, io, cfg.zoom_mode)
        return updated, sorted(set(evidence_registry(updated)) - before)
    # A cached segment is new information only if it exposes lines not already in
    # the initial capped audio prompt. Reformatting existing audio does not qualify.
    updated = tools.apply_audio_segment(state, plan)
    full = prompting.audio_block(state.audio, state.frames)
    by_id = {f["frame_id"]: f["t_sec"] for f in state.frames}
    start, end = (by_id[i] for i in plan["where"])
    audio = state.audio or {}
    novel = [a for a in audio.get("asr", []) if a["t_end"] >= start and a["t_start"] <= end
             and a["text"].strip() and a["text"] not in full]
    novel += [t for t in audio.get("tags", []) if start <= t["t_sec"] <= end and t["label"] not in full]
    ids = [f"audio:{plan['where'][0]}-{plan['where'][1]}"] if novel else []
    return updated, ids
