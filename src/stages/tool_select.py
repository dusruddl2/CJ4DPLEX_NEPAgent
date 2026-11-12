"""Tool selection and execution: one call names a tool and a frame window on the observed clip, then that tool is run."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from string import Template
from typing import Any

from src.agent import observer, tools
from src.agent.vlm_ops import generate_validated
from src.models.vlm import GenerationConfig, VLMRequest
from src.schemas import RunConfig
from src.stages import direct
from src.stages.routing import bill
from src.stages.state import RoutedState
from src.stages.window import clamp_window, execute, normalize_where, parse_window_plan, validate_plan

SELECT_MAX_TOKENS = 384


def _select_request(state: RoutedState, cfg: RunConfig) -> VLMRequest:
    text = Template((Path(__file__).parent / "prompts" / "select_tool.txt").read_text(encoding="utf-8")).substitute(
        direct_event=state.direct["event"], menu="\n".join(tools.TOOL_DESCRIPTIONS[t] for t in tools.available_tools(state, cfg)),
        first_id=state.frames[0]["frame_id"], last_id=state.frames[-1]["frame_id"], **direct.common(state, cfg))
    return VLMRequest(text=text, images=direct.images(state))


def parse_selection(payload: Any, state: RoutedState, cfg: RunConfig) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("observable_gap"), bool):
        return None
    ids = observer.frame_ids_of(state)
    normalized = clamp_window(normalize_where(dict(payload)), ids)
    plan = validate_plan(parse_window_plan(normalized, ids, tools.available_tools(state, cfg)), state)
    if plan is None:
        return None
    times = {f["frame_id"]: f["t_sec"] for f in state.frames}
    text = lambda key: payload[key] if isinstance(payload.get(key), str) else ""
    return {"plan": plan, "tool": plan["tool"], "where": list(plan["where"]), "cue": plan["missing"], "observable_gap": payload["observable_gap"],
            "target_interval": [times[plan["where"][0]], times[plan["where"][1]]], "reason": text("reason"),
            "expected_evidence": text("expected_evidence"), "clamped_from": normalized.get("clamped_from")}


def select_stage(states: list[RoutedState], cfg: RunConfig, services) -> list[RoutedState]:
    requests = [_select_request(s, cfg) for s in states]
    validators = [lambda p, s=s: parse_selection(p, s, cfg) for s in states]
    charged, parsed, replies = generate_validated(services.vlm, states, requests, validators, GenerationConfig(max_tokens=SELECT_MAX_TOKENS))
    out = []
    for before, after, selection, reply in zip(states, charged, parsed, replies):
        billed = bill(before, "select", after.calls - before.calls, after.elapsed - before.elapsed)
        record = selection if selection is not None else {"status": "invalid", "raw": reply or ""}
        out.append(replace(billed, tool_selection=record, tool_observation=None if selection else {"status": "plan_failed", "new_evidence_ids": []}))
    return out


def run_tool(state: RoutedState, cfg: RunConfig, services) -> RoutedState:
    """One tool on the observed clip. On success the returned state carries the added frames; otherwise it is unchanged."""
    selection = state.tool_selection or {}
    plan = selection.get("plan")
    if not plan:
        return state
    base = {"tool": plan["tool"], "where": list(plan["where"])}
    try:
        updated, new_ids = execute(replace(state, routed=True, rounds=1, plan=plan), plan, cfg, services.frames)
    except (OSError, RuntimeError, ValueError) as exc:
        return replace(state, tool_observation={**base, "status": "failed:" + type(exc).__name__, "new_evidence_ids": []}, flags=state.flags + ("tool_failed",))
    if not new_ids:
        return replace(state, tool_observation={**base, "status": "no_new_evidence", "new_evidence_ids": []}, flags=state.flags + ("no_new_evidence",))
    observation = {**base, "status": "executed", "new_evidence_ids": list(new_ids)}
    if plan["tool"] == tools.AUDIO_SEGMENT:  # audio evidence is text, not an image: keep it readable in the result
        observation["evidence_text"] = updated.extra_evidence[-1]["text"]
    return replace(updated, new_evidence_ids=tuple(new_ids), tool_observation=observation)
