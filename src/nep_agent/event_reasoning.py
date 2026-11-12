"""One-call temporal / causal reasoning for NEPAgent: prompt schemas per variant and validated parsing.

Variants share one prompt (`prompts/reason.txt`) and differ only in the fields asked for:
  coe            observed events in time order (chain of events, after Video-CoE) -> predicted event
  causal         terminal state, ongoing action, causal transition (causal gap, after VepAgent) -> predicted event
  coe_causal     both, plus a missing-evidence request that the tool variant may act on
  coe2, coe_causal2  the same with a compact event list (at most 5 merged events) and a larger output budget
  causal_sft     observed (event chain, terminal state, ongoing action, evidence) / inferred (intermediate states,
                 causal transition, predicted event): the schema of the distillation targets
  causal_sft_nc  causal_sft without the observed event chain (the submitted model)
Timestamps are not written by the model: events cite frame ids and the seconds come from the sampled frames, so they can
never leave the observed clip. Observed events that cite no valid frame are kept but marked ungrounded.
"""
from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Any, Sequence

from src.stages.window import ID_PATTERN

VARIANTS = ("coe", "causal", "coe_causal", "causal_sft", "coe2", "coe_causal2", "causal_sft_nc")
_EVENTS = '"observed_events": [{"first_frame": "<frame id>", "last_frame": "<frame id>", "description": "<what happens in those frames>"}]'
_CAUSAL = ('"terminal_state": "<state of the main people and objects in the last frame>", '
           '"ongoing_action": "<action still in progress at the last frame, or null>", '
           '"causal_transition": "<hypothesis: how that state changes next>"')
_PRED = '"predicted_event": "<one sentence>", "supporting_evidence": ["<frame id>"]'
_MISSING = ('"missing_evidence": {"needed": false, "question": "<what detail inside the clip is unclear, or null>", '
            '"tool": "<tool name>", "where": ["<first frame id>", "<last frame id>"], "region": "center"}')
TASKS = {
    "coe": "First list the events actually visible in the clip, in time order, as many as needed (usually 2 to 6). Then use that sequence to predict the next event.",
    "causal": "First state the terminal state at the last frame and the action still in progress, then the most likely transition from that state. Then predict the next event that transition leads to.",
    "coe_causal": ("First list the events actually visible in the clip, in time order, as many as needed (usually 2 to 6). Then state the terminal state "
                   "at the last frame, the action still in progress, and the most likely transition. Then predict the next event. Finally say whether one "
                   "detail INSIDE the clip is unclear and could change the prediction if re-examined (fast motion, a small object, whether the last action "
                   "finished, a spoken line); not knowing what comes next is not such a detail. If so set \"needed\" to true and name one tool and a window of at "
                   "most 8 consecutive frames ($first_id to $last_id). Tools:\n$menu"),
}
_SFT = ('{"observed": {"events": [{"first_frame": "<frame id>", "last_frame": "<frame id>", "description": "<what happens in those frames>"}], '
        '"terminal_state": "<state in the last frame>", "ongoing_action": "<action in progress at the last frame, or null>", '
        '"supporting_evidence": ["<frame id>"]}, '
        '"inferred": {"intermediate_states": ["<hypothesized state after the clip>"], "causal_transition": "<how the terminal state changes>", '
        '"predicted_event": "<one sentence>"}}')
TASKS["causal_sft"] = ("Separate what the clip shows from what you infer. Under \"observed\" list the events actually visible, in time order, the state in "
                       "the last frame and the action still in progress, citing frame ids. Under \"inferred\" give up to 3 hypothesized states after the "
                       "clip, the causal transition from the terminal state, and the predicted next event. Nothing under \"inferred\" may describe the clip.")
# causal_sft without the observed event chain. Video-CoE (arXiv 2603.14935) keeps event chains out of its SFT targets because
# MLLM-written chains are unreliable.
_SFT_NC = ('{"observed": {"terminal_state": "<state in the last frame>", "ongoing_action": "<action in progress at the last frame, or null>", '
           '"supporting_evidence": ["<frame id>"]}, '
           '"inferred": {"intermediate_states": ["<hypothesized state after the clip>"], "causal_transition": "<how the terminal state changes>", '
           '"predicted_event": "<one sentence>"}}')
TASKS["causal_sft_nc"] = ("Separate what the clip shows from what you infer. Under \"observed\" give the state in the last frame and the action "
                          "still in progress, citing frame ids. Under \"inferred\" give up to 3 hypothesized states after the clip, the causal "
                          "transition from the terminal state, and the predicted next event. Nothing under \"inferred\" may describe the clip.")
SCHEMAS = {"coe": "{" + _EVENTS + ", " + _PRED + "}", "causal": "{" + _CAUSAL + ", " + _PRED + "}",
           "coe_causal": "{" + _EVENTS + ", " + _CAUSAL + ", " + _PRED + ", " + _MISSING + "}", "causal_sft": _SFT, "causal_sft_nc": _SFT_NC}

# Listing one event per frame can exhaust the 640-token cap before "predicted_event" is written. The *2 variants ask for at
# most 5 merged events and get a larger output budget.
_COMPACT = (" List at most 5 events: merge consecutive frames that show the same action into one event, and do not describe "
            "individual frames.")
TASKS["coe2"] = TASKS["coe"].replace(" as many as needed (usually 2 to 6).", ".") + _COMPACT
TASKS["coe_causal2"] = TASKS["coe_causal"].replace(" as many as needed (usually 2 to 6).", ".", 1) + _COMPACT
SCHEMAS["coe2"] = SCHEMAS["coe"]
SCHEMAS["coe_causal2"] = SCHEMAS["coe_causal"]
MAX_TOKENS = {"coe2": 1536, "coe_causal2": 1536, "causal_sft": 1536, "causal_sft_nc": 1536}
UPDATE_SCHEMA = "{" + _CAUSAL + ", " + _PRED + "}"


def template(name: str) -> str:
    return (Path(__file__).parent / "prompts" / f"{name}.txt").read_text(encoding="utf-8")


def render_reason(variant: str, common: dict[str, str], first_id: str, last_id: str, menu: str) -> str:
    task = Template(TASKS[variant]).substitute(first_id=first_id, last_id=last_id, menu=menu)
    return Template(template("reason")).substitute(task=task, schema=SCHEMAS[variant], **common)


def render_update(previous: dict[str, Any], extra: str, common: dict[str, str]) -> str:
    kept = {k: previous.get(k) for k in ("observed_events", "terminal_state", "ongoing_action", "causal_transition", "predicted_event")}
    return Template(template("update")).substitute(previous=json.dumps(kept, ensure_ascii=False), extra=extra, schema=UPDATE_SCHEMA, **common)


def _ids_in(value: Any) -> list[str]:
    return ID_PATTERN.findall(str(value)) if value is not None else []


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() and value.strip().lower() not in ("null", "none", "unknown") else None


def flatten_sft(payload: dict[str, Any]) -> dict[str, Any]:
    """The causal_sft schema nests observed facts and inferred states; map it onto the flat record the rest of NEPAgent reads."""
    obs, inf = payload["observed"], payload["inferred"]
    states = [s for s in inf.get("intermediate_states") or [] if isinstance(s, str) and s.strip()] if isinstance(inf.get("intermediate_states"), list) else []
    return {"observed_events": obs.get("events"), "terminal_state": obs.get("terminal_state"), "ongoing_action": obs.get("ongoing_action"),
            "causal_transition": inf.get("causal_transition"), "predicted_event": inf.get("predicted_event"),
            "supporting_evidence": obs.get("supporting_evidence"), "intermediate_states": states}


def parse_reasoning(payload: Any, frames: Sequence[dict[str, Any]], citable: Sequence[str]) -> dict[str, Any] | None:
    """Validated record, or None when there is no usable predicted event (the caller then keeps Direct)."""
    if isinstance(payload, dict) and isinstance(payload.get("observed"), dict) and isinstance(payload.get("inferred"), dict):
        payload = flatten_sft(payload)
    if not isinstance(payload, dict) or not _text(payload.get("predicted_event")):
        return None
    times = {f["frame_id"]: f["t_sec"] for f in frames}
    events = []
    for item in payload.get("observed_events") or []:
        if not isinstance(item, dict) or not _text(item.get("description")):
            continue
        first = next(iter(_ids_in(item.get("first_frame"))), None)
        last = next(iter(_ids_in(item.get("last_frame"))), first)
        grounded = first in times and last in times and times[first] <= times[last]
        events.append({"description": item["description"].strip(), "first_frame": first, "last_frame": last, "grounded": grounded,
                       "start_time": times[first] if grounded else None, "end_time": times[last] if grounded else None})
    allowed = set(citable)
    return {"observed_events": events, "terminal_state": _text(payload.get("terminal_state")),
            "ongoing_action": _text(payload.get("ongoing_action")), "causal_transition": _text(payload.get("causal_transition")),
            "predicted_event": payload["predicted_event"].strip(),
            "supporting_evidence": [i for i in _ids_in(payload.get("supporting_evidence")) if i in allowed],
            "missing_evidence": payload.get("missing_evidence") if isinstance(payload.get("missing_evidence"), dict) else None,
            "intermediate_states": payload.get("intermediate_states") or []}
