"""NEPAgent: Direct first; tools and causal reasoning only when the Direct sentence is not confident.

    Direct (one open-ended sentence) -> text-only matcher                             always
    confidence = exp(mean token log-prob of the sentence); if it is >= cfg.gate_tau the Direct answer is final.
    Otherwise the candidates named in `cfg.final_answer` are tried in order until one exists:
      tool_reason : one tool-selection call -> the tool (no VLM call) -> the reasoning variant on the clip plus the new
                    evidence -> matcher
      <variant>   : one structured reasoning call on the clip -> matcher (coe | causal | coe_causal | causal_sft ...)
    and Direct is kept when none does. Options reach only the matcher.

With `cfg.store_all_candidates` every variant in `cfg.reasoning_variants` is computed for every item instead (ablations and
offline policy comparison, scripts/policy_eval.py); `cfg.final_answer` then only picks which stored candidate is reported.
The extra variant `tool` (re-observation requested by coe_causal, then an update call) exists only in that mode.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.agent import observer, prompting, tools
from src.agent.parsing import extract_json
from src.nep_agent import event_reasoning as er
from src.nep_agent.state import NEPAgentState, lift
from src.models.vlm import VLMRequest
from src.schemas import STATUS_OK, RunConfig, SampleState
from src.stages import direct as direct_step, routing
from src.stages.evidence import evidence_registry
from src.stages.questions import normalize_question
from src.stages.tool_select import parse_selection, run_tool, select_stage
from src.stages.window import clamp_window, execute, normalize_where, parse_window_plan, validate_plan

REASON_MAX_TOKENS = 640
TOOL = "tool"
TOOL_REASON = "tool_reason"  # select one tool, run it, then the reasoning variant again on the clip plus the new evidence
ALL_VARIANTS = (*er.VARIANTS, TOOL, TOOL_REASON)


def _variants(cfg: RunConfig) -> list[str]:
    names = [v.strip() for v in cfg.reasoning_variants.split(",") if v.strip()]
    unknown = set(names) - set(ALL_VARIANTS)
    if unknown:
        raise ValueError(f"unknown reasoning variants: {sorted(unknown)}")
    if TOOL in names and not {"coe_causal", "coe_causal2"} & set(names):
        names.insert(names.index(TOOL), "coe_causal")  # the tool path acts on the coe_causal request
    return names


def _citable(state: NEPAgentState) -> list[str]:
    return list(dict.fromkeys([*observer.frame_ids_of(state), *evidence_registry(state), *state.new_evidence_ids]))


def _menu(state: NEPAgentState, cfg: RunConfig) -> str:
    return "\n".join(tools.TOOL_DESCRIPTIONS[t] for t in tools.available_tools(state, cfg))


NO_AUDIO = "Audio: not available."


def _reason_common(state: NEPAgentState, cfg: RunConfig) -> dict[str, str]:
    """Prompt blocks of the reasoning calls; reason_audio "none" matches the training prompts, which had no audio text."""
    common = direct_step.common(state, cfg)
    return {**common, "audio": NO_AUDIO} if cfg.reason_audio == "none" else common


def _tool_note(state: NEPAgentState) -> str:
    frames = [i for i in state.new_evidence_ids if i.startswith("#")]
    if not frames:
        return ""
    return ("\nFrames added by re-examining the clip: " + ", ".join(frames) + ". They come from inside the observed clip and sit "
            "between the labelled frames with the neighbouring numbers; use them to check the state at the end of the clip.\n")


def _reason(states: list[NEPAgentState], cfg: RunConfig, services, texts: list[str], max_tokens: int = REASON_MAX_TOKENS) -> list[tuple[NEPAgentState, dict[str, Any] | None, str]]:
    requests = [VLMRequest(text=t, images=direct_step.images(s)) for s, t in zip(states, texts)]
    charged, replies = direct_step.generate(services.vlm, states, requests, max_tokens)
    return [(s, er.parse_reasoning(extract_json(r), s.frames, _citable(s)), r or "") for s, r in zip(charged, replies)]


def _match_record(states: list[NEPAgentState], records: list[dict[str, Any]], services) -> list[tuple[NEPAgentState, dict[str, Any] | None]]:
    """Freeze each predicted event and run the text-only matcher on throwaway copies."""
    copies = [direct_step.freeze(replace(s, predicted_event=r["predicted_event"], final={"predicted_event": r["predicted_event"],
                                "supporting_evidence": r["supporting_evidence"]})) for s, r in zip(states, records)]
    matched = direct_step.match_stage(copies, services)
    return [(m, routing.candidate(m)) for m in matched]


def _store(state: NEPAgentState, name: str, before_calls: int, after: NEPAgentState | None, record: dict[str, Any] | None,
           candidate: dict[str, Any] | None, extra: dict[str, Any] | None = None, raw: str = "") -> NEPAgentState:
    calls = (after.calls - before_calls) if after is not None else 0
    seconds = (after.elapsed - state.elapsed) if after is not None else 0.0
    entry = {"record": record, "candidate": candidate, "raw_reply": raw if record is None else "", **(extra or {})}
    return replace(routing.bill(state, name, calls, seconds), reasoning={**state.reasoning, name: entry})


def reasoning_stage(states: list[NEPAgentState], cfg: RunConfig, services, variant: str) -> list[NEPAgentState]:
    texts = [er.render_reason(variant, _reason_common(s, cfg), s.frames[0]["frame_id"], s.frames[-1]["frame_id"], _menu(s, cfg)) for s in states]
    reasoned = _reason(states, cfg, services, texts, er.MAX_TOKENS.get(variant, REASON_MAX_TOKENS))
    ok = [i for i, (_, rec, _) in enumerate(reasoned) if rec is not None]
    matched = dict(zip(ok, _match_record([reasoned[i][0] for i in ok], [reasoned[i][1] for i in ok], services)))
    out = []
    for i, (state, (charged, rec, raw)) in enumerate(zip(states, reasoned)):
        after, cand = matched.get(i, (charged, None))
        out.append(_store(state, variant, state.calls, after, rec, cand, raw=raw))
    return out


def _tool_plan(state: NEPAgentState, cfg: RunConfig) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    entry = state.reasoning.get("coe_causal2") or state.reasoning.get("coe_causal") or {}
    need = (entry.get("record") or {}).get("missing_evidence") or {}
    if need.get("needed") is not True:
        return None, {"status": "not_requested"}
    ids = observer.frame_ids_of(state)
    payload = clamp_window(normalize_where({**need, "cue": need.get("question") or "detail inside the clip"}), ids)
    if (payload.get("tool") == tools.TERMINAL_STATE) and not payload.get("where"):
        payload = {**payload, "where": [ids[-2], ids[-1]]}
    plan = validate_plan(parse_window_plan(payload, ids, tools.available_tools(state, cfg)), state)
    if plan is None:
        return None, {"status": "invalid_request", "requested": need}
    times = {f["frame_id"]: f["t_sec"] for f in state.frames}
    return plan, {"status": "planned", "tool": plan["tool"], "where": list(plan["where"]), "question": need.get("question"),
                  "target_interval": [times[plan["where"][0]], times[plan["where"][1]]], "clamped_from": payload.get("clamped_from")}


def tool_stage(states: list[NEPAgentState], cfg: RunConfig, services) -> list[NEPAgentState]:
    out = list(states)
    pending: list[tuple[int, NEPAgentState, dict[str, Any]]] = []
    for i, state in enumerate(states):
        plan, info = _tool_plan(state, cfg)
        if plan is None:
            out[i] = _store(state, TOOL, state.calls, None, None, None, {"observation": info})
            continue
        try:
            updated, new_ids = execute(replace(state, routed=True, rounds=1, plan=plan), plan, cfg, services.frames)
        except (OSError, RuntimeError, ValueError) as exc:
            out[i] = _store(state, TOOL, state.calls, None, None, None, {"observation": {**info, "status": "failed:" + type(exc).__name__}})
            continue
        if not new_ids:
            out[i] = _store(state, TOOL, state.calls, None, None, None, {"observation": {**info, "status": "no_new_evidence"}})
            continue
        pending.append((i, replace(updated, new_evidence_ids=tuple(new_ids)), {**info, "status": "executed", "new_evidence_ids": list(new_ids)}))
    if not pending:
        return out
    bases = [p[1] for p in pending]
    texts = [er.render_update((s.reasoning.get("coe_causal2") or s.reasoning["coe_causal"])["record"], prompting.extra_block(s.extra_evidence), direct_step.common(s, cfg))
             for s in bases]
    reasoned = _reason(bases, cfg, services, texts, er.MAX_TOKENS.get("coe_causal2", REASON_MAX_TOKENS))
    ok = [k for k, (_, rec, _) in enumerate(reasoned) if rec is not None]
    matched = dict(zip(ok, _match_record([reasoned[k][0] for k in ok], [reasoned[k][1] for k in ok], services)))
    for k, ((i, base, info), (charged, rec, raw)) in enumerate(zip(pending, reasoned)):
        after, cand = matched.get(k, (charged, None))
        stored = _store(base, TOOL, base.calls, after, rec, cand, {"observation": info}, raw=raw)
        out[i] = replace(stored, tool_observation=info)
    return out


def _reason_variant(cfg: RunConfig) -> str:
    """The reasoning variant that tool_reason repeats: the last reasoning variant listed before it."""
    names = [v for v in _variants(cfg) if v in er.VARIANTS]
    if not names:
        raise ValueError("tool_reason needs a reasoning variant in reasoning_variants")
    return names[-1]


def _default_selection(state: NEPAgentState, cfg: RunConfig) -> dict[str, Any] | None:
    """Used when the selection reply is not a valid plan: re-inspect the end of the clip, where the next event starts."""
    ids = observer.frame_ids_of(state)
    payload = {"observable_gap": False, "tool": tools.TERMINAL_STATE, "where": [ids[-2], ids[-1]],
               "cue": "state of the main people and objects at the end of the clip", "reason": "default plan"}
    chosen = parse_selection(payload, state, cfg)
    return {**chosen, "default_plan": True} if chosen else None


def tool_reason_stage(states: list[NEPAgentState], cfg: RunConfig, services) -> list[NEPAgentState]:
    """Tool calling before reasoning: one selection call, the tool, then the reasoning variant on clip + new evidence.
    Stored for every sample; whether its answer is used (always, or only below a Direct-confidence threshold) is a policy."""
    variant = _reason_variant(cfg)
    selected = select_stage(states, cfg, services)
    selected = [s if (s.tool_selection or {}).get("plan") else replace(s, tool_selection=_default_selection(s, cfg) or s.tool_selection)
                for s in selected]
    tooled = [run_tool(s, cfg, services) for s in selected]
    out = [_store(s, TOOL_REASON, s.calls, None, None, None, {"selection": _public(s.tool_selection), "observation": s.tool_observation})
           for s in tooled]
    index = [i for i, s in enumerate(tooled) if (s.tool_observation or {}).get("status") == "executed"]
    bases = [tooled[i] for i in index]
    texts = [er.render_reason(variant, _reason_common(s, cfg), s.frames[0]["frame_id"], s.frames[-1]["frame_id"], _menu(s, cfg))
             + "\n" + prompting.extra_block(s.extra_evidence) + (_tool_note(s) if cfg.tool_note else "") for s in bases]
    reasoned = _reason(bases, cfg, services, texts, er.MAX_TOKENS.get(variant, REASON_MAX_TOKENS))
    ok = [k for k, (_, rec, _) in enumerate(reasoned) if rec is not None]
    matched = dict(zip(ok, _match_record([reasoned[k][0] for k in ok], [reasoned[k][1] for k in ok], services)))
    for k, (i, (charged, rec, raw)) in enumerate(zip(index, reasoned)):
        after, cand = matched.get(k, (charged, None))
        base = bases[k]
        out[i] = _store(base, TOOL_REASON, base.calls, after, rec, cand,
                        {"selection": _public(base.tool_selection), "observation": base.tool_observation, "variant": variant}, raw=raw)
    return out


def _public(selection: dict[str, Any] | None) -> dict[str, Any] | None:
    return {k: v for k, v in selection.items() if k != "plan"} if selection else None


def _preference(cfg: RunConfig) -> list[str]:
    """Candidates to report, in order of preference; "direct" ends the list."""
    return [name.strip() for name in cfg.final_answer.split(",") if name.strip()]


def _has_candidate(state: NEPAgentState, names: list[str]) -> bool:
    return any((state.reasoning.get(name) or {}).get("candidate") for name in names)


def gated_stage(states: list[NEPAgentState], cfg: RunConfig, services) -> list[NEPAgentState]:
    """Direct first: only items routed by the Direct confidence get more calls, and a later candidate of the preference list
    is computed only for the items that still have none. Every stage sees the clip as Direct saw it."""
    out = list(states)
    tried: list[str] = []
    for name in _preference(cfg):
        if name == "direct":
            break
        pending = [i for i, s in enumerate(out) if (s.routing or {}).get("routed") and not _has_candidate(s, tried)]
        views = [replace(out[i], frames=states[i].frames, extra_evidence=states[i].extra_evidence, extra_frames=states[i].extra_frames,
                         new_evidence_ids=states[i].new_evidence_ids) for i in pending]
        if name == TOOL_REASON:
            for i, done in zip(pending, tool_reason_stage(views, cfg, services)):
                out[i] = done
        else:
            for i, done in zip(pending, reasoning_stage(views, cfg, services, name)):  # keep the evidence a tool added earlier
                out[i] = replace(out[i], calls=done.calls, elapsed=done.elapsed, stage_calls=done.stage_calls, reasoning=done.reasoning)
        tried.append(name)
    return out


def _finalize(state: NEPAgentState, cfg: RunConfig) -> NEPAgentState:
    """Report the first preferred candidate that exists; Direct when none does. The confidence follows the reported candidate."""
    if not cfg.store_all_candidates and not (state.routing or {}).get("routed"):
        return state  # confident Direct answer: nothing else was computed
    for name in _preference(cfg):
        if name == "direct":
            return state
        chosen = (state.reasoning.get(name) or {}).get("candidate")
        if not chosen:
            continue
        confidence = chosen["option_probs"].get(chosen["choice"]) if chosen["option_probs"] else None
        return replace(state, final_source=name, predicted_event=chosen["event"], frozen_sha=chosen["frozen_sha"], matched=chosen["option_probs"],
                       prediction={**state.prediction, "choice": chosen["choice"], "answer": chosen["choice"], "option_probs": chosen["option_probs"],
                                   "confidence": confidence, "matcher_confidence": confidence, "next_event": chosen["event"]},
                       revision_accepted=True, decision_reason=f"{name}_selected")
    return replace(state, final_source="direct_fallback")


def run(states: list[SampleState], cfg: RunConfig, services) -> list[NEPAgentState]:
    from src.agent.pipeline import attach_audio

    if cfg.input_mode != "images":
        raise ValueError("nep_agent requires image inputs")
    variants = _variants(cfg)
    preference = _preference(cfg)
    if not preference or set(preference) - {"direct", *variants}:
        raise ValueError(f"final_answer {cfg.final_answer!r} is not computed by reasoning_variants")
    if not cfg.store_all_candidates and TOOL in preference:
        raise ValueError("the tool variant needs store_all_candidates; use tool_reason for the gated pipeline")
    current = [lift(s) for s in states]
    if cfg.use_audio:
        current = [attach_audio(s, services) for s in current]
    current = [replace(s, question_text=(note := normalize_question(s.input.question, cfg.question_mode))["text"], question_note=note) for s in current]
    current = direct_step.match_stage([direct_step.freeze(s) for s in direct_step.direct_stage(current, cfg, services)], services)
    current = [routing.route(replace(s, stage_calls={"direct_video": 1, "direct_match": max(0, s.calls - 1)}, direct=routing.candidate(s)), cfg) for s in current]
    alive = [i for i, s in enumerate(current) if s.alive and s.direct is not None]
    work = [replace(current[i], status=STATUS_OK) for i in alive]
    if not cfg.store_all_candidates:
        work = gated_stage(work, cfg, services)
    for variant in variants if cfg.store_all_candidates else ():
        if variant == TOOL:
            work = tool_stage(work, cfg, services)
        elif variant == TOOL_REASON:
            work = tool_reason_stage(work, cfg, services)
        else:
            work = reasoning_stage(work, cfg, services, variant)
    for i, state in zip(alive, work):
        current[i] = _finalize(state, cfg)
    return current
