"""Candidate bookkeeping and the routing signal: per-stage call counts, the stored form of a matched prediction, and the
generation confidence of the Direct sentence."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from src.schemas import RunConfig
from src.stages.state import RoutedState


def bill(state: RoutedState, key: str, calls: int, seconds: float = 0.0) -> RoutedState:
    return replace(state, calls=state.calls + calls, elapsed=state.elapsed + seconds,
                   stage_calls={**state.stage_calls, key: state.stage_calls.get(key, 0) + calls})


def candidate(state: RoutedState) -> dict[str, Any] | None:
    if not state.alive or not state.prediction or not state.predicted_event:
        return None
    final = state.final or {}
    return {"event": state.predicted_event, "frozen_sha": state.frozen_sha, "choice": state.prediction["choice"],
            "option_probs": state.prediction["option_probs"], "transition": final.get("predicted_transition", ""),
            "supporting_evidence": list(final.get("supporting_evidence") or []), "json_ok": state.final is not None, "raw_reply": final.get("raw_reply", "")}


def route(state: RoutedState, cfg: RunConfig) -> RoutedState:
    logprob = state.generation_logprob
    confidence = math.exp(logprob) if logprob is not None else None
    routed = bool(state.alive and cfg.use_reflector and confidence is not None and confidence < cfg.gate_tau)
    return replace(state, routing={"signal": "direct_sentence_mean_token_logprob", "mean_logprob": logprob,
                                   "generation_confidence": confidence, "gate_tau": cfg.gate_tau, "routed": routed})
