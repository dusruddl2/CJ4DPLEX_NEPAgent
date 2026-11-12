"""Open-ended Direct step: one sentence about the next event from the frames and audio text, then the text-only matcher.

The answer options are never shown before the sentence is frozen; the matcher maps the frozen sentence onto the options.
Without options the matcher is skipped and the sentence is the prediction.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import replace
from pathlib import Path
from string import Template
from typing import Any

from src.agent import observer, prompting
from src.agent.vlm_ops import charge
from src.models.vlm import GenerationConfig, VLMRequest
from src.schemas import RunConfig
from src.stages import matcher
from src.stages.evidence import evidence_registry
from src.stages.state import DirectState

SENTENCE_MAX_TOKENS = 64


def _template(name: str) -> str:
    return (Path(__file__).parent / "prompts" / f"{name}.txt").read_text(encoding="utf-8")


def _render(name: str, **values: Any) -> str:
    return Template(_template(name)).substitute(values)


def audio_cues(state: DirectState, cfg: RunConfig) -> dict | None:
    """Audio shown to the model: the speech transcript, plus sound tags only when cfg.audio_tags is on."""
    if not cfg.use_audio or not state.audio:
        return None
    return state.audio if cfg.audio_tags else {**state.audio, "tags": []}


def common(state: DirectState, cfg: RunConfig) -> dict[str, str]:
    """Prompt blocks shared by every call that sees the clip: frame preamble, audio text and the option-free question."""
    return {"preamble": prompting.frame_preamble(observer.frame_ids_of(state), cfg.use_vp),
            "audio": prompting.audio_block(audio_cues(state, cfg), state.frames),
            "question_block": f"Question: {state.question_text}"}


def images(state: DirectState) -> tuple[str, ...]:
    return tuple(e["path"] for e in evidence_registry(state).values())


def generate(vlm, states: list[DirectState], requests: list[VLMRequest], max_tokens: int) -> tuple[list[DirectState], list[str | None]]:
    if not requests:
        return [], []
    started = time.perf_counter()
    replies = vlm.generate(requests, GenerationConfig(max_tokens=max_tokens))
    share = (time.perf_counter() - started) / len(requests)
    return [charge(s, share) for s in states], replies


def direct_stage(states: list[DirectState], cfg: RunConfig, services) -> list[DirectState]:
    requests = [VLMRequest(text=_render("direct_next", **common(s, cfg)), images=images(s)) for s in states]
    started = time.perf_counter()
    replies = services.vlm.generate_scored(requests, GenerationConfig(max_tokens=SENTENCE_MAX_TOKENS))
    share = (time.perf_counter() - started) / max(1, len(requests))
    return [replace(charge(s, share), predicted_event=(r[0] if r else "").strip(), generation_logprob=(r[1] if r else None))
            for s, r in zip(states, replies)]


def freeze(state: DirectState) -> DirectState:
    text = state.predicted_event.strip()
    if not text:
        return replace(state, status="parse_error", flags=state.flags + ("empty_prediction",))
    return replace(state, predicted_event=text, frozen_sha=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])


def match_stage(states: list[DirectState], services) -> list[DirectState]:
    """The only place the options are read: text-only, after the sentence is frozen."""
    live = [s for s in states if s.alive and s.input.options]
    matched, probs = matcher.match(services.vlm, live, [s.predicted_event for s in live])
    by_id = {}
    for s in states:
        if s.alive and not s.input.options:  # no options given: the sentence itself is the output, no matcher call
            record = {"choice": None, "option_probs": {}, "confidence": None, "evidence": {}, "next_event": s.predicted_event,
                      "evidence_sufficient": True, "matcher_confidence": None}
            by_id[s.input.sample_id] = replace(s, prediction=record, direct_prediction=record, matched=None)
    for s, dist in zip(matched, probs):
        if dist is None:
            by_id[s.input.sample_id] = replace(s, status="parse_error", flags=s.flags + ("match_failed",))
            continue
        record = matcher.build_prediction(dist, {"evidence": {}, "next_event": s.predicted_event, "evidence_sufficient": True})
        record = {**record, "matcher_confidence": record["confidence"]}
        by_id[s.input.sample_id] = replace(s, prediction=record, direct_prediction=record, matched=dist,
                                           choice_history=(record["choice"],))
    return [by_id.get(s.input.sample_id, s) for s in states]
