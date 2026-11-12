"""Batched VLM calls that update per-sample call counts, amortised time and error status."""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Callable, Sequence

from src.agent.parsing import extract_json
from src.models.vlm import GenerationConfig, VLMBackend, VLMRequest
from src.schemas import STATUS_PARSE_ERROR, SampleState

RETRY_HINT = "\n\nYour previous reply was not valid JSON. Return only the JSON object."


def charge(state: SampleState, seconds: float, calls: int = 1) -> SampleState:
    return replace(state, calls=state.calls + calls, elapsed=state.elapsed + seconds)


def generate_charged(
    vlm: VLMBackend,
    states: Sequence[SampleState],
    requests: Sequence[VLMRequest],
    cfg: GenerationConfig,
) -> tuple[list[SampleState], list[str | None]]:
    if not requests:
        return [], []
    started = time.perf_counter()
    replies = vlm.generate(requests, cfg)
    share = (time.perf_counter() - started) / len(requests)
    return [charge(state, share) for state in states], replies


def probs_charged(
    vlm: VLMBackend,
    states: Sequence[SampleState],
    requests: Sequence[VLMRequest],
    letters: Sequence[Sequence[str]],
) -> tuple[list[SampleState], list[dict[str, float] | None]]:
    if not requests:
        return [], []
    started = time.perf_counter()
    probs = vlm.option_probs(requests, letters)
    share = (time.perf_counter() - started) / len(requests)
    return [charge(state, share) for state in states], probs


def generate_validated(
    vlm: VLMBackend,
    states: Sequence[SampleState],
    requests: Sequence[VLMRequest],
    validators: Sequence[Callable[[Any], Any | None]],
    cfg: GenerationConfig,
) -> tuple[list[SampleState], list[Any | None], list[str | None]]:
    """Call the VLM, retry unparsable replies once, and mark remaining failures parse_error."""
    current, replies = generate_charged(vlm, states, requests, cfg)
    parsed = [validate(extract_json(reply)) for validate, reply in zip(validators, replies)]

    retry_index = [i for i, value in enumerate(parsed) if value is None]
    retry_requests = [replace(requests[i], text=requests[i].text + RETRY_HINT) for i in retry_index]
    retried, retry_replies = generate_charged(vlm, [current[i] for i in retry_index], retry_requests, cfg)
    for slot, index in enumerate(retry_index):
        current[index] = retried[slot]
        replies[index] = retry_replies[slot]
        parsed[index] = validators[index](extract_json(retry_replies[slot]))

    final = [
        replace(state, status=STATUS_PARSE_ERROR) if value is None else state
        for state, value in zip(current, parsed)
    ]
    return final, parsed, replies
