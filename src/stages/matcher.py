"""Text-only matcher: map a predicted sentence onto the answer options. The only stage that reads the options."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Sequence

from src.agent import prompting
from src.agent.vlm_ops import probs_charged
from src.models.vlm import VLMRequest
from src.schemas import SampleState


@lru_cache(maxsize=None)
def _template(name: str) -> str:
    return (Path(__file__).parent / "prompts" / f"{name}.txt").read_text(encoding="utf-8")


def match_request(state: SampleState, text: str) -> VLMRequest:
    """Text-only: map the sentence onto the options. No images, no video, no labels."""
    return VLMRequest(text=Template(_template("match")).safe_substitute(
        prediction=text.strip(), question_block=prompting.question_block(state.input)), images=())


def match(vlm, states: Sequence[SampleState], texts: Sequence[str]) -> tuple[list[SampleState], list[dict[str, float] | None]]:
    return probs_charged(vlm, states, [match_request(s, t) for s, t in zip(states, texts)], [s.input.letters for s in states])


def build_prediction(probs: dict[str, float], evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create the internal prediction record from option probabilities."""
    choice = max(probs, key=lambda letter: probs[letter])
    return {
        "choice": choice,
        "option_probs": probs,
        "confidence": probs[choice],
        "evidence": evidence["evidence"] if evidence else {},
        "next_event": evidence["next_event"] if evidence else "",
        "evidence_sufficient": evidence["evidence_sufficient"] if evidence else True,
    }
