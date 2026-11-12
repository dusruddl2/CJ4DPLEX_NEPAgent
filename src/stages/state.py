"""Per-sample state of the pipeline stages. Each class adds the fields its stage writes."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, TypeVar

from src.schemas import SampleState

T = TypeVar("T", bound=SampleState)


@dataclass(frozen=True)
class ToolState(SampleState):
    """Matcher record of the Direct sentence and the bookkeeping of one re-observation."""
    direct_prediction: dict[str, Any] | None = None
    routed: bool = False
    plan: dict[str, Any] | None = None
    new_evidence_ids: tuple[str, ...] = ()
    revision_accepted: bool = False
    decision_reason: str = "direct"


@dataclass(frozen=True)
class DirectState(ToolState):
    """Option-free question, the predicted sentence, its frozen hash and the matcher distribution."""
    question_text: str = ""
    question_note: dict[str, Any] | None = None
    final: dict[str, Any] | None = None
    predicted_event: str = ""
    frozen_sha: str = ""
    generation_logprob: float | None = None
    matched: dict[str, float] | None = None


@dataclass(frozen=True)
class RoutedState(DirectState):
    """Direct candidate kept apart, the routing signal, the selected tool with its outcome, and per-stage call counts."""
    direct: dict[str, Any] | None = None
    routing: dict[str, Any] | None = None
    tool_selection: dict[str, Any] | None = None
    tool_observation: dict[str, Any] | None = None
    stage_calls: dict[str, int] = field(default_factory=dict)


def lift_to(cls: type[T], state: SampleState) -> T:
    """Copy `state` into `cls`, keeping every field both classes share."""
    if isinstance(state, cls):
        return state
    known = {f.name for f in fields(cls)}
    return cls(**{f.name: getattr(state, f.name) for f in fields(type(state)) if f.name in known})
