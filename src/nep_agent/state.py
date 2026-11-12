"""Per-sample state of NEPAgent: the routed Direct state plus the reasoning candidates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.schemas import SampleState
from src.stages.state import RoutedState, lift_to


@dataclass(frozen=True)
class NEPAgentState(RoutedState):
    reasoning: dict[str, Any] = field(default_factory=dict)  # variant -> {"record": parsed JSON, "candidate": matched prediction}
    final_source: str = "direct"


def lift(state: SampleState) -> NEPAgentState:
    return lift_to(NEPAgentState, state)
