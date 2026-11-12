"""Citable images of a sample: sampled frames, frames added by a tool, and enlarged crops share one id space."""
from __future__ import annotations

from typing import Any

from src.agent import tools
from src.schemas import SampleState


def evidence_registry(state: SampleState) -> dict[str, dict[str, Any]]:
    """Uniform, resampled and enlarged images all belong to the same citation space."""
    entries = {f["frame_id"]: f for f in state.frames}
    entries.update({e["frame_id"]: e for e in state.extra_evidence if e["kind"] == tools.ZOOM})
    return entries
