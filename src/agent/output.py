"""Output: serialise the final state to the result schema (the timeline figure is drawn by src/agent/figure.py)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.schemas import STATUS_OK, RunConfig, SampleState

TOP_K = 3


def is_uncertain(state: SampleState, cfg: RunConfig) -> bool | None:
    """Low final confidence, an unresolvable gap, or every cited frame rejected. None when no options were matched."""
    if not state.prediction:
        return True
    if state.prediction["confidence"] is None:
        return None
    return (
        state.prediction["confidence"] < cfg.uncertain_tau
        or state.all_cited_unsupported
        or state.unresolvable
    )


def _top_k(prediction: dict[str, Any], state: SampleState) -> list[dict[str, Any]]:
    probs = prediction["option_probs"]
    texts = dict(state.input.options)
    ranked = sorted(probs, key=lambda letter: probs[letter], reverse=True)[:TOP_K]
    return [
        {
            "option": letter,
            "event": texts[letter],
            "evidence_events": prediction["evidence"].get(letter, {}).get("evidence_events", []),
            "evidence_frames": prediction["evidence"].get(letter, {}).get("evidence_frames", []),
        }
        for letter in ranked
    ]


def to_result(state: SampleState, cfg: RunConfig) -> dict[str, Any]:
    """Build the JSON record; a failed sample has a null prediction and counts as wrong."""
    prediction = state.prediction
    predicted: dict[str, Any] | None = None
    if prediction and state.status == STATUS_OK:
        predicted = {
            "answer": prediction["choice"],
            "option_probs": prediction["option_probs"],
            "top_k": _top_k(prediction, state),
            "confidence": prediction["confidence"],
            "next_event": prediction["next_event"],
            "uncertain": is_uncertain(state, cfg),
        }
    return {
        "sample_id": state.input.sample_id,
        "status": state.status,
        "analysis": {"event_chain": list(state.events), "motion_summary": None},
        "prediction": predicted,
        "trace": {
            "reflector_rounds": state.rounds,
            "vlm_calls": state.calls,
            "latency_sec": round(state.elapsed, 3),
            "frames_used": len(state.frames),
            "extra_frames": state.extra_frames,
            "invalid_citations": state.invalid_citations,
            "reflector_checks": list(state.checks),
            "evidence_gaps": list(state.evidence_gaps),
            "choice_history": list(state.choice_history),
            "flags": list(state.flags),
        },
        "frames": [
            {"frame_id": f["frame_id"], "t_sec": round(f["t_sec"], 3), "source": f["source"], "path": f["path"]}
            for f in state.frames
        ],
        "run_config": cfg.config_path,
    }


def write_result(result: dict[str, Any], output_dir: str | Path) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{result['sample_id']}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
