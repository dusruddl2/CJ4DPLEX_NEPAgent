"""Result adapter of the stages: the base record plus every citable image and one trace group per stage."""
from src.agent.output import to_result as base_result
from src.stages.evidence import evidence_registry


def to_result(state, cfg):
    result = base_result(state, cfg)
    images = evidence_registry(state)
    crops = [e for e in images.values() if e.get("kind") == "zoom"]
    result["frames"].extend(
        {"frame_id": e["frame_id"], "t_sec": round(e["t_sec"], 3), "source": "zoom", "path": e["path"]}
        for e in crops
    )
    trace = result["trace"]
    trace["frames_used"] = len(images)
    trace["extra_frames"] += len(crops)
    trace["evidence_reviews"] = list(state.verification_notes)
    trace["tool"] = {
        "direct_prediction": state.direct_prediction, "routed": state.routed, "plan": state.plan,
        "new_evidence_ids": list(state.new_evidence_ids), "revision_accepted": state.revision_accepted,
        "decision_reason": state.decision_reason,
    }
    trace["open"] = {
        "question_mode": cfg.question_mode, "question_text": state.question_text, "question_note": state.question_note,
        "tool_executed": bool(state.new_evidence_ids), "new_evidence_ids": list(state.new_evidence_ids), "final": state.final,
        "predicted_event": state.predicted_event, "frozen_sha": state.frozen_sha, "generation_logprob": state.generation_logprob,
        "matched": state.matched, "confidence_source": "mc_matcher",
    }
    selection = state.tool_selection
    trace["direct"] = {"direct_prediction": state.direct, "routing": state.routing,
                       "tool_selection": {k: v for k, v in selection.items() if k != "plan"} if selection else None,
                       "tool_observation": state.tool_observation, "stage_calls": dict(state.stage_calls)}
    return result
