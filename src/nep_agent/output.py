"""NEPAgent trace adapter: the stage record (Direct kept apart) plus every reasoning candidate and the reported source."""
from src.stages.output import to_result as stage_result


def to_result(state, cfg):
    result = stage_result(state, cfg)
    result["trace"]["nep_agent"] = {"variants": cfg.reasoning_variants, "final_source": state.final_source, "reasoning": state.reasoning,
                                 "frames_in_prompt": len(state.frames), "extra_evidence": len(state.extra_evidence),
                                 "elapsed_sec": round(state.elapsed, 3), "stage_calls": dict(state.stage_calls)}
    return result
