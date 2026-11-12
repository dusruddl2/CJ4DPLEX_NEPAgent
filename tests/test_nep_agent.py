"""NEPAgent: Direct kept apart, one-call reasoning candidates, tool only on an in-clip request, safe fallback, no option leakage."""
import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from src.agent.pipeline import build_services
from src.nep_agent import core
from src.nep_agent.event_reasoning import parse_reasoning
from src.nep_agent.output import to_result
from src.models.vlm import MockVLM
from src.schemas import AgentInput, RunConfig, SampleState

NEP_AGENT_DIR = Path(__file__).parents[1] / "src" / "nep_agent"


def cfg(**changes):
    base = RunConfig(run_name="t", method="nep_agent", num_frames=32, use_audio=False, question_mode="conditioned", use_reflector=True, gate_tau=1.01,
                     store_all_candidates=True)
    return replace(base, **changes)


def sample(tmp_path):
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 48), "red").save(image)
    frames = tuple({"frame_id": f"#{i:02d}", "t_sec": float(i), "source": "uniform", "path": str(image), "raw_path": str(image), "caption": ""}
                   for i in (1, 2, 3))
    return SampleState(input=AgentInput("s", "prefix.mp4", "Given the premise event: 'He lifts the cup', which event is its most direct conclusion?",
                                       (("A", "He drinks the coffee"), ("B", "He leaves the room"), ("C", "He answers the phone"))), frames=frames)


class Backend(MockVLM):
    def __init__(self, needed=True, broken=()):
        super().__init__()
        self.needed, self.broken = needed, set(broken)

    def _reply(self, request):
        text = request.text
        assert "Options:" not in text and "He leaves the room" not in text, "a reasoning stage saw the options"
        if "Update the reasoning" in text:
            return json.dumps({"terminal_state": "cup at lips", "ongoing_action": "tilting", "causal_transition": "liquid flows",
                               "predicted_event": "He leaves the room after drinking.", "supporting_evidence": ["frame #03z"]})
        if "Reason about what happens next" in text:
            kind = "coe_causal" if "missing_evidence" in text else "coe" if "observed_events" in text else "causal"
            if kind in self.broken:
                return "no json"
            payload = {"observed_events": [{"first_frame": "#01", "last_frame": "#02", "description": "He lifts a cup."},
                                           {"first_frame": "#09", "last_frame": "#09", "description": "A woman waves."}],
                       "terminal_state": "cup near mouth", "ongoing_action": "lifting", "causal_transition": "cup reaches lips",
                       "predicted_event": "He drinks the coffee." if kind != "causal" else "He answers the phone.", "supporting_evidence": ["#02", "#77"],
                       "missing_evidence": {"needed": self.needed, "question": "is the cup tilted", "tool": "zoom", "where": ["#03"], "region": "center"}}
            return json.dumps(payload)
        assert "Describe, in one sentence" in text
        return "He leaves the room."

    def option_probs(self, requests, letters):
        self.requests.extend(requests)
        out = []
        for r in requests:
            assert r.images == () and "A model watched the beginning" in r.text
            pred = r.text.split('"')[1]  # the matcher prompt quotes the frozen prediction before listing the options
            out.append({"A": .1, "B": .8, "C": .1} if "leaves" in pred else {"A": .1, "B": .1, "C": .8} if "phone" in pred else {"A": .8, "B": .1, "C": .1})
        return out


def _run(tmp_path, model, config):
    return core.run([sample(tmp_path)], config, build_services(model, tmp_path / "cache"))[0]


def test_all_candidates_are_stored_and_direct_stays_the_answer(tmp_path):
    result = _run(tmp_path, Backend(), cfg(reasoning_variants="coe,causal,coe_causal,tool"))
    assert result.prediction["choice"] == "B" and result.direct["choice"] == "B" and result.final_source == "direct"
    r = result.reasoning
    assert r["coe"]["candidate"]["choice"] == "A" and r["causal"]["candidate"]["choice"] == "C" and r["coe_causal"]["candidate"]["choice"] == "A"
    assert r["tool"]["observation"]["status"] == "executed" and r["tool"]["observation"]["where"] == ["#03", "#03"]
    assert r["tool"]["candidate"]["choice"] == "B" and "#03z" in r["tool"]["record"]["supporting_evidence"]
    assert result.calls == 2 + 2 * 3 + 2  # direct+match, three reasoning calls+matches, update+match; the tool itself costs no call
    events = r["coe"]["record"]["observed_events"]
    assert events[0]["grounded"] and events[0]["start_time"] == 1.0 and events[0]["end_time"] == 2.0 and not events[1]["grounded"]
    assert r["coe"]["record"]["supporting_evidence"] == ["#02"]
    trace = to_result(result, cfg())["trace"]["nep_agent"]
    assert set(trace["reasoning"]) == {"coe", "causal", "coe_causal", "tool"}


def test_reporting_a_candidate_and_falling_back_when_it_failed(tmp_path):
    chosen = _run(tmp_path, Backend(), cfg(reasoning_variants="causal", final_answer="causal"))
    assert chosen.prediction["choice"] == "C" and chosen.final_source == "causal" and chosen.direct["choice"] == "B"
    failed = _run(tmp_path, Backend(broken={"causal"}), cfg(reasoning_variants="causal", final_answer="causal"))
    assert failed.prediction["choice"] == "B" and failed.final_source == "direct_fallback" and failed.reasoning["causal"]["candidate"] is None
    assert failed.reasoning["causal"]["raw_reply"] == "no json"


def test_no_tool_without_an_in_clip_request(tmp_path):
    result = _run(tmp_path, Backend(needed=False), cfg(reasoning_variants="tool"))
    assert "coe_causal" in result.reasoning  # the tool path needs the coe_causal request and adds it
    assert result.reasoning["tool"]["observation"]["status"] == "not_requested" and result.reasoning["tool"]["candidate"] is None
    assert result.calls == 4 and not result.new_evidence_ids


def test_timestamps_come_from_frames_and_cannot_leave_the_clip():
    frames = [{"frame_id": "#01", "t_sec": 0.5}, {"frame_id": "#02", "t_sec": 1.5}]
    rec = parse_reasoning({"observed_events": [{"first_frame": "#02", "last_frame": "#01", "description": "x"},
                                               {"first_frame": "frame #01", "last_frame": "#02", "description": "y"}],
                           "predicted_event": "z", "supporting_evidence": "#02"}, frames, ["#01", "#02"])
    assert [e["grounded"] for e in rec["observed_events"]] == [False, True] and rec["observed_events"][1]["end_time"] == 1.5
    assert parse_reasoning({"predicted_event": " "}, frames, []) is None


def test_empty_variant_list_is_the_direct_step_only(tmp_path):
    model = Backend()
    result = _run(tmp_path, model, cfg(reasoning_variants="", final_answer="direct"))
    assert result.calls == 2 and result.reasoning == {} and result.final_source == "direct"
    assert result.prediction["choice"] == "B" and result.prediction["next_event"] == "He leaves the room." and result.generation_logprob == -0.5
    assert result.question_text.endswith("what event is its most direct conclusion? Describe that event in one sentence.")
    assert result.stage_calls == {"direct_video": 1, "direct_match": 1}


def test_chain_free_reasoning_is_reported_and_falls_back_to_direct(tmp_path):
    class Model(Backend):
        def __init__(self, ok=True):
            super().__init__()
            self.ok = ok

        def _reply(self, request):
            if "Reason about what happens next" not in request.text:
                return super()._reply(request)
            assert "Options:" not in request.text and '"events"' not in request.text and '"terminal_state"' in request.text
            if not self.ok:
                return "not json"
            return json.dumps({"observed": {"terminal_state": "cup near mouth", "ongoing_action": "lifting", "supporting_evidence": ["frame #03", "#77"]},
                               "inferred": {"intermediate_states": ["cup at lips"], "causal_transition": "he tilts it", "predicted_event": "He drinks the coffee."}})

    config = cfg(reasoning_variants="causal_sft_nc", final_answer="causal_sft_nc")
    chosen = _run(tmp_path, Model(), config)
    record = chosen.reasoning["causal_sft_nc"]["record"]
    assert chosen.prediction["choice"] == "A" and chosen.final_source == "causal_sft_nc" and chosen.direct["choice"] == "B" and chosen.calls == 4
    assert record["supporting_evidence"] == ["#03"] and record["observed_events"] == [] and record["intermediate_states"] == ["cup at lips"]
    failed = _run(tmp_path, Model(ok=False), config)
    assert failed.prediction["choice"] == "B" and failed.final_source == "direct_fallback" and failed.calls == 3


def test_router_uses_sentence_logprob_not_matcher_confidence(tmp_path):
    routed = _run(tmp_path, Backend(), cfg(reasoning_variants="", gate_tau=0.7))
    assert routed.routing["routed"] is True and abs(routed.routing["generation_confidence"] - 0.60653) < 1e-3
    assert routed.routing["signal"] == "direct_sentence_mean_token_logprob"
    assert _run(tmp_path, Backend(), cfg(reasoning_variants="", gate_tau=0.5)).routing["routed"] is False  # exp(-0.5)=0.61 is not below 0.5


def test_trace_separates_the_records(tmp_path):
    config = cfg(reasoning_variants="causal")
    result = _run(tmp_path, Backend(), config)
    trace = to_result(result, config)["trace"]
    assert set(trace["direct"]) == {"direct_prediction", "routing", "tool_selection", "tool_observation", "stage_calls"}
    assert trace["direct"]["direct_prediction"]["event"] == "He leaves the room." and trace["nep_agent"]["final_source"] == "direct"
    assert trace["open"]["frozen_sha"] == hashlib.sha256(result.predicted_event.encode()).hexdigest()[:16]
    assert trace["open"]["confidence_source"] == "mc_matcher" and trace["tool"]["direct_prediction"]["choice"] == "B"


def test_run_batch_dispatches_the_agent_and_rejects_other_methods(tmp_path):
    from src.agent.pipeline import run_batch

    assert run_batch([], cfg(), build_services(Backend(), tmp_path / "cache")) == []
    with pytest.raises(ValueError):
        run_batch([], cfg(method="other"), build_services(Backend(), tmp_path / "cache"))


def test_unknown_variant_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, Backend(), cfg(reasoning_variants="coe,magic"))


@pytest.mark.parametrize("path", sorted(NEP_AGENT_DIR.glob("*.py")), ids=lambda p: p.name)
def test_agent_code_has_no_label_or_post_split_references(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef)) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(("src.eval", "src.data", "src.train")), node.module
        words = [node.id] if isinstance(node, ast.Name) else [node.attr] if isinstance(node, ast.Attribute) else \
            [node.value] if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings else []
        for word in words:
            low = word.lower()
            assert not any(s in low for s in ("future", "ground_truth", "groundtruth", "gt_")), f"{path.name}: {word!r}"
            assert low not in ("correct", "correct_index", "label_path"), f"{path.name}: {word!r}"


def test_prompts_never_mention_options():
    for prompt in sorted((NEP_AGENT_DIR / "prompts").glob("*.txt")):
        assert "options:" not in prompt.read_text(encoding="utf-8").lower(), prompt.name




class ToolBackend(Backend):
    def __init__(self, valid_selection=True):
        super().__init__()
        self.valid_selection = valid_selection

    def _reply(self, request):
        text = request.text
        if "A first pass predicted that next" in text:
            assert "Options:" not in text
            if not self.valid_selection:
                return "no json"
            return json.dumps({"observable_gap": True, "tool": "zoom", "where": ["#03", "#03"], "cue": "whether the cup is tilted", "region": "center",
                               "reason": "small object", "expected_evidence": "cup angle"})
        if "Reason about what happens next" in text and "Additional observations from re-examining the clip" in text:
            assert len(request.images) > 3  # the enlarged crops follow the clip frames
            return json.dumps({"observed": {"terminal_state": "cup at lips", "ongoing_action": "tilting", "supporting_evidence": ["#03"]},
                               "inferred": {"intermediate_states": ["liquid flows"], "causal_transition": "he swallows", "predicted_event": "He drinks the coffee."}})
        return super()._reply(request)


def test_tool_reason_selects_a_tool_then_reasons_on_the_new_evidence(tmp_path):
    result = _run(tmp_path, ToolBackend(), cfg(reasoning_variants="causal_sft_nc,tool_reason", final_answer="tool_reason"))
    entry = result.reasoning["tool_reason"]
    assert entry["selection"]["tool"] == "zoom" and entry["observation"]["status"] == "executed" and entry["variant"] == "causal_sft_nc"
    assert entry["candidate"]["choice"] == "A" and result.prediction["choice"] == "A" and result.final_source == "tool_reason"
    assert result.reasoning["causal_sft_nc"]["candidate"]["choice"] == "C" and result.direct["choice"] == "B"  # earlier candidates are untouched
    assert result.calls == 2 + 2 + 3  # direct+match, reasoning+match, selection+reasoning+match


def test_tool_reason_falls_back_to_the_end_of_the_clip_when_the_selection_is_invalid(tmp_path):
    result = _run(tmp_path, ToolBackend(valid_selection=False), cfg(reasoning_variants="causal_sft_nc,tool_reason"))
    entry = result.reasoning["tool_reason"]
    assert entry["selection"]["default_plan"] and entry["selection"]["tool"] == "terminal_state"


def test_preference_list_reports_the_first_available_candidate(tmp_path):
    preferred = cfg(reasoning_variants="causal_sft_nc,tool_reason", final_answer="tool_reason,causal_sft_nc")
    with_tool = _run(tmp_path, ToolBackend(), preferred)
    assert with_tool.final_source == "tool_reason" and with_tool.prediction["choice"] == "A" and with_tool.direct["choice"] == "B"
    assert with_tool.prediction["confidence"] == with_tool.reasoning["tool_reason"]["candidate"]["option_probs"]["A"] == .8  # not Direct's value

    class NoToolCandidate(ToolBackend):  # the reasoning after the tool returns no JSON, so tool_reason has no candidate
        def _reply(self, request):
            if "Reason about what happens next" in request.text and "Additional observations from re-examining the clip" in request.text:
                return "no json"
            return super()._reply(request)

    clip_only = _run(tmp_path, NoToolCandidate(), preferred)
    assert clip_only.reasoning["tool_reason"]["candidate"] is None and clip_only.reasoning["tool_reason"]["observation"]["status"] == "executed"
    assert clip_only.final_source == "causal_sft_nc" and clip_only.prediction["choice"] == "C" and clip_only.decision_reason == "causal_sft_nc_selected"

    class NoReasoning(ToolBackend):  # every reasoning call fails, so only Direct is left
        def _reply(self, request):
            return "no json" if "Reason about what happens next" in request.text else super()._reply(request)

    direct_only = _run(tmp_path, NoReasoning(), preferred)
    assert direct_only.final_source == "direct_fallback" and direct_only.prediction["choice"] == "B"
    assert direct_only.reasoning["causal_sft_nc"]["candidate"] is None and direct_only.reasoning["tool_reason"]["candidate"] is None


def test_preference_list_may_end_with_direct_and_must_name_computed_candidates(tmp_path):
    listed = _run(tmp_path, Backend(broken={"causal"}), cfg(reasoning_variants="causal", final_answer="causal,direct"))
    assert listed.final_source == "direct" and listed.prediction["choice"] == "B"
    for bad in ("tool_reason,causal", "", ","):
        with pytest.raises(ValueError):
            _run(tmp_path, Backend(), cfg(reasoning_variants="causal", final_answer=bad))


def test_without_options_the_matcher_is_skipped_and_the_sentence_is_the_output(tmp_path):
    from scripts.run_agent import make_sample, summary

    class Model(ToolBackend):
        def option_probs(self, requests, letters):
            raise AssertionError("the matcher must not be called without options")

    config = cfg(reasoning_variants="causal_sft_nc,tool_reason", final_answer="tool_reason,causal_sft_nc")
    given = sample(tmp_path)
    state = replace(given, input=make_sample("s", "prefix.mp4"))
    assert state.input.options == () and state.input.question == "What happens next, right after the clip ends?"
    done = core.run([state], config, build_services(Model(), tmp_path / "cache"))[0]
    result = to_result(done, config)
    assert result["status"] == "ok" and result["prediction"]["answer"] is None and result["prediction"]["option_probs"] == {}
    assert result["prediction"]["next_event"] == "He drinks the coffee." and result["prediction"]["confidence"] is None and result["prediction"]["uncertain"] is None
    assert done.final_source == "tool_reason" and done.calls == 4  # direct, reasoning, selection, reasoning on the new evidence
    shown = summary(result)
    assert shown["final_source"] == "tool_reason" and shown["reasoning"]["predicted_event"] == "He drinks the coffee." and shown["tool"]["selection"]["tool"] == "zoom"


GATED = dict(store_all_candidates=False, reasoning_variants="causal_sft_nc,tool_reason", final_answer="tool_reason,causal_sft_nc")  # Direct confidence is exp(-0.5) = 0.61


def _kinds(model):
    """What each generation request was, in order."""
    kind = lambda t: "select" if "A first pass predicted that next" in t else "tool_reason" if "Additional observations from re-examining the clip" in t \
        else "reason" if "Reason about what happens next" in t else "direct" if "Describe, in one sentence" in t else "match"
    return [kind(r.text) for r in model.requests]


def test_gate_keeps_a_confident_direct_answer_without_further_calls(tmp_path):
    model = ToolBackend()
    result = _run(tmp_path, model, cfg(gate_tau=0.5, **GATED))
    assert _kinds(model) == ["direct", "match"] and result.calls == 2
    assert result.routing["routed"] is False and result.final_source == "direct" and result.reasoning == {} and result.tool_selection is None
    assert result.prediction["choice"] == "B" and not result.new_evidence_ids


def test_gate_routes_an_unconfident_item_to_the_tool_and_skips_clip_only_reasoning(tmp_path):
    model = ToolBackend()
    result = _run(tmp_path, model, cfg(gate_tau=0.75, **GATED))
    assert _kinds(model) == ["direct", "match", "select", "tool_reason", "match"] and result.calls == 5
    assert result.routing["routed"] is True and result.final_source == "tool_reason" and set(result.reasoning) == {"tool_reason"}
    assert result.prediction["choice"] == "A" and result.prediction["confidence"] == .8 and result.reasoning["tool_reason"]["variant"] == "causal_sft_nc"


def test_gate_falls_back_to_clip_only_reasoning_when_the_tool_adds_nothing(tmp_path, monkeypatch):
    from src.stages import tool_select

    monkeypatch.setattr(tool_select, "execute", lambda state, plan, cfg, io: (state, []))
    model = ToolBackend()
    result = _run(tmp_path, model, cfg(gate_tau=0.75, **GATED))
    assert _kinds(model) == ["direct", "match", "select", "reason", "match"] and result.calls == 5
    assert result.reasoning["tool_reason"]["observation"]["status"] == "no_new_evidence" and result.reasoning["tool_reason"]["candidate"] is None
    assert result.final_source == "causal_sft_nc" and result.prediction["choice"] == "C"


def test_gate_runs_clip_only_reasoning_on_the_original_frames_after_a_failed_tool_answer(tmp_path):
    class BrokenAfterTool(ToolBackend):
        def _reply(self, request):
            if "Reason about what happens next" in request.text and "Additional observations from re-examining the clip" in request.text:
                return "no json"
            return super()._reply(request)

    model = BrokenAfterTool()
    result = _run(tmp_path, model, cfg(gate_tau=0.75, **GATED))
    assert _kinds(model) == ["direct", "match", "select", "tool_reason", "reason", "match"]
    clip_only = [r for r in model.requests if "Reason about what happens next" in r.text][-1]
    assert len(clip_only.images) == 3 and "Additional observations" not in clip_only.text  # the crops of the tool are not shown
    assert result.final_source == "causal_sft_nc" and result.extra_evidence and result.reasoning["tool_reason"]["observation"]["status"] == "executed"


def test_gate_keeps_direct_when_every_later_stage_fails(tmp_path):
    class NoReasoning(ToolBackend):
        def _reply(self, request):
            return "no json" if "Reason about what happens next" in request.text else super()._reply(request)

    model = NoReasoning()
    result = _run(tmp_path, model, cfg(gate_tau=0.75, **GATED))
    assert _kinds(model) == ["direct", "match", "select", "tool_reason", "reason"]
    assert result.final_source == "direct_fallback" and result.prediction["choice"] == "B" and result.alive


def test_gate_works_without_options(tmp_path):
    from scripts.run_agent import make_sample

    class Model(ToolBackend):
        def option_probs(self, requests, letters):
            raise AssertionError("the matcher must not be called without options")

    for tau, source, calls in ((0.5, "direct", 1), (0.75, "tool_reason", 3)):
        state = replace(sample(tmp_path), input=make_sample("s", "prefix.mp4"))
        done = core.run([state], cfg(gate_tau=tau, **GATED), build_services(Model(), tmp_path / "cache"))[0]
        assert done.final_source == source and done.calls == calls and to_result(done, cfg(**GATED))["prediction"]["answer"] is None


def test_gated_mode_rejects_the_update_variant(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, Backend(), cfg(store_all_candidates=False, reasoning_variants="coe_causal,tool", final_answer="tool"))


def test_final_config_is_gated_and_ablation_configs_store_everything():
    from src.agent.config import load_config

    final = load_config("configs/nep_agent.yaml")
    assert (final.gate_tau, final.store_all_candidates, final.final_answer, final.use_reflector) == (0.75, False, "tool_reason,causal_sft_nc", True)
    assert (final.reason_audio, final.tool_menu, final.tool_note) == ("none", "dense_resample,zoom,terminal_state", False)
    stored = load_config("configs/nep_agent_all_candidates.yaml")
    assert (stored.reason_audio, stored.tool_menu) == (final.reason_audio, final.tool_menu)
    for name in ("nep_agent_all_candidates", "ablation/tools_only_terminal", "ablation/tools_only_dense", "ablation/tools_only_zoom"):
        ablation = load_config(f"configs/{name}.yaml")
        assert ablation.store_all_candidates and ablation.gate_tau == 1.01, name


def test_failed_reasoning_reply_is_stored_in_full(tmp_path):
    long_reply = "not json " * 400  # 3,600 characters

    class Model(Backend):
        def _reply(self, request):
            return long_reply if "Reason about what happens next" in request.text else super()._reply(request)

    result = _run(tmp_path, Model(), cfg(reasoning_variants="causal"))
    assert result.reasoning["causal"]["raw_reply"] == long_reply and result.reasoning["causal"]["record"] is None
    assert _run(tmp_path, Backend(), cfg(reasoning_variants="causal")).reasoning["causal"]["raw_reply"] == ""  # empty when the reply parsed
    bad_selection = "x" * 2000

    class Selector(ToolBackend):
        def _reply(self, request):
            return bad_selection if "A first pass predicted that next" in request.text else super()._reply(request)

    from src.stages.tool_select import select_stage
    from src.nep_agent.state import lift

    state = replace(lift(sample(tmp_path)), question_text="q", direct={"event": "He leaves the room."})
    selected = select_stage([state], cfg(), build_services(Selector(), tmp_path / "cache"))[0]
    assert selected.tool_selection == {"status": "invalid", "raw": bad_selection}


def test_audio_tool_observation_carries_the_text_it_added(tmp_path):
    from src.nep_agent.state import lift
    from src.stages.tool_select import parse_selection, run_tool

    audio = {"asr": [{"t_start": 1.0, "t_end": 2.0, "text": "watch the cup"}], "tags": []}
    config = cfg(use_audio=True)
    state = replace(lift(sample(tmp_path)), audio=audio)
    chosen = parse_selection({"observable_gap": True, "tool": "audio_segment", "where": ["#01", "#02"], "cue": "the spoken line"}, state, config)
    silent = run_tool(replace(state, tool_selection=chosen), config, build_services(Backend(), tmp_path / "cache"))
    assert silent.tool_observation["status"] == "no_new_evidence" and "evidence_text" not in silent.tool_observation  # already in the prompt
    from src.agent import prompting
    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as patch:  # a line the capped audio prompt did not show
        patch.setattr(prompting, "audio_block", lambda audio, frames: "Audio: not shown")
        done = run_tool(replace(state, tool_selection=chosen), config, build_services(Backend(), tmp_path / "cache"))
    assert done.tool_observation["status"] == "executed" and done.tool_observation["evidence_text"] == 'speech: "watch the cup"'
    assert done.tool_observation["new_evidence_ids"] == ["audio:#01-#02"]


def test_single_video_summary_says_why_direct_was_kept_and_the_figure_is_drawn(tmp_path):
    from scripts.run_agent import decision, make_sample, summary
    from src.agent.figure import describe, draw_timeline

    kept_cfg, routed_cfg = cfg(gate_tau=0.5, **GATED), cfg(gate_tau=0.75, **GATED)
    kept = to_result(_run(tmp_path, ToolBackend(), kept_cfg), kept_cfg)
    assert decision(kept) == "Direct answer kept: its confidence 0.61 is at or above the threshold 0.5; no tool or reasoning call was made"
    assert summary(kept)["decision"] == decision(kept) and summary(kept)["reasoning"] is None
    routed = to_result(_run(tmp_path, ToolBackend(), routed_cfg), routed_cfg)
    assert decision(routed) == "Direct confidence 0.61 is below the threshold 0.75; reported answer: tool_reason"
    text = "\n".join(describe(routed))
    for expected in ("Direct prediction: He leaves the room.", "Direct confidence: 0.61 (threshold 0.75", "Tool: zoom on frames #03 to #03 (executed; added #03z)",
                     "Terminal state: cup at lips", "Causal transition: he swallows", "Predicted next event: He drinks the coffee.",
                     "Reported candidate: tool_reason", "A 0.80  He drinks the coffee"):
        assert expected in text, expected
    assert str(tmp_path) not in text  # no file paths in the figure text
    for name, result in (("kept", kept), ("routed", routed)):
        target = tmp_path / f"{name}.png"
        draw_timeline(result, target)
        assert target.stat().st_size > 5000
    state = replace(sample(tmp_path), input=make_sample("s", "prefix.mp4"))  # option-free: the figure has no option block
    free = to_result(core.run([state], routed_cfg, build_services(ToolBackend(), tmp_path / "cache"))[0], routed_cfg)
    assert "Options" not in "\n".join(describe(free))
    draw_timeline(free, tmp_path / "free.png")
    missing = {**routed, "frames": [{**f, "path": "cache/absent.jpg"} for f in routed["frames"]]}  # frames not on this machine
    draw_timeline(missing, tmp_path / "missing.png")
    assert (tmp_path / "free.png").exists() and (tmp_path / "missing.png").exists()


def test_prompt_variants_change_only_the_reasoning_prompt(tmp_path, monkeypatch):
    from src.stages import direct as direct_step

    real_common = direct_step.common
    monkeypatch.setattr(direct_step, "common", lambda state, c: {**real_common(state, c), "audio": "Audio: speech line."})

    class Rec(ToolBackend):
        def _reply(self, request):
            self.texts = getattr(self, "texts", []) + [request.text]
            return super()._reply(request)

    model = Rec()
    _run(tmp_path, model, cfg(reasoning_variants="causal_sft_nc,tool_reason", reason_audio="none", tool_note=True))
    reasoning = [t for t in model.texts if "Reason about what happens next" in t]
    assert reasoning and all("Audio: not available." in t and "speech line" not in t for t in reasoning)
    assert any("Frames added by re-examining the clip" in t for t in reasoning)
    assert all("speech line" in t for t in model.texts if "Describe, in one sentence" in t or "A first pass predicted" in t)  # Direct and selection keep it
    default = Rec()
    _run(tmp_path, default, cfg(reasoning_variants="causal_sft_nc,tool_reason"))
    assert all("speech line" in t for t in default.texts if "Reason about what happens next" in t)
    assert not any("Frames added by re-examining" in t for t in default.texts)


def test_gated_clip_only_fallback_uses_the_same_reasoning_audio(tmp_path, monkeypatch):
    from src.stages import direct as direct_step, tool_select

    real_common = direct_step.common
    monkeypatch.setattr(direct_step, "common", lambda state, c: {**real_common(state, c), "audio": "Audio: speech line."})
    monkeypatch.setattr(tool_select, "execute", lambda state, plan, cfg, io: (state, []))  # the tool adds nothing: clip-only fallback runs
    model = ToolBackend()
    result = _run(tmp_path, model, cfg(gate_tau=0.75, reason_audio="none", **GATED))
    assert result.final_source == "causal_sft_nc" and _kinds(model) == ["direct", "match", "select", "reason", "match"]
    clip_only = [r.text for r in model.requests if "Reason about what happens next" in r.text]
    assert len(clip_only) == 1 and "Audio: not available." in clip_only[0] and "speech line" not in clip_only[0]
    assert "speech line" in model.requests[0].text  # Direct keeps the audio text


def test_tool_menu_limits_the_selection_prompt_and_the_plans(tmp_path):
    from src.agent import tools
    from src.nep_agent.state import lift
    from src.stages.tool_select import parse_selection

    state = replace(lift(sample(tmp_path)), audio={"asr": [{"t_start": 1.0, "t_end": 2.0, "text": "hi"}], "tags": []})
    every = cfg(use_audio=True)
    limited = cfg(use_audio=True, tool_menu="dense_resample,zoom,terminal_state")
    assert tools.available_tools(state, every) == ("dense_resample", "zoom", "terminal_state", "audio_segment")
    assert tools.available_tools(state, limited) == ("dense_resample", "zoom", "terminal_state")
    assert tools.available_tools(state, cfg(use_audio=True, tool_menu=" terminal_state ")) == ("terminal_state",)
    request = {"observable_gap": True, "tool": "audio_segment", "where": ["#01", "#02"], "cue": "the spoken line"}
    assert parse_selection(request, state, every) is not None and parse_selection(request, state, limited) is None

    model = ToolBackend()
    _run(tmp_path, model, cfg(gate_tau=0.75, tool_menu="dense_resample,zoom,terminal_state", **GATED))
    selection = next(r.text for r in model.requests if "A first pass predicted that next" in r.text)
    assert "zoom:" in selection and "audio_segment" not in selection

