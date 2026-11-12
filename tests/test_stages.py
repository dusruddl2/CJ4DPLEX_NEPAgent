"""Stages: option-free questions, the text-only matcher and re-observation windows."""
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from src.models.vlm import MockVLM
from src.schemas import AgentInput, RunConfig, SampleState
from src.stages import matcher, window
from src.stages.questions import GENERIC_QUESTION, normalize_question
from src.stages.state import DirectState, RoutedState, ToolState, lift_to

STAGES_DIR = Path(__file__).parents[1] / "src" / "stages"


def cfg(**changes):
    return replace(RunConfig(run_name="stages", num_frames=32, use_audio=False), **changes)


def sample(tmp_path, count=2):
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 48), "red").save(image)
    frames = tuple({"frame_id": f"#{i:02d}", "t_sec": float(i), "source": "uniform",
                    "path": str(image), "raw_path": str(image), "caption": ""} for i in range(1, count + 1))
    return SampleState(input=AgentInput("s", "prefix.mp4", "What next?", (("A", "one"), ("B", "two"), ("C", "three"))), frames=frames)


def plan(**changes):
    return {"sufficient": False, "resolvable": True, "gap_scope": "observed", "missing": "visible handle position",
            "tool": "zoom", "where": ["#01", "#01"], "region": "center", **changes}


def test_question_rewrites():
    fo = normalize_question("Given the premise event: 'X', which event is its most direct conclusion?", "conditioned")
    assert fo["text"] == "Given the premise event: 'X', what event is its most direct conclusion? Describe that event in one sentence."
    assert fo["rules"] == ["which_event_conclusion"]
    fb = normalize_question("Based on the given video, predict future events and fill in the potential events in the following sequence: 1. [?] 2. A man leaves.", "conditioned")
    assert fb["slots"] == 1 and fb["text"].endswith("Describe the event for the first [?] slot in one sentence.")
    assert fb["observer_text"] == "Based on the given video, predict future events and fill in the potential events in the following sequence:"
    assert "A man leaves" not in fb["observer_text"] and fo["observer_text"] == fo["text"]
    assert normalize_question("Which of the following happens next?", "conditioned")["text"] == "what happens next?"
    assert normalize_question("anything", "generic")["text"] == GENERIC_QUESTION
    with pytest.raises(ValueError):
        normalize_question("q", "other")


def test_matcher_is_text_only_and_quotes_the_stripped_sentence(tmp_path):
    state = sample(tmp_path)
    request = matcher.match_request(state, "  He opens the door. ")
    assert request.images == () and '"He opens the door."' in request.text
    assert "Options:\nA. one\nB. two\nC. three" in request.text and "$" not in request.text
    model = MockVLM()
    charged, probs = matcher.match(model, [state], ["He opens the door."])
    assert charged[0].calls == 1 and set(probs[0]) == {"A", "B", "C"} and model.requests[-1].images == ()
    record = matcher.build_prediction({"A": .2, "B": .7, "C": .1})
    assert record["choice"] == "B" and record["confidence"] == .7 and record["next_event"] == ""


def test_lift_keeps_shared_fields_and_adds_defaults(tmp_path):
    state = replace(sample(tmp_path), calls=3, flags=("x",))
    lifted = lift_to(RoutedState, lift_to(DirectState, state))
    assert isinstance(lifted, ToolState) and lifted.calls == 3 and lifted.flags == ("x",) and lifted.frames == state.frames
    assert lifted.direct is None and lifted.stage_calls == {} and lifted.decision_reason == "direct"
    assert lift_to(RoutedState, lifted) is lifted


def test_normalize_where_accepts_single_frame_prefixed_and_enumerated_windows():
    assert window.normalize_where({"where": ["#32"]})["where"] == ["#32", "#32"]
    assert window.normalize_where({"where": ["frame #31", "frame #32"]})["where"] == ["#31", "#32"]
    assert window.normalize_where({"where": ["frame #23", "frame #24", "frame #25", "frame #28"]})["where"] == ["#23", "#28"]
    assert window.normalize_where({"where": "#07"})["where"] == ["#07", "#07"]
    assert window.normalize_where({"where": ["#12.1", "#13"]})["where"] == ["#12.1", "#13"]
    assert window.normalize_where({"cue": "x"}) == {"cue": "x"}


def test_wide_window_is_clamped_to_the_last_eight_frames():
    ids = [f"#{i:02d}" for i in range(1, 33)]
    out = window.clamp_window({"tool": "dense_resample", "where": ["#21", "#32"], "cue": "c"}, ids)
    assert out["where"] == ["#25", "#32"] and out["clamped_from"] == ["#21", "#32"]
    assert window.clamp_window({"where": ["#30", "#32"]}, ids) == {"where": ["#30", "#32"]}
    assert window.clamp_window({"where": ["#01", "#32"]}, ids)["where"] == ["#25", "#32"]


def test_window_plan_requires_cue_and_short_window():
    ids = [f"#{i:02d}" for i in range(1, 33)]
    menu = ["dense_resample", "zoom"]
    assert window.parse_window_plan({"cue": "hand reaching for the lid", "where": ["#20", "#24"], "tool": "dense_resample"}, ids, menu)["missing"] == "hand reaching for the lid"
    assert window.parse_window_plan({"cue": "", "where": ["#20", "#24"], "tool": "dense_resample"}, ids, menu) is None
    assert window.parse_window_plan({"cue": "x", "where": ["#01", "#32"], "tool": "dense_resample"}, ids, menu) is not None  # width is checked by validate_plan
    assert window.parse_window_plan({"cue": "x", "where": ["#24", "#20"], "tool": "dense_resample"}, ids, menu) is None
    assert window.parse_window_plan({"cue": "x", "where": ["#20", "#24"], "tool": "terminal_state"}, ids, menu) is None


def test_window_plan_accepts_positions_without_vp():
    ids = [f"#{i:02d}" for i in range(1, 33)]
    parsed = window.parse_window_plan({"cue": "lid being lifted", "where": ["20", 24], "tool": "dense_resample"}, ids, ["dense_resample"])
    assert parsed["where"] == ["#20", "#24"]
    assert window.parse_window_plan({"cue": "x", "where": ["0", "5"], "tool": "dense_resample"}, ids, ["dense_resample"]) is None


def test_template_echo_and_wide_window_plans_are_rejected(tmp_path):
    state = lift_to(ToolState, sample(tmp_path, count=32))
    assert window.validate_plan(plan(missing="observable fact to check", where=["#03", "#06"]), state) is None
    assert window.validate_plan(plan(where=["#01", "#32"]), state) is None
    assert window.validate_plan(plan(where=["#10", "#17"]), state) is not None
    assert window.validate_plan(plan(where=["#10", "#18"]), state) is None
    assert window.validate_plan({"sufficient": True, "missing": "", "resolvable": False}, state) is not None


def test_zoom_adds_a_citable_crop_and_reformatted_audio_is_not_new_evidence(tmp_path):
    from src.agent.observer import FrameIO

    state = lift_to(ToolState, sample(tmp_path))
    zoomed, new_ids = window.execute(state, plan(), cfg(), FrameIO(tmp_path / "cache"))
    assert new_ids == ["#01z"] and zoomed.extra_evidence[0]["parent"] == "#01" and Path(zoomed.extra_evidence[0]["path"]).exists()
    spoken = replace(state, audio={"asr": [{"t_start": 0, "t_end": 2, "text": "visible speech"}], "tags": []})
    _, audio_ids = window.execute(spoken, plan(tool="audio_segment", where=["#01", "#02"]), cfg(use_audio=True), None)
    assert audio_ids == []


def test_direct_prompt_never_mentions_options():
    text = (STAGES_DIR / "prompts" / "direct_next.txt").read_text(encoding="utf-8").lower()
    assert "$options" not in text and "options:" not in text and "option" not in text
