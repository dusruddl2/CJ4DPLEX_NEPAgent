"""NEPAgent training pieces that need no GPU: source parsing, Teacher validation and hindsight checks, SFT target schema."""
import json

from src.train import build_sft_reasoning as b

ROW = {"sample_id": "nep_0001", "video": "v.mp4", "status": "ok", "record": {"images": [f"/f/{i:03d}.jpg" for i in range(1, 33)],
       "messages": [{"role": "user", "content": "<image>Please examine frames.\nFill in: 1. [?] 2. The end. Which option best fits?"},
                    {"role": "assistant", "content": "The woman throws a frisbee."}]}}
OBS = {"observed_events": [{"first_frame": "#03", "last_frame": "#05", "description": "a woman picks up a frisbee"}],
       "terminal_state": "She holds the frisbee.", "ongoing_action": "raising her arm", "supporting_evidence": {"terminal_state": ["#32"], "ongoing_action": ["#31"]}}


def teacher_row(bridge, observe=OBS):
    return {**b.parse_nep_source(ROW), "teacher_observe_raw": json.dumps(observe), "teacher_bridge_raw": json.dumps(bridge)}


def test_input_prompt_hides_options_and_has_no_tool_field():
    item = b.parse_nep_source(ROW)
    assert item["gold_text"] == "The woman throws a frisbee."
    prompt = b.input_prompt(item)
    assert "frisbee" not in prompt and "Options:" not in prompt and "missing_evidence" not in prompt and '"observed"' in prompt


def test_stage1_facts_are_kept_verbatim_and_target_separates_observed_and_inferred():
    target, status, flags = b.decide(teacher_row({"intermediate_states": ["her arm swings forward"], "causal_transition": "the disc leaves her hand",
                                                  "predicted_event": "She throws the frisbee."}))
    assert status == "ok" and target["observed"]["events"] == OBS["observed_events"] and target["observed"]["terminal_state"] == "She holds the frisbee."
    assert target["observed"]["supporting_evidence"] == ["#31", "#32"] and target["inferred"]["predicted_event"] == "The woman throws a frisbee."
    assert "missing_evidence" not in json.dumps(target) and not flags["retro_reference"]


def test_hindsight_checks_reject_retro_references_and_unsupported_content():
    _, status, _ = b.decide(teacher_row({"intermediate_states": ["as shown in frame #30 she had already thrown it"], "causal_transition": "c"}))
    assert status == "hindsight_retro_reference"
    _, status, flags = b.decide(teacher_row({"intermediate_states": ["zebras gallop beside glacier volcanoes"], "causal_transition": "submarines surface quickly"}))
    assert status == "ok" and flags["unsupported_content_flag"] and flags["unsupported_word_ratio"] > b.UNSUPPORTED_TAU  # flagged for the audit, not rejected


def test_stage1_needs_grounded_events_and_a_terminal_state():
    _, status, _ = b.decide(teacher_row({"causal_transition": "c"}, {**OBS, "observed_events": [{"first_frame": "#40", "last_frame": "#41", "description": "x"}]}))
    assert status == "no_grounded_event"
    _, status, _ = b.decide(teacher_row({"causal_transition": "c"}, {**OBS, "terminal_state": ""}))
    assert status == "no_terminal_state"


def test_frame_id_terminal_state_is_rejected_and_events_are_trimmed_to_the_end():
    _, status, _ = b.decide(teacher_row({"causal_transition": "c"}, {**OBS, "terminal_state": "frame #32"}))
    assert status == "no_terminal_state"
    many = [{"first_frame": f"#{i:02d}", "last_frame": f"#{i:02d}", "description": f"step {i}"} for i in range(1, 9)]
    target, status, flags = b.decide(teacher_row({"intermediate_states": [], "causal_transition": "she swings"}, {**OBS, "observed_events": many}))
    assert status == "ok" and [e["first_frame"] for e in target["observed"]["events"]] == ["#04", "#05", "#06", "#07", "#08"] and flags["events_trimmed"] == 3
    _, status, _ = b.decide(teacher_row({"intermediate_states": ["the frame shows her turning"], "causal_transition": "c"}))
    assert status == "ok"  # the plain word "frame" is not a retro reference


def test_inferred_states_that_copy_the_terminal_state_or_the_answer_are_dropped():
    target, status, flags = b.decide(teacher_row({"intermediate_states": ["She holds the frisbee.", "The woman throws a frisbee.", "her arm swings back"],
                                                  "causal_transition": "the disc leaves her hand"}))
    assert status == "ok" and target["inferred"]["intermediate_states"] == ["her arm swings back"] and flags["inferred_copies_dropped"] == 2
    _, status, _ = b.decide(teacher_row({"intermediate_states": ["She holds the frisbee."], "causal_transition": "c"}))
    assert status == "inferred_only_copies"


def test_assemble_keeps_one_item_per_video_and_audits_every_row():
    row = teacher_row({"intermediate_states": [], "causal_transition": "she swings", "predicted_event": "x"})
    records, counts, audit = b.assemble([row, {**row, "sample_id": "nep_0002"}])
    assert len(records) == 1 and counts["duplicate_video"] == 1 and len(audit) == 1 and len(records[0]["images"]) == 32


def test_nep_source_reads_answer_only_sft_items_in_order(tmp_path):
    """The NEPAgent SFT uses exactly the answer-only SFT items; question and target come from that record."""
    rows = []
    for sid in ("nep_a", "nep_b"):
        images = [f"/x/frames_nep/{sid}/fit/{i:03d}.jpg" for i in range(1, 4)]
        rows.append({"sample_id": sid, "status": "ok", "video": f"{sid}.mp4", "source": "youtube",
                     "record": {"messages": [{"role": "user", "content": "<image>Please examine frames.\nWhat happens next?"},
                                             {"role": "assistant", "content": f"Caption for {sid}."}], "images": images}})
    source = tmp_path / "records.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    train = tmp_path / "train.json"
    train.write_text(json.dumps([{"images": rows[1]["record"]["images"]}, {"images": rows[0]["record"]["images"]}]), encoding="utf-8")
    items = b.nep_items(str(source), str(train))
    assert [i["sample_id"] for i in items] == ["nep_b", "nep_a"]
    assert items[0]["gold_text"] == "Caption for nep_b." and items[0]["question"] == "What happens next?" and items[0]["options"] == []
