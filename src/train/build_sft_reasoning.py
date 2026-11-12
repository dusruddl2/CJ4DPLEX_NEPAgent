"""NEPAgent SFT data: a two-stage Teacher writes the reasoning targets; observed facts and inferred states come from separate passes.

Source items: the items of the answer-only SFT set (`src/train/build_sft_nep.py`: its `records.jsonl`, in the order of its
`train.json`), so both trainings see the same clips. Charades/Ego4D and every evaluation video are already removed there;
32 VP frames each. Teacher: Qwen3-VL-8B-Instruct via vLLM, temperature 0.

  Stage 1 (answer NOT shown): observed events with first/last frame ids, terminal state, ongoing action, and the frames that
          support them. Seconds are computed from the frame ids, never written by the model.
  Stage 2 (answer shown): only what comes after the clip: up to 3 hypothesized intermediate states, the causal transition and
          the predicted event. Stage 1 facts are passed in as fixed and are copied into the target unchanged.
Hindsight checks (`hindsight_flags`): stage 2 cites frames or refers back to the clip ("in the clip", "earlier", "already"...);
stage 2 introduces many content words found neither in the stage-1 facts, the question nor the answer; stage 2's restated
event drifts from the answer. The stage-1 / stage-2 split is the main guard: the observed part of every target was written
without the answer. `scripts/teacher_quality.py` audits a 100-item sample (visual verifier, answer-shown control)
before the full set is generated. Target = NEPAgent `causal_sft` JSON (observed / inferred; no tool field).

Usage (GPU): PYTHONPATH=. python -m src.train.build_sft_reasoning [--limit 100] [--out-dir cache/sft/reasoning_sft]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from string import Template
from typing import Any

from src.agent import tools
from src.agent.parsing import extract_json
from src.agent.prompting import frame_preamble
from src.models.vlm import DEFAULT_MODEL
from src.nep_agent import event_reasoning as er
from src.stages.questions import normalize_question
from src.train.sft_format import sft_record, write_dataset, write_manifest

logger = logging.getLogger("build_sft_reasoning")
DATASET_NAME = "reasoning_sft"
TEACHER_VERSION = "v2"
SOURCE = Path("cache/sft/stage1_nep_sft/records.jsonl")
PROMPTS = Path(__file__).parent / "prompts"
TEACHER_MAX_TOKENS = 2048  # stage-1 replies can be truncated at 1536
MENU = "\n".join(tools.TOOL_DESCRIPTIONS[t] for t in (tools.DENSE_RESAMPLE, tools.ZOOM, tools.TERMINAL_STATE))
RETRO = re.compile(r"#\d+|\bin the (?:clip|video)\b|\bas shown\b|\bearlier in the (?:clip|video)\b|\bwas (?:already )?shown\b|\bpreviously\b", re.I)  # the bare word "frame" is not a trigger: it rejects valid targets
MAX_EVENTS = 5
COPY_TAU = 0.7  # an inferred state whose words are mostly the terminal state's or the answer's is a copy, not a transition
FRAME_ONLY = re.compile(r"^\s*(?:frame\s*)?#?\d+(?:\s*(?:-|to|,)\s*#?\d+)*\s*\.?\s*$", re.I)
STOP = set("the a an and or of to in on at for with from by is are was were be been being it its this that these those his her their he she they them "
           "then next after before while into onto over under up down out off as so very more most some any all one two which who what when where".split())
UNSUPPORTED_TAU = 0.6


def parse_nep_source(row: dict[str, Any]) -> dict[str, Any]:
    """A record of the answer-only SFT set (`cache/sft/stage1_nep*`): option-free question, target = caption of the future part.

    Used so the NEPAgent SFT trains on exactly the items of the answer-only SFT; only the target format differs.
    """
    messages = row["record"]["messages"]
    question = messages[0]["content"].rsplit("\n", 1)[-1].strip()
    return {"sample_id": row["sample_id"], "video": row["video"], "question": question, "options": [], "correct": None,
            "gold_text": messages[-1]["content"].strip(), "images": row["record"]["images"]}


def nep_items(source: str, ids_from: str) -> list[dict[str, Any]]:
    """The answer-only SFT items in their training order: ids are the frame folders of that dataset's train.json."""
    by_id = {r["sample_id"]: r for r in map(json.loads, open(source, encoding="utf-8")) if r.get("status") == "ok"}
    order = [Path(x["images"][0]).parts[-3] for x in json.loads(Path(ids_from).read_text(encoding="utf-8"))]
    missing = [i for i in order if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} answer-only SFT items have no source record, e.g. {missing[:3]}")
    return [parse_nep_source(by_id[i]) for i in order]


def frame_ids(n: int) -> list[str]:
    return [f"#{i:02d}" for i in range(1, n + 1)]


def grams(text: str, n: int = 4) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def containment(text: str | None, reference: str) -> float:
    g = grams(text or "")
    return len(g & grams(reference)) / len(g) if g else 0.0


def validate_events(payload: Any, ids: list[str]) -> tuple[list[dict[str, Any]], int]:
    """Grounded events in order, and how many were dropped for missing or invalid frame ids."""
    items = payload.get("observed_events") if isinstance(payload, dict) else None
    kept, dropped = [], 0
    for event in items if isinstance(items, list) else []:
        if not isinstance(event, dict) or not isinstance(event.get("description"), str) or not event["description"].strip():
            dropped += 1
            continue
        first = next(iter(re.findall(r"#\d+", str(event.get("first_frame")))), None)
        last = next(iter(re.findall(r"#\d+", str(event.get("last_frame")))), first)
        if first not in ids or last not in ids or ids.index(first) > ids.index(last):
            dropped += 1
            continue
        kept.append({"first_frame": first, "last_frame": last, "description": event["description"].strip()})
    return kept, dropped


def validate_observation(payload: Any, ids: list[str]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Stage-1 facts: grounded events in order, terminal state, ongoing action and their supporting frames."""
    events, dropped = validate_events(payload, ids)
    info = {"events_dropped": dropped}
    if not isinstance(payload, dict):
        return None, {**info, "status": "observe_not_json"}
    terminal = payload.get("terminal_state").strip() if isinstance(payload.get("terminal_state"), str) and payload.get("terminal_state").strip() else None
    ongoing = payload.get("ongoing_action").strip() if isinstance(payload.get("ongoing_action"), str) and payload.get("ongoing_action").strip().lower() not in ("", "null", "none") else None
    if not events:
        return None, {**info, "status": "no_grounded_event"}
    if not terminal or FRAME_ONLY.match(terminal) or len(terminal.split()) < 4:
        return None, {**info, "status": "no_terminal_state"}  # a bare frame id such as "frame #32" is not a state
    if ongoing and ongoing.strip().lower() == terminal.lower():
        ongoing = None  # a copy of the terminal state is not an ongoing action
    if len(events) > MAX_EVENTS:
        info["events_trimmed"] = len(events) - MAX_EVENTS
        events = events[-MAX_EVENTS:]  # keep the events nearest the end of the clip; every kept event is still a stage-1 observation
    support = payload.get("supporting_evidence") if isinstance(payload.get("supporting_evidence"), dict) else {}
    cite = lambda key: [i for i in re.findall(r"#\d+", str(support.get(key))) if i in ids]
    observed = {"events": events, "terminal_state": terminal, "ongoing_action": ongoing,
                "supporting_evidence": sorted(set(cite("terminal_state") + cite("ongoing_action")), key=ids.index) or ids[-2:]}
    return observed, {**info, "status": "ok", "terminal_cited": bool(cite("terminal_state"))}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 3 and w not in STOP}


def hindsight_flags(inferred: dict[str, Any], observed: dict[str, Any], gold: str, question: str) -> dict[str, Any]:
    stage2 = " ".join([*inferred["intermediate_states"], inferred["causal_transition"]])
    known = content_words(json.dumps(observed, ensure_ascii=False) + " " + gold + " " + question)
    new = content_words(stage2) - known
    ratio = len(new) / max(1, len(content_words(stage2)))
    drift = 1 - containment(inferred.get("predicted_event") or "", gold) if inferred.get("predicted_event") else 1.0
    return {"retro_reference": bool(RETRO.search(stage2)), "unsupported_word_ratio": round(ratio, 3), "unsupported_words": sorted(new)[:12],
            "restated_event_overlap": round(1 - drift, 3)}


def validate_inferred(payload: Any) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(payload, dict):
        return None, "bridge_not_json"
    states = [x.strip() for x in payload.get("intermediate_states") or [] if isinstance(x, str) and x.strip()] if isinstance(payload.get("intermediate_states"), list) else []
    transition = payload.get("causal_transition").strip() if isinstance(payload.get("causal_transition"), str) and payload.get("causal_transition").strip() else None
    if not transition:
        return None, "bridge_missing_transition"
    event = payload.get("predicted_event").strip() if isinstance(payload.get("predicted_event"), str) else ""
    return {"intermediate_states": states[:3], "causal_transition": transition, "predicted_event": event}, "ok"


def decide(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    """Accept or reject one teacher row; returns (target payload, status, flags)."""
    ids = frame_ids(len(row["images"]))
    observed, info = validate_observation(extract_json(row["teacher_observe_raw"]), ids)
    if observed is None:
        return None, info["status"], info
    inferred, status = validate_inferred(extract_json(row["teacher_bridge_raw"]))
    if inferred is None:
        return None, status, info
    flags = {**info, **hindsight_flags(inferred, observed, row["gold_text"], row["question"])}
    if flags["retro_reference"]:
        return None, "hindsight_retro_reference", flags
    flags["unsupported_content_flag"] = flags["unsupported_word_ratio"] > UNSUPPORTED_TAU  # recorded, not a rejection: inferred states legitimately add words
    kept_states, copies = [], 0
    for state in inferred["intermediate_states"]:
        words = content_words(state)
        same = lambda ref: len(words & content_words(ref)) / max(1, len(words))
        if words and (same(observed["terminal_state"]) >= COPY_TAU or same(row["gold_text"]) >= COPY_TAU):
            copies += 1
            continue
        kept_states.append(state)
    flags["inferred_copies_dropped"] = copies
    if inferred["intermediate_states"] and not kept_states:
        return None, "inferred_only_copies", flags
    target = {"observed": observed, "inferred": {"intermediate_states": kept_states,
                                                 "causal_transition": inferred["causal_transition"], "predicted_event": row["gold_text"]}}
    return target, "ok", flags


def input_prompt(item: dict[str, Any]) -> str:
    ids = frame_ids(len(item["images"]))
    common = {"preamble": frame_preamble(ids, True), "audio": "Audio: not available.",
              "question_block": "Question: " + normalize_question(item["question"], "conditioned")["text"]}
    return er.render_reason("causal_sft", common, ids[0], ids[-1], MENU)


def run_teacher(items: list[dict[str, Any]], model: str, batch: int, vlm=None) -> list[dict[str, Any]]:
    from src.models.vlm import GenerationConfig, VLLMQwen, VLMRequest

    vlm = vlm or VLLMQwen(model)
    gen = GenerationConfig(max_tokens=TEACHER_MAX_TOKENS, temperature=0.0)
    observe_t = Template((PROMPTS / "teacher_observe.txt").read_text(encoding="utf-8"))
    bridge_t = Template((PROMPTS / "teacher_bridge.txt").read_text(encoding="utf-8"))
    out = []
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        pre = [frame_preamble(frame_ids(len(it["images"])), True) for it in chunk]
        obs = vlm.generate([VLMRequest(text=observe_t.substitute(preamble=p), images=tuple(it["images"])) for p, it in zip(pre, chunk)], gen)
        facts = [validate_observation(extract_json(r), frame_ids(len(it["images"])))[0] for r, it in zip(obs, chunk)]
        need = [k for k, f in enumerate(facts) if f is not None]
        bridges = vlm.generate([VLMRequest(text=bridge_t.substitute(preamble=pre[k], observed=json.dumps(facts[k], ensure_ascii=False),
                                                                    gold=chunk[k]["gold_text"]), images=tuple(chunk[k]["images"])) for k in need], gen)
        by_k = dict(zip(need, bridges))
        for k, (it, raw_obs) in enumerate(zip(chunk, obs)):
            out.append({**it, "teacher_version": TEACHER_VERSION, "teacher_observe_raw": raw_obs, "teacher_bridge_raw": by_k.get(k)})
        logger.info("teacher %d/%d", min(start + batch, len(items)), len(items))
    return out


def assemble(teacher_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    counts: dict[str, int] = {"teacher_rows": len(teacher_rows), "duplicate_video": 0}
    records, audit, seen = [], [], set()
    for row in teacher_rows:
        if row["video"] in seen:
            counts["duplicate_video"] += 1
            continue
        target, status, flags = decide(row)
        counts[status] = counts.get(status, 0) + 1
        audit.append({"sample_id": row["sample_id"], "status": status, **flags})
        if target is None:
            continue
        seen.add(row["video"])
        records.append(sft_record(input_prompt(row), json.dumps(target, ensure_ascii=False), row["images"]))
    counts["kept"] = len(records)
    return records, counts, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--out-dir", default="cache/sft/reasoning_sft")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Teacher model; default $VLM_PATH")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--ids-from", default="cache/sft/stage1_nep_sft/train.json", help="dataset whose items to use, in its order")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    items = nep_items(args.source, args.ids_from)
    items = items[: args.limit] if args.limit else items
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    teacher_file = out / "teacher.jsonl"
    done = {json.loads(l)["sample_id"]: json.loads(l) for l in open(teacher_file, encoding="utf-8")} if teacher_file.exists() else {}
    pending = [it for it in items if it["sample_id"] not in done]
    if pending:
        with teacher_file.open("a", encoding="utf-8") as fh:
            for row in run_teacher(pending, args.model, args.batch):
                done[row["sample_id"]] = row
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    records, counts, audit = assemble([done[it["sample_id"]] for it in items if it["sample_id"] in done])
    (out / "audit.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audit), encoding="utf-8")
    write_dataset(out, DATASET_NAME, records)
    write_manifest(out, {"dataset": DATASET_NAME, "source": args.source, "source_format": "nep",
                         "ids_from": args.ids_from, "teacher_model": args.model, "teacher_temperature": 0.0,
                         "teacher_prompts": ["src/train/prompts/teacher_observe.txt", "src/train/prompts/teacher_bridge.txt"],
                         "teacher_version": TEACHER_VERSION, "input_prompt": "src/nep_agent/prompts/reason.txt (causal_sft), audio not available", "counts": counts,
                         "hindsight_thresholds": {"unsupported_word_ratio": UNSUPPORTED_TAU}, "tool_field": "excluded (no tool supervision)",
                         "license_note": "V1-33K / FutureBench train"})
    logger.info("written %d records: %s", len(records), counts)


if __name__ == "__main__":
    main()
