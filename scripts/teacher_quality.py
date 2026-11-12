"""Audit of the NEPAgent Teacher data on a sample, before the full set is generated. Training-side (reads answers).

1. Automatic: status counts, events per item, terminal-state citation, hindsight flags from build_sft_reasoning.decide.
2. Visual verifier (same Qwen3-VL-8B, answer NOT shown, yes/no by letter probability):
   a. each stage-1 event against only the frames it cites  -> grounding precision of the observed facts;
   b. each stage-2 intermediate state against the whole clip ("already visible?")  -> inferred states that claim the past.
3. Answer-shown control: stage 1 rerun WITH the answer appended. If its observed events overlap the answer much more than the
   blind stage-1 events do, the teacher is prone to hindsight and the two-stage split is what keeps it out of the targets.
4. 20 random cases written out for manual reading.
The verifier is the teacher model itself; its yes/no is a consistency check, not ground truth.

Usage (GPU): PYTHONPATH=. python scripts/teacher_quality.py --dir cache/sft/reasoning_sft_audit --out tables/teacher_quality.md
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path
from string import Template

from src.agent.parsing import extract_json
from src.agent.prompting import frame_preamble
from src.models.vlm import DEFAULT_MODEL, GenerationConfig, VLLMQwen, VLMRequest
from src.train import build_sft_reasoning as b

EVENT_Q = 'These frames come from a video clip. Do they show the following?\n"$text"\nOptions:\nA. yes\nB. no\n\nRespond with only the letter.'
PAST_Q = ('$preamble\nThe clip ends at the last frame. Consider this statement:\n"$text"\nIs it already visible or audible in the clip itself (rather than '
          'something that could only happen after the clip)?\nOptions:\nA. yes\nB. no\n\nRespond with only the letter.')
CONTROL_SUFFIX = '\n\nFor reference, after the clip this happens next: "$gold"'


def yes(vlm, requests):
    probs = vlm.option_probs(requests, [("A", "B")] * len(requests)) if requests else []
    return [bool(p) and p.get("A", 0) >= p.get("B", 0) for p in probs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Teacher model; default $VLM_PATH")
    args = parser.parse_args()
    rows = [json.loads(l) for l in open(Path(args.dir) / "teacher.jsonl", encoding="utf-8") if l.strip()]
    decided = [(r, *b.decide(r)) for r in rows]
    vlm = VLLMQwen(args.model)
    kept = [(r, t, f) for r, t, s, f in decided if t is not None]
    # 2a. events against their own frames
    ev_req, ev_owner = [], []
    for r, t, _ in kept:
        ids = b.frame_ids(len(r["images"]))
        for e in t["observed"]["events"]:
            frames = r["images"][ids.index(e["first_frame"]):ids.index(e["last_frame"]) + 1]
            ev_req.append(VLMRequest(text=Template(EVENT_Q).substitute(text=e["description"]), images=tuple(frames[:8])))
            ev_owner.append(r["sample_id"])
    ev_yes = yes(vlm, ev_req)
    # 2b. inferred states: already visible?
    st_req, st_owner = [], []
    for r, t, _ in kept:
        pre = frame_preamble(b.frame_ids(len(r["images"])), True)
        for s in t["inferred"]["intermediate_states"] + [t["inferred"]["causal_transition"]]:
            st_req.append(VLMRequest(text=Template(PAST_Q).substitute(preamble=pre, text=s), images=tuple(r["images"])))
            st_owner.append(r["sample_id"])
    st_yes = yes(vlm, st_req)
    # 3. answer-shown control of stage 1
    observe_t = Template((b.PROMPTS / "teacher_observe.txt").read_text(encoding="utf-8"))
    ctrl = vlm.generate([VLMRequest(text=observe_t.substitute(preamble=frame_preamble(b.frame_ids(len(r["images"])), True))
                                    + Template(CONTROL_SUFFIX).substitute(gold=r["gold_text"]), images=tuple(r["images"])) for r in rows],
                        GenerationConfig(max_tokens=b.TEACHER_MAX_TOKENS))
    def overlap(text, gold):
        g = b.content_words(gold)
        return len(b.content_words(text) & g) / max(1, len(g))
    blind_ov, ctrl_ov, blind_last, ctrl_last = [], [], [], []
    for r, raw in zip(rows, ctrl):
        blind, _ = b.validate_observation(extract_json(r["teacher_observe_raw"]), b.frame_ids(len(r["images"])))
        shown, _ = b.validate_observation(extract_json(raw), b.frame_ids(len(r["images"])))
        if blind is None or shown is None:
            continue
        txt = lambda o: " ".join(e["description"] for e in o["events"]) + " " + o["terminal_state"] + " " + (o["ongoing_action"] or "")
        blind_ov.append(overlap(txt(blind), r["gold_text"])); ctrl_ov.append(overlap(txt(shown), r["gold_text"]))
        blind_last.append(len(blind["events"])); ctrl_last.append(len(shown["events"]))
    status = {}
    for _, _, s, _ in decided:
        status[s] = status.get(s, 0) + 1
    flags = [f for _, t, _, f in decided if "unsupported_word_ratio" in f]
    lines = [f"# NEPAgent Teacher audit (n={len(rows)})", "", f"Status: {status}", "",
             "| check | value |", "|---|---|",
             f"| kept | {len(kept)}/{len(rows)} |",
             f"| events per kept item (mean) | {st.mean(len(t['observed']['events']) for _, t, _ in kept):.2f} |" if kept else "| events per kept item | – |",
             f"| terminal state cites frames | {sum(f.get('terminal_cited', False) for _, _, f in kept)}/{len(kept)} |",
             f"| stage-1 events judged visible in their own frames (verifier) | {sum(ev_yes)}/{len(ev_yes)} ({100 * sum(ev_yes) / max(1, len(ev_yes)):.1f}%) |",
             f"| inferred states/transitions judged ALREADY visible in the clip (verifier) | {sum(st_yes)}/{len(st_yes)} ({100 * sum(st_yes) / max(1, len(st_yes)):.1f}%) |",
             f"| stage-2 retro references (rejected) | {sum(f['retro_reference'] for f in flags)}/{len(flags)} |",
             f"| stage-2 unsupported content-word ratio (median) | {st.median(f['unsupported_word_ratio'] for f in flags):.2f} |" if flags else "| unsupported ratio | – |",
             f"| answer words in observed facts: blind stage 1 vs answer-shown control (mean) | {st.mean(blind_ov):.3f} vs {st.mean(ctrl_ov):.3f} (n={len(blind_ov)}) |" if blind_ov else "| control | – |",
             f"| observed events per item: blind vs answer-shown | {st.mean(blind_last):.2f} vs {st.mean(ctrl_last):.2f} |" if blind_last else "| events control | – |", ""]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    rng = random.Random(0)
    cases = rng.sample(kept, min(20, len(kept)))
    ev_by = {}
    for o, y in zip(ev_owner, ev_yes):
        ev_by.setdefault(o, []).append(y)
    st_by = {}
    for o, y in zip(st_owner, st_yes):
        st_by.setdefault(o, []).append(y)
    with open(Path(args.dir) / "audit_cases.jsonl", "w", encoding="utf-8") as fh:
        for r, t, f in cases:
            fh.write(json.dumps({"sample_id": r["sample_id"], "question": r["question"], "gold": r["gold_text"], "target": t, "flags": f,
                                 "event_visible": ev_by.get(r["sample_id"]), "inferred_already_visible": st_by.get(r["sample_id"])}, ensure_ascii=False) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
