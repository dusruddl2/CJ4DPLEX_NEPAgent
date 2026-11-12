"""Ablation tables split by question type: FutureBench by hop (hop1/2/3/5), FutureOmni by pattern
(Causal / Routine / Thematic). Reads stored results only; no model calls. Evaluation side (reads labels).

Rows come from a YAML spec (configs/report_tables.yaml). Row kinds:
  run        - answers stored in a run (prediction.answer)
  candidate  - a stored candidate of a run under a named policy from scripts/policy_eval.py
  tool       - a run that stored every candidate, under a policy: direct | reason (clip-only reasoning for every item) |
               tool (tool + reasoning for every item) | gate@<tau> (Direct when the Direct sentence confidence is at least
               tau, else tool + reasoning) | tool@<tau> and tool@gap (clip-only reasoning, replaced by tool + reasoning when
               the confidence is below tau / when the selection call reported an unclear detail inside the clip).
               A missing candidate falls back tool -> clip-only reasoning -> Direct.
`config` / `run` may be one name or {dev: name, test: name}. Missing results print as Pending.
In each block the best value per column is bold and the second best underlined (ties share the mark).

Usage: PYTHONPATH=. python scripts/subtype_table.py --phase dev --out results/quantitative/tables/ablation_by_type_dev.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy_eval as pe  # noqa: E402
import eval_common as common  # noqa: E402

PHASES = {"dev": {"FO": ("dev", "fo_devx"), "FB": ("fb_dev", "fb_devx")}, "test": {"FO": ("test",), "FB": ("fb_test",)},
          "all": {"FO": ("dev", "fo_devx", "test"), "FB": ("fb_dev", "fb_devx", "fb_test")},  # all = development set and test pooled
          "heldout": {"FO": ("fo_rest",), "FB": ("fb_rest",)}}  # items in neither the development set nor test
PHASE_NAMES = {"dev": "개발 세트", "test": "test", "all": "개발 세트 + test", "heldout": "추가 검증 세트"}
FO_AUDIO_TYPES = ("Speech", "Music", "Sound")  # the benchmark's audio label of a FutureOmni item; FutureBench items have none
KEPT_AUDIO: set[str] = set()  # FutureOmni audio types kept in every table (set by --audio-types); empty = all


def label_rows(split: str) -> dict[str, dict[str, Any]]:
    """Labels of a split. With --audio-types only the FutureOmni items of those audio types stay; FutureBench is never filtered."""
    rows = common.label_rows(split)
    if not KEPT_AUDIO:
        return rows
    return {sid: row for sid, row in rows.items() if row.get("audio_type") not in FO_AUDIO_TYPES or row["audio_type"] in KEPT_AUDIO}


def keep_audio(types: list[str] | tuple[str, ...]) -> None:
    """Apply --audio-types (replaces any earlier choice)."""
    unknown = set(types) - set(FO_AUDIO_TYPES)
    if unknown:
        raise SystemExit(f"unknown audio types {sorted(unknown)}; choose from {FO_AUDIO_TYPES}")
    KEPT_AUDIO.clear()
    KEPT_AUDIO.update(types)


def phase_title(phase: str) -> str:
    kept = [t for t in FO_AUDIO_TYPES if t in KEPT_AUDIO]
    return PHASE_NAMES[phase] + (f", FutureOmni는 {'·'.join(kept)} 문항만" if kept else "")
SUBTYPES = {"FB": ("hop1", "hop2", "hop3", "hop5"), "FO": ("Causal", "Routine Sequences", "Thematic Montages")}
SHORT = {"Routine Sequences": "Routine", "Thematic Montages": "Thematic"}
MIN_COVERAGE = 0.98  # below this share of items a cell is shown as Pending (partial run)


def rows_for(block: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    """Rows of a block that exist in this phase; the pooled phase needs a row to exist in both dev and test."""
    needed = ("dev", "test") if phase == "all" else (phase,)
    return [r for r in block["rows"] if all(p in r.get("phases", ("dev", "test")) for p in needed)]


def pick_name(value: Any, phase: str) -> str:
    return value[phase] if isinstance(value, dict) else value


def run_answers(config: str, split: str) -> dict[str, str]:
    for source in common.SOURCE_SPLITS.get(split, (split,)):
        rows = common.load_rows(source, config)
        if rows:
            return {s: (r.get("prediction") or {}).get("answer") for s, r in rows.items()}
    return {}


def candidate_answers(run: str, split: str, policy: str) -> dict[str, str]:
    pick, _ = pe.POLICIES[policy]
    rows = pe.load(common.RESULTS_ROOT, split, run)
    return {s: pick(pe.item(r)) for s, r in rows.items() if common.has_candidates(r)}


def tool_decisions(run: str, split: str, policy: str) -> dict[str, tuple[str, int, int]]:
    """sample id -> (answer, video-model calls, matcher calls) of a stored all-candidates run under a tool policy.

    Calls are the ones a pipeline following that policy makes, taken from the stored per-stage counts: the clip-only
    reasoning is counted only where the policy uses it (always for `reason` and `tool@...`; for `tool` and `gate@...` only
    when the tool path gave no candidate).
    """
    out = {}
    for sid, r in pe.load(common.RESULTS_ROOT, split, run).items():
        trace = r.get("trace", {})
        direct = ((trace.get("direct") or {}).get("direct_prediction") or {}).get("choice")
        agent = trace.get("nep_agent") or {}
        reasoning = agent.get("reasoning") or {}
        if not direct or "tool_reason" not in reasoning:
            continue
        pick = lambda name: ((reasoning.get(name) or {}).get("candidate") or {}).get("choice")
        reason = pick("causal_sft_nc") or direct
        tool = pick("tool_reason") or reason
        conf = (trace["direct"].get("routing") or {}).get("generation_confidence")
        calls = agent.get("stage_calls") or {}
        split_calls = lambda n: (min(n, 1), max(n - 1, 0))  # one video call, then the matcher call if the reply was usable
        cost_direct = (calls.get("direct_video", 1), calls.get("direct_match", 1))
        cost_reason = split_calls(calls.get("causal_sft_nc", 0))
        after_tool = split_calls(calls.get("tool_reason", 0))
        cost_tool = (calls.get("select", 0) + after_tool[0], after_tool[1])
        lazy_tool = cost_tool if pick("tool_reason") else tuple(a + b for a, b in zip(cost_tool, cost_reason))
        add = lambda *costs: tuple(sum(c[i] for c in costs) for i in (0, 1))
        below = lambda tau: conf is not None and conf < tau
        if policy == "direct":
            choice, cost = direct, cost_direct
        elif policy == "reason":
            choice, cost = reason, add(cost_direct, cost_reason)
        elif policy == "tool":
            choice, cost = tool, add(cost_direct, lazy_tool)
        elif policy.startswith("gate@"):
            routed = below(float(policy[5:]))
            choice, cost = (tool, add(cost_direct, lazy_tool)) if routed else (direct, cost_direct)
        elif policy == "tool@gap" or policy.startswith("tool@"):
            use = bool((reasoning["tool_reason"].get("selection") or {}).get("observable_gap")) if policy == "tool@gap" else below(float(policy[5:]))
            choice, cost = (tool, add(cost_direct, cost_reason, cost_tool)) if use else (reason, add(cost_direct, cost_reason))
        else:
            raise ValueError(f"unknown tool policy {policy}")
        out[sid] = (choice, *cost)
    return out


def tool_answers(run: str, split: str, policy: str) -> dict[str, str]:
    return {sid: decision[0] for sid, decision in tool_decisions(run, split, policy).items()}


def answers_for(row: dict[str, Any], split: str, phase: str) -> dict[str, str]:
    kind = row["kind"]
    if kind == "run":
        return run_answers(pick_name(row["config"], phase), split)
    if kind == "candidate":
        return candidate_answers(pick_name(row["run"], phase), split, row["policy"])
    if kind == "tool":
        return tool_answers(pick_name(row["run"], phase), split, row["policy"])
    raise ValueError(f"unknown row kind {kind}")


def row_scores(row: dict[str, Any], phase: str, gold: dict[str, dict[str, dict]]) -> dict[str, float | None]:
    """Accuracy per bench subtype, per bench average (micro), and overall; None when coverage is incomplete."""
    hits: dict[str, list[bool]] = {}
    complete = True
    for bench, splits in PHASES[phase].items():
        for split in splits:
            labels = gold[bench][split]
            ans = answers_for(row, split, phase)
            done = [s for s in labels if ans.get(s)]
            complete &= len(done) >= MIN_COVERAGE * len(labels)
            for sid in done:
                ok = ans[sid] == labels[sid]["correct"]
                for key in (f"{bench}:{labels[sid]['pattern']}", f"{bench}:AVG", "ALL"):
                    hits.setdefault(key, []).append(ok)
    if not complete:
        return {}
    return {k: 100 * sum(v) / len(v) for k, v in hits.items()}


def columns() -> list[tuple[str, str]]:
    cols = []
    for bench in ("FB", "FO"):
        cols += [(f"{bench}:{t}", f"{bench} {SHORT.get(t, t)}") for t in SUBTYPES[bench]] + [(f"{bench}:AVG", f"{bench} AVG")]
    return cols + [("ALL", "Overall")]


def mark(values: list[float | None]) -> list[str]:
    ranked = sorted({round(v, 1) for v in values if v is not None}, reverse=True)
    out = []
    for v in values:
        if v is None:
            out.append("Pending")
            continue
        text = f"{v:.1f}"
        out.append(f"**{text}**" if ranked and round(v, 1) == ranked[0] else f"<u>{text}</u>" if len(ranked) > 1 and round(v, 1) == ranked[1] else text)
    return out


def counts_line(phase: str, gold: dict[str, dict[str, dict]]) -> str:
    parts = []
    for key, name in columns():
        if key == "ALL":
            n = sum(len(v) for b in gold.values() for v in b.values())
        else:
            bench, sub = key.split(":")
            items = [r for v in gold[bench].values() for r in v.values()]
            n = len(items) if sub == "AVG" else sum(r["pattern"] == sub for r in items)
        parts.append(f"{name} {n}")
    return "문항 수: " + ", ".join(parts) + "."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/report_tables.yaml")
    parser.add_argument("--phase", choices=tuple(PHASES), default="dev")
    parser.add_argument("--out", default="")
    parser.add_argument("--audio-types", nargs="*", default=[], help="keep only the FutureOmni items of these audio types, e.g. Speech")
    args = parser.parse_args()
    keep_audio(args.audio_types)
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    gold = {bench: {s: label_rows(s) for s in splits} for bench, splits in PHASES[args.phase].items()}
    cols = columns()
    lines = [f"# 유형별 ablation 표 ({phase_title(args.phase)})", "",
             "FutureBench는 hop별, FutureOmni는 질문 유형별 정확도(%)다. AVG는 각 벤치 전체 문항 기준이다. "
             "블록마다 열별 최고값은 굵게, 두 번째는 밑줄로 표시했다. 결과가 다 모이지 않은 칸은 Pending이다.",
             "", counts_line(args.phase, gold), ""]
    for block in spec["blocks"]:
        rows = rows_for(block, args.phase)
        if not rows:
            continue
        scores = [row_scores(r, args.phase, gold) for r in rows]
        lines += [f"## {block['title']}", "", "| 설정 | " + " | ".join(n for _, n in cols) + " |", "|---|" + "---:|" * len(cols)]
        per_col = [mark([s.get(k) if s else None for s in scores]) for k, _ in cols]
        for i, row in enumerate(rows):
            lines.append(f"| {row['name']} | " + " | ".join(per_col[j][i] for j in range(len(cols))) + " |")
        if block.get("note"):
            lines += ["", block["note"]]
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
