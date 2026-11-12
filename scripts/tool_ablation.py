"""Tool-menu ablation on test (report Tables 2 and 3). Reads stored results only; evaluation side (reads labels).

The final model's run stores every candidate for every item (`nep_agent_all_candidates`). Each one-tool run repeats only the
items whose first-prediction confidence is below the threshold, with a single tool on the menu
(`results/quantitative/runs/<split>_rt/ablation_tools_only_<tool>`); the other items keep the first prediction of the final run.
Scope: FutureBench test and the Speech items of FutureOmni test.

Usage: PYTHONPATH=. python scripts/tool_ablation.py --out results/quantitative/tables/tool_ablation_test.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as common  # noqa: E402

FINAL_RUN, BASELINE_RUN, GATE = "nep_agent_all_candidates", "plain_qwen", 0.75
SPLITS = {"test": "FO", "fb_test": "FB"}
ONE_TOOL = [("마지막 상태 재확인만 (1종 강제)", "ablation_tools_only_terminal"), ("구간 재샘플링만 (1종 강제)", "ablation_tools_only_dense"),
            ("영역 확대만 (1종 강제)", "ablation_tools_only_zoom")]


def answer(row: dict) -> str | None:
    """Final answer of a run that stored every candidate: tool + reasoning, else clip-only reasoning, else the first prediction."""
    trace = row["trace"]
    reasoning = (trace.get("nep_agent") or {}).get("reasoning") or {}
    pick = lambda name: ((reasoning.get(name) or {}).get("candidate") or {}).get("choice")
    return pick("tool_reason") or pick("causal_sft_nc") or trace["direct"]["direct_prediction"]["choice"]


def load_items() -> list[dict]:
    items = []
    for split, bench in SPLITS.items():
        gold = common.label_rows(split)
        final, plain = common.load_rows(split, FINAL_RUN), common.load_rows(split, BASELINE_RUN)
        one_tool = {run: common.load_rows(f"{split}_rt", run) for _, run in ONE_TOOL}
        for sid, row in final.items():
            label = gold[sid]
            if bench == "FO" and label.get("audio_type") != "Speech":
                continue
            direct = row["trace"]["direct"]
            confidence = (direct.get("routing") or {}).get("generation_confidence")
            routed = confidence is not None and confidence < GATE
            first = direct["direct_prediction"]["choice"]
            item = {"bench": bench, "pattern": label.get("pattern"), "routed": routed, "gold": label["correct"],
                    "순수 Qwen": (plain[sid].get("prediction") or {}).get("answer"),
                    "도구 3종 중 모델 선택 (최종 NEPAgent)": answer(row) if routed else first}
            for name, run in ONE_TOOL:
                item[name] = first if not routed else (answer(one_tool[run][sid]) if sid in one_tool[run] else None)
            items.append(item)
    return items


def accuracy(items: list[dict], name: str) -> str:
    scored = [x for x in items if x[name] is not None]
    return f"{100 * sum(x[name] == x['gold'] for x in scored) / len(scored):.1f}" if scored else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    items = load_items()
    names = ["순수 Qwen"] + [name for name, _ in ONE_TOOL] + ["도구 3종 중 모델 선택 (최종 NEPAgent)"]
    fb, fo, routed = [x for x in items if x["bench"] == "FB"], [x for x in items if x["bench"] == "FO"], [x for x in items if x["routed"]]
    lines = ["# 도구 구성별 정확도 (test, %)", "",
             "최종 구성에서 도구 메뉴만 바꿔 실행했다. 확신도가 임계값 미만인 문항만 각 구성으로 다시 실행했고, 나머지는 최종 실행의 초기 예측을 쓴다.", "",
             f"| 구성 | 전체 {len(items)} | FutureBench {len(fb)} | FutureOmni Speech {len(fo)} | 라우팅된 {len(routed)}문항 |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {name} | " + " | ".join(accuracy(group, name) for group in (items, fb, fo, routed)) + " |" for name in names]
    patterns = sorted({x["pattern"] for x in fo})
    lines += ["", "## FutureOmni Speech 질문 유형별", "",
              "| 구성 | " + " | ".join(f"{p} {sum(x['pattern'] == p for x in fo)}" for p in patterns) + " |", "|---|" + "---:|" * len(patterns)]
    lines += [f"| {name} | " + " | ".join(accuracy([x for x in fo if x["pattern"] == p], name) for p in patterns) + " |" for name in names]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
