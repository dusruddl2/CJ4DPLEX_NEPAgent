"""Table 1 (main results): each row vs a reference row on the same items, per benchmark and overall.

Columns: accuracy and the difference from the reference row.
Rows use the spec format of scripts/subtype_table.py (kinds run / candidate / tool). Reads stored results only; evaluation
side (reads labels).

Usage: PYTHONPATH=. python scripts/main_table.py --phase test --block main --out results/quantitative/tables/main_test.md
       (--phase all pools the development set and test; --block policy adds calls per item;
        --audio-types Speech keeps only the FutureOmni items whose audio type is Speech; FutureBench is not filtered)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import subtype_table as st  # noqa: E402

BENCH_NAMES = {"FO": "FutureOmni", "FB": "FutureBench", "ALL": "전체"}


def correctness(row: dict, phase: str, gold: dict) -> dict[str, tuple[bool, str, str]]:
    """sample id -> (correct, bench, video) for the items this row answered."""
    out = {}
    for bench, splits in st.PHASES[phase].items():
        for split in splits:
            labels = gold[bench][split]
            ans = st.answers_for(row, split, phase)
            for sid, lab in labels.items():
                if ans.get(sid):
                    out[sid] = (ans[sid] == lab["correct"], bench, lab.get("video") or sid)
    return out


def calls_per_item(row: dict, phase: str) -> dict[str, tuple[int, int, str]]:
    """sample id -> (video-model calls, matcher calls, bench) for rows of kind `tool`; empty for other kinds."""
    out = {}
    if row["kind"] != "tool":
        return out
    for bench, splits in st.PHASES[phase].items():
        for split in splits:
            for sid, (_, video, matcher) in st.tool_decisions(st.pick_name(row["run"], phase), split, row["policy"]).items():
                out[sid] = (video, matcher, bench)
    return out


def compare(rows: list[dict], ref_index: int, phase: str, gold: dict, with_calls: bool = False) -> list[str]:
    results = [correctness(r, phase, gold) for r in rows]
    calls = [calls_per_item(r, phase) if with_calls else {} for r in rows]
    ref = results[ref_index]
    lines = []
    for bench in ("FO", "FB", "ALL"):
        lines += [f"### {BENCH_NAMES[bench]}", "", f"기준: {rows[ref_index]['name']}", "",
                  "| 설정 | 문항 | 정확도 | Δ |" + (" 영상 모델 호출/문항 | Matcher 호출/문항 |" if with_calls else ""),
                  "|---|---:|---:|---:|" + ("---:|---:|" if with_calls else "")]
        for row, res, cost in zip(rows, results, calls):
            ids = sorted(s for s in res if s in ref and (bench == "ALL" or res[s][1] == bench))
            if not ids:
                lines.append(f"| {row['name']} | 0 | Pending | |" + (" | |" if with_calls else ""))
                continue
            counted = [cost[s] for s in ids if s in cost]
            extra = "" if not with_calls else " – | – |" if not counted else \
                f" {sum(c[0] for c in counted) / len(counted):.2f} | {sum(c[1] for c in counted) / len(counted):.2f} |"
            a = np.array([res[s][0] for s in ids]); b = np.array([ref[s][0] for s in ids])
            lines.append(f"| {row['name']} | {len(ids)} | {100 * a.mean():.1f} | {100 * (a.mean() - b.mean()):+.1f} |" + extra)
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/report_tables.yaml")
    parser.add_argument("--phase", choices=tuple(st.PHASES), default="test")
    parser.add_argument("--block", default="main", help="block key in the spec (blocks with `key: <name>`)")
    parser.add_argument("--out", default="")
    parser.add_argument("--audio-types", nargs="*", default=[], help="keep only the FutureOmni items of these audio types, e.g. Speech")
    args = parser.parse_args()
    st.keep_audio(args.audio_types)
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    block = next(b for b in spec["blocks"] if b.get("key") == args.block)
    rows = st.rows_for(block, args.phase)
    gold = {bench: {s: st.label_rows(s) for s in splits} for bench, splits in st.PHASES[args.phase].items()}
    ref_index = next(i for i, r in enumerate(rows) if r.get("reference"))
    lines = [f"# {block['title']} ({st.phase_title(args.phase)})", "",
             "모든 행은 같은 문항, 같은 Matcher다. "
             "전체는 두 벤치 문항을 합친 값이다(문항 수 가중). 결과가 없는 행은 Pending이다."
             + (" 호출 수는 그 정책대로 실행할 때 필요한 호출만 센 문항당 평균이다(저장된 단계별 호출 수 기준)." if block.get("calls") else "")
             + (" 개발 세트(700문항)와 test 문항을 합친 것이다." if args.phase == "all" else ""), ""]
    for ref_name in [i for i, r in enumerate(rows) if r.get("reference") or r.get("second_reference")]:
        lines += [f"## 기준 = {rows[ref_name]['name']}", ""] + compare(rows, ref_name, args.phase, gold, bool(block.get("calls")))
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
