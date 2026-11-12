"""Compare NEPAgent candidates on the same development-set items, from stored results; no model calls. Evaluation side (reads labels).

Policies are fixed in advance (no search): Direct; CoE, Causal, CoE+Causal and the observed/inferred causal schema (always
replace Direct, falling back to it when the reasoning JSON failed); CoE+Causal+Tools (the updated prediction when the tool
fired, else CoE+Causal); the same under a confidence router (tau 0.75 / 0.85); Tools only when the updated prediction cites
the new evidence; and the three-way rule KEEP / REASON / REOBSERVE.
Calls are counted per item for the path each policy actually needs (Direct = 2 incl. matcher).

Usage: PYTHONPATH=. python scripts/policy_eval.py --run nep_agent [--out results/quantitative/tables/policies_dev.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import RESULTS_ROOT, has_candidates, label_rows  # noqa: E402

GROUPS = {"FutureOmni": ("dev", "fo_devx"), "FutureBench": ("fb_dev", "fb_devx")}


def _read(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load(root: str, split: str, run: str) -> dict[str, dict]:
    paths = glob.glob(f"{root}/{split}/{run}/*.json")
    with ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(_read, paths))
    return {r["sample_id"]: r for r in rows}


def labels(split: str) -> dict[str, str]:
    return {sid: row["correct"] for sid, row in label_rows(split).items()}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def cand(entry: dict | None) -> str | None:
    c = (entry or {}).get("candidate")
    return c["choice"] if c else None


def item(row: dict) -> dict:
    trace = row["trace"]; r = trace["nep_agent"]["reasoning"]
    conf = (trace["direct"]["routing"] or {}).get("generation_confidence")
    tool_obs = (r.get("tool") or {}).get("observation") or {}
    tool_rec = (r.get("tool") or {}).get("record") or {}
    new_ids = set(tool_obs.get("new_evidence_ids") or [])
    return {"direct": trace["direct"]["direct_prediction"]["choice"], "conf": conf, "coe": cand(r.get("coe2") or r.get("coe")), "causal": cand(r.get("causal")),
            "cc": cand(r.get("coe_causal2") or r.get("coe_causal")), "tool": cand(r.get("tool")), "sft_schema": cand(r.get("causal_sft") or r.get("causal_sft_nc")), "tool_fired": tool_obs.get("status") == "executed",
            "requested": tool_obs.get("status") not in (None, "not_requested"), "tool_name": tool_obs.get("tool"),
            "cites_new": bool(new_ids & set(tool_rec.get("supporting_evidence") or [])),
            "elapsed": trace.get("elapsed_sec") or trace.get("elapsed") or 0.0}


def or_direct(x: dict, choice: str | None) -> str:
    return choice or x["direct"]


def routed(x: dict, tau: float) -> bool:
    return x["conf"] is not None and x["conf"] < tau


DIRECT = "Direct"
REASONING = "Causal reasoning (observed/inferred schema)"
POLICIES = {
    DIRECT: (lambda x: x["direct"], lambda x: 2),
    "CoE": (lambda x: or_direct(x, x["coe"]), lambda x: 4),
    "Causal": (lambda x: or_direct(x, x["causal"]), lambda x: 4),
    "CoE+Causal": (lambda x: or_direct(x, x["cc"]), lambda x: 4),
    "CoE+Causal+Tools": (lambda x: or_direct(x, x["tool"] or x["cc"]), lambda x: 4 + 2 * x["tool_fired"]),
    REASONING: (lambda x: or_direct(x, x["sft_schema"]), lambda x: 4),
    "Tools, only if the answer cites new evidence": (lambda x: x["tool"] if (x["tool"] and x["cites_new"]) else x["direct"], lambda x: 4 + 2 * x["tool_fired"]),
}
for tau in (0.75, 0.85):
    POLICIES[f"CoE+Causal, routed τ{tau}"] = (lambda x, t=tau: or_direct(x, x["cc"]) if routed(x, t) else x["direct"], lambda x, t=tau: 2 + 2 * routed(x, t))
    POLICIES[f"CoE+Causal+Tools, routed τ{tau}"] = (lambda x, t=tau: or_direct(x, x["tool"] or x["cc"]) if routed(x, t) else x["direct"],
                                                    lambda x, t=tau: 2 + (2 + 2 * x["tool_fired"]) * routed(x, t))
    POLICIES[f"3-way τ{tau}: KEEP / REASON / REOBSERVE"] = (
        lambda x, t=tau: x["direct"] if not routed(x, t) else or_direct(x, x["tool"]) if x["tool_fired"] else or_direct(x, x["cc"]),
        lambda x, t=tau: 2 + (2 + 2 * x["tool_fired"]) * routed(x, t))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="")
    parser.add_argument("--run", default="nep_agent")
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    args = parser.parse_args()
    lines = [f"# NEPAgent development-set comparison (run `{args.run}`)", "",
             "Candidates fixed before any policy is applied; policies were declared before results.", ""]
    for group, splits in GROUPS.items():
        items, gold = [], []
        for split in splits:
            lab = labels(split)
            rows = load(args.results_root, split, args.run)
            for sid in sorted(set(rows) & set(lab)):
                if has_candidates(rows[sid]):
                    items.append(item(rows[sid])); gold.append(lab[sid])
        n = len(items)
        if not n:
            lines += [f"## {group}: no results yet", ""]
            continue
        base = [x["direct"] == g for x, g in zip(items, gold)]
        lines += [f"## {group}: n={n}", "",
                  "| policy | acc | Δ vs Direct | Direct wrong→right / right→wrong | p | answer changed | calls/item |", "|---|---:|---:|---|---:|---:|---:|"]
        d_acc = 100 * sum(base) / n
        for name, (pick, cost) in POLICIES.items():
            ok = [pick(x) == g for x, g in zip(items, gold)]
            w = sum(o and not b for o, b in zip(ok, base)); r = sum(b and not o for o, b in zip(ok, base))
            lines.append(f"| {name} | {100 * sum(ok) / n:.1f} | {100 * sum(ok) / n - d_acc:+.1f} | "
                         f"{w} / {r} | {mcnemar(w, r):.3f} | {sum(pick(x) != x['direct'] for x in items)} | {sum(cost(x) for x in items) / n:.2f} |")
        fails = Counter(k for x in items for k in ("coe", "causal", "cc", "sft_schema") if x[k] is None)
        req = sum(x["requested"] for x in items); fired = sum(x["tool_fired"] for x in items)
        tools_used = Counter(x["tool_name"] for x in items if x["tool_fired"])
        oracle = sum(any(c == g for c in (x["direct"], x["coe"], x["causal"], x["cc"], x["tool"])) for x, g in zip(items, gold))
        lines += ["", f"Reasoning JSON failures (fall back to Direct): {dict(fails)}. Tool requested {req}/{n}, executed {fired}/{n} ({dict(tools_used)}). "
                  f"Oracle over Direct and the reasoning candidates (analysis bound only): {100 * oracle / n:.1f}. Frames in every prompt: 32 (+ up to 6 or 2 crops when a tool fires).", ""]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
