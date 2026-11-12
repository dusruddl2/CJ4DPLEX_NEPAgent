"""Question file with only the items the gate routes: data/splits/questions_<split>_rt.jsonl (used by configs/ablation/*.yaml).

Reads the stored run that holds every candidate (default nep_agent_all_candidates) and keeps the items whose first-prediction
confidence is below the threshold. `--audio-types Speech` also drops FutureOmni items of other audio types; that reads the label
file, so it belongs to the evaluation side.

Usage: PYTHONPATH=. python scripts/build_routed_splits.py --split test --audio-types Speech
       PYTHONPATH=. python scripts/build_routed_splits.py --split fb_test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as common  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--run", default="nep_agent_all_candidates")
    parser.add_argument("--tau", type=float, default=0.75)
    parser.add_argument("--audio-types", nargs="*", default=[])
    args = parser.parse_args()
    rows = common.load_rows(args.split, args.run)
    labels = common.label_rows(args.split) if args.audio_types else {}
    source = Path(common.SPLITS_DIR) / f"questions_{args.split}.jsonl"
    kept = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sid = json.loads(line)["sample_id"]
        confidence = (((rows.get(sid) or {}).get("trace", {}).get("direct") or {}).get("routing") or {}).get("generation_confidence")
        if confidence is None or confidence >= args.tau:
            continue
        if args.audio_types and labels[sid].get("audio_type") not in args.audio_types:
            continue
        kept.append(line)
    target = source.with_name(f"questions_{args.split}_rt.jsonl")
    target.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"{target}: {len(kept)} items")


if __name__ == "__main__":
    main()
