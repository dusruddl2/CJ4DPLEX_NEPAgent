#!/usr/bin/env python
"""Create FutureBench dev/test question and label files, plus a distribution report."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.futureomni import to_records, write_jsonl  # noqa: E402
from src.data.futurebench import distribution_report, load_futurebench_items, split_futurebench  # noqa: E402

SPLIT_SEED = 42
DEFAULT_ANNOTATIONS = str(Path(os.environ.get("V1_33K_ROOT", ROOT / "data/raw/v1_33k")) / "futurebench.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--questions-dir", default=str(ROOT / "data/splits"))
    parser.add_argument("--labels-dir", default=str(ROOT / "data/labels"))
    parser.add_argument("--report", default=str(ROOT / "logs/futurebench_split_report.json"))
    parser.add_argument("--dev", type=int, default=40)
    parser.add_argument("--test", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    items = load_futurebench_items(args.annotations)
    dev, test = split_futurebench(items, args.dev, args.test, SPLIT_SEED)
    for name, group in (("dev", dev), ("test", test)):
        pairs = [to_records(item, SPLIT_SEED) for item in group]
        write_jsonl(f"{args.questions_dir}/questions_fb_{name}.jsonl", [q for q, _ in pairs])
        labels = [{**l, "task": "interpolation" if l["pattern"] == "hop5" else "extrapolation"} for _, l in pairs]
        write_jsonl(f"{args.labels_dir}/labels_fb_{name}.jsonl", labels)
        logging.info("fb_%s: %d items", name, len(pairs))

    report = distribution_report(items, dev, test)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("%s", json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
