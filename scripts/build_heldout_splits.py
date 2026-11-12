#!/usr/bin/env python
"""Create the held-out splits: every benchmark item that is in neither the 700-item development set nor test.

Run after scripts/build_full_splits.py and scripts/build_dev_extra_splits.py. For each benchmark the rows of the full set
(which already leaves out every video of the small dev split) that are not in the extra dev split and not in the test split
are written in the order of the full set: fo_rest (FutureOmni) and fb_rest (FutureBench). Question and label rows are copied
unchanged from the full files.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.futureomni import read_jsonl, write_jsonl  # noqa: E402

SPLITS = {"fo_rest": ("full", ("fo_devx", "test")), "fb_rest": ("fb_full", ("fb_devx", "fb_test"))}  # new split -> (full set, splits to leave out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(ROOT / "data"), help="folder with splits/ and labels/")
    parser.add_argument("--out-root", default="", help="where to write splits/ and labels/ (default: --data-root)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    source, target = Path(args.data_root), Path(args.out_root or args.data_root)
    for name, (full, used_splits) in SPLITS.items():
        used = {row["sample_id"] for split in used_splits for row in read_jsonl(source / f"splits/questions_{split}.jsonl")}
        questions = [row for row in read_jsonl(source / f"splits/questions_{full}.jsonl") if row["sample_id"] not in used]
        labels = [row for row in read_jsonl(source / f"labels/labels_{full}.jsonl") if row["sample_id"] not in used]
        write_jsonl(target / f"splits/questions_{name}.jsonl", questions)
        write_jsonl(target / f"labels/labels_{name}.jsonl", labels)
        logging.info("%s: %d items from %s without %s", name, len(questions), full, ", ".join(used_splits))


if __name__ == "__main__":
    main()
