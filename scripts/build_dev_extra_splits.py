#!/usr/bin/env python
"""Create the two extra dev splits: 300 FutureOmni and 300 FutureBench items drawn from the full sets without the test items.

Run after scripts/build_splits.py, scripts/build_splits_fb.py and scripts/build_full_splits.py. For each benchmark the items of
the full set that are not in the test split are sampled with a fixed seed (300 ids, random.Random(42).sample) and written in
the order of the full set. The full sets already exclude every video of the small dev splits, so
dev 700 = dev (60) + fo_devx (300) + fb_dev (40) + fb_devx (300) shares no item with test. Question and label rows are copied
unchanged from the full files.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.futureomni import read_jsonl, write_jsonl  # noqa: E402

SEED = 42
SIZE = 300
SPLITS = {"fo_devx": ("full", "test"), "fb_devx": ("fb_full", "fb_test")}  # new split -> (full set, test split to leave out)


def select(full_ids: list[str], test_ids: set[str], size: int = SIZE, seed: int = SEED) -> set[str]:
    """Seeded sample of the full-set ids that are not test items."""
    pool = [sample_id for sample_id in full_ids if sample_id not in test_ids]
    if len(pool) < size:
        raise ValueError(f"only {len(pool)} items outside the test split, need {size}")
    return set(random.Random(seed).sample(pool, size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(ROOT / "data"), help="folder with splits/ and labels/")
    parser.add_argument("--out-root", default="", help="where to write splits/ and labels/ (default: --data-root)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    source, target = Path(args.data_root), Path(args.out_root or args.data_root)
    for name, (full, test) in SPLITS.items():
        questions = read_jsonl(source / f"splits/questions_{full}.jsonl")
        labels = read_jsonl(source / f"labels/labels_{full}.jsonl")
        test_ids = {row["sample_id"] for row in read_jsonl(source / f"splits/questions_{test}.jsonl")}
        chosen = select([row["sample_id"] for row in questions], test_ids)
        write_jsonl(target / f"splits/questions_{name}.jsonl", [row for row in questions if row["sample_id"] in chosen])
        write_jsonl(target / f"labels/labels_{name}.jsonl", [row for row in labels if row["sample_id"] in chosen])
        logging.info("%s: %d items from %s without %s", name, len(chosen), full, test)


if __name__ == "__main__":
    main()
