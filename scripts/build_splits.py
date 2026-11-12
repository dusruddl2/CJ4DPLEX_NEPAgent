#!/usr/bin/env python
"""Create dev/test question files and separate label files from the FutureOmni annotations."""
from __future__ import annotations

import argparse
import logging
from collections import Counter

from src.data.futureomni import load_items, stratified_split, to_records, write_jsonl

logger = logging.getLogger("build_splits")
SPLIT_SEED = 42


def _write(items, name: str, seed: int | None, args: argparse.Namespace) -> None:
    pairs = [to_records(item, seed) for item in items]
    write_jsonl(f"{args.questions_dir}/questions_{name}.jsonl", [question for question, _ in pairs])
    write_jsonl(f"{args.labels_dir}/labels_{name}.jsonl", [label for _, label in pairs])
    logger.info("%s: %d items, label letters %s", name, len(pairs), sorted(Counter(l["correct"] for _, l in pairs).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/raw/futureomni/futureomni_test.json")
    parser.add_argument("--questions-dir", default="data/splits")
    parser.add_argument("--labels-dir", default="data/labels")
    parser.add_argument("--dev", type=int, default=60)
    parser.add_argument("--test", type=int, default=300)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dev, test = stratified_split(load_items(args.annotations), args.dev, args.test, SPLIT_SEED)
    _write(dev, "dev", SPLIT_SEED, args)
    _write(test, "test", SPLIT_SEED, args)
    _write(test, "test_original", None, args)


if __name__ == "__main__":
    main()
