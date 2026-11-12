#!/usr/bin/env python
"""Create full-benchmark splits: every usable item except those sharing a video with dev."""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.futurebench import load_futurebench_items, split_futurebench  # noqa: E402
from src.data.futureomni import load_items, stratified_split, to_records, write_jsonl  # noqa: E402

SEED = 42
FB_ANNOTATIONS = str(Path(os.environ.get("V1_33K_ROOT", ROOT / "data/raw/v1_33k")) / "futurebench.json")
FO_ANNOTATIONS = ROOT / "data/raw/futureomni/futureomni_test.json"
_MERGE = re.compile(r"_merge(?=\.[^.]+$)")


def canonical(video: str) -> str:
    return _MERGE.sub("", video)


def without_dev(items: list, dev: list) -> list:
    dev_videos = {canonical(item.video) for item in dev}
    return [item for item in items if canonical(item.video) not in dev_videos]


def write(items: list, name: str, add_task: bool) -> None:
    pairs = [to_records(item, SEED) for item in items]
    labels = [
        {**label, "task": "interpolation" if label["pattern"] == "hop5" else "extrapolation"} if add_task else label
        for _, label in pairs
    ]
    write_jsonl(ROOT / f"data/splits/questions_{name}.jsonl", [question for question, _ in pairs])
    write_jsonl(ROOT / f"data/labels/labels_{name}.jsonl", labels)
    logging.info("%s: %d items", name, len(pairs))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fb_items = load_futurebench_items(FB_ANNOTATIONS)
    fb_dev, _ = split_futurebench(fb_items, 40, 200, SEED)
    write(without_dev(fb_items, fb_dev), "fb_full", add_task=True)

    fo_items = load_items(FO_ANNOTATIONS)
    fo_dev, _ = stratified_split(fo_items, 60, 300, SEED)
    write(without_dev(fo_items, fo_dev), "full", add_task=False)


if __name__ == "__main__":
    main()
