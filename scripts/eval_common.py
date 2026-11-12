"""Shared readers for the evaluation scripts: stored results and labels. Evaluation side (reads labels).

Results: $RESULTS_ROOT/<split>/<run>/<sample_id>.json (default results/quantitative/runs/). Labels: $LABELS_DIR/labels_<split>.jsonl
(default data/labels/), written by the split builders; they are never read by the agent. Question files (for the clip
length): $SPLITS_DIR/questions_<split>.jsonl (default data/splits/).
"""
from __future__ import annotations

import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results/quantitative/runs")  # scripts/run_benchmark.py writes <root>/<split>/<run>/<sample_id>.json
LABELS_DIR = os.environ.get("LABELS_DIR", "data/labels")
SPLITS_DIR = os.environ.get("SPLITS_DIR", "data/splits")
READ_WORKERS = 24
SOURCE_SPLITS = {"fo_devx": ("fo_devx", "full"), "fb_devx": ("fb_devx", "fb_full")}  # the devx questions are a subset of the full sets


def _read(path: str) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        print(f"skip {path}: {err}", file=sys.stderr)
        return None


def load_rows(split: str, run: str, root: str = RESULTS_ROOT) -> dict[str, Any]:
    with ThreadPoolExecutor(READ_WORKERS) as pool:
        return {r["sample_id"]: r for r in pool.map(_read, glob.glob(f"{root}/{split}/{run}/*.json")) if r}


def label_rows(split: str) -> dict[str, dict[str, Any]]:
    path = Path(LABELS_DIR) / f"labels_{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing label file {path}: build the splits (scripts/build_full_splits.py) or set LABELS_DIR")
    lines = path.read_text(encoding="utf-8").splitlines()
    return {r["sample_id"]: r for r in map(json.loads, lines) if r}


def has_candidates(row: dict[str, Any]) -> bool:
    """True when the result holds the NEPAgent trace and a Direct candidate."""
    trace = row.get("trace", {})
    return bool(trace.get("nep_agent")) and bool((trace.get("direct") or {}).get("direct_prediction"))


def question_rows(split: str) -> dict[str, dict[str, Any]]:
    path = Path(SPLITS_DIR) / f"questions_{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing question file {path}: build the splits or set SPLITS_DIR")
    return {r["sample_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}
