#!/usr/bin/env python
"""Create data/splits/{sample_id}/prefix.mp4 for dev and test. Safe to re-run; see section 5 of README.md.

--benchmark futureomni (default): extract clips, verify length against t_split, cut when needed.
--benchmark futurebench: clips are already first segments, so they are linked and measured.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Explicit root so that this runs under `python -I`, which ignores PYTHONPATH and the cwd.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.prefix_prep import VideosNotReady, prepare_linked_prefixes, prepare_prefixes  # noqa: E402

EXIT_NOT_READY = 2
EXIT_FAILURES = 1
FUTUREBENCH_VIDEO_ROOT = str(Path(os.environ.get("V1_33K_ROOT", ROOT / "data/raw/v1_33k")) / "first_segment_video")
FUTUREBENCH_QUESTIONS = [str(ROOT / "data/splits/questions_fb_dev.jsonl"), str(ROOT / "data/splits/questions_fb_test.jsonl")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("futureomni", "futurebench"), default="futureomni")
    parser.add_argument(
        "--questions", nargs="+",
        default=[str(ROOT / "data/splits/questions_dev.jsonl"), str(ROOT / "data/splits/questions_test.jsonl")],
    )
    parser.add_argument("--video-root", default=FUTUREBENCH_VIDEO_ROOT, help="futurebench: directory holding first_part_video/")
    parser.add_argument("--archive-dir", default=str(ROOT / "data/raw/futureomni"))
    parser.add_argument("--video-dir", default=str(ROOT / "data/raw/futureomni/videos"))
    parser.add_argument("--splits-root", default=str(ROOT / "data/splits"))
    parser.add_argument("--report", default=str(ROOT / "logs/prepare_prefix_report.json"))
    args = parser.parse_args()
    if args.benchmark == "futurebench":
        if args.questions == parser.get_default("questions"):
            args.questions = FUTUREBENCH_QUESTIONS
        if args.report == parser.get_default("report"):
            args.report = str(ROOT / "logs/prepare_prefix_fb_report.json")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.benchmark == "futurebench":
            report = prepare_linked_prefixes(args.questions, args.video_root, args.splits_root, args.report)
        else:
            report = prepare_prefixes(args.questions, args.archive_dir, args.video_dir, args.splits_root, args.report)
    except VideosNotReady as error:
        print(f"ERROR: test videos are not ready. {error}", file=sys.stderr)
        return EXIT_NOT_READY
    summary = report.get("counts", {"linked": report.get("linked")})
    print(f"samples={report['n_samples']} counts={summary} report={args.report}")
    return EXIT_FAILURES if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
