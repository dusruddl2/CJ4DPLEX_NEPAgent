#!/usr/bin/env python
"""Cache Whisper and PANNs results for a split so VLM workers do not load audio models."""
from __future__ import annotations

import argparse
import logging
import subprocess

from src.agent.pipeline import AUDIO_CACHE_KEY
from src.data.futureomni import load_agent_inputs
from src.models.audio import AudioAnalyzer
from src.utils.cache import JsonCache
from src.utils.media import has_audio_stream

logger = logging.getLogger("precompute_audio")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--splits-root", default="data/splits")
    parser.add_argument("--cache-root", default="cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    index, total = (int(part) for part in args.shard.split("/"))
    samples = load_agent_inputs(f"{args.splits_root}/questions_{args.split}.jsonl", args.splits_root)[index::total]
    cache = JsonCache(args.cache_root)
    analyzer = AudioAnalyzer()
    failures = 0
    for number, sample in enumerate(samples, 1):
        if cache.get(sample.sample_id, "audio", AUDIO_CACHE_KEY) is not None:
            continue
        try:
            audio = analyzer.analyze(sample.prefix_path) if has_audio_stream(sample.prefix_path) else None
        except (subprocess.SubprocessError, RuntimeError, ValueError) as exc:
            failures += 1
            logger.error("%s: audio analysis failed: %s", sample.sample_id, exc)
            audio = None
        cache.put(sample.sample_id, "audio", AUDIO_CACHE_KEY, {"audio": audio})
        if number % 10 == 0:
            logger.info("%d/%d, failures %d", number, len(samples), failures)
    logger.info("done: %d samples, failures %d", len(samples), failures)


if __name__ == "__main__":
    main()
