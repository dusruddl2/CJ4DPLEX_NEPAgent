#!/usr/bin/env python
"""Run one or more configurations on a split; finished samples are skipped on restart."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.agent.config import load_config
from src.agent.output import write_result
from src.agent.pipeline import build_services, run_batch
from src.schemas import AgentInput, RunConfig
from src.data.futureomni import load_agent_inputs
from src.models.vlm import DEFAULT_MODEL, VLLMQwen, resolve_model_path

logger = logging.getLogger("run_benchmark")


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse key=value pairs; values are decoded as JSON when possible."""
    overrides: dict[str, Any] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator:
            raise ValueError(f"override must be key=value: {pair}")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    return overrides


def parse_shard(text: str) -> tuple[int, int]:
    index, _, count = text.partition("/")
    shard, total = int(index), int(count)
    if not 0 <= shard < total:
        raise ValueError(f"invalid shard: {text}")
    return shard, total


def _run_config(
    cfg: RunConfig, samples: list[AgentInput], services, output_root: Path, batch_size: int
) -> None:
    output_dir = output_root / cfg.run_name
    pending = [s for s in samples if not (output_dir / f"{s.sample_id}.json").exists()]
    logger.info("%s: %d samples to run, %d already finished", cfg.run_name, len(pending), len(samples) - len(pending))
    for start in range(0, len(pending), batch_size):
        for result in run_batch(pending[start : start + batch_size], cfg, services):
            write_result(result, output_dir)
        logger.info("%s: %d/%d", cfg.run_name, min(start + batch_size, len(pending)), len(pending))


def _build_vlm(configs: list[RunConfig], model: str):
    """One process serves one model: the base model, or the merged model of one adapter."""
    adapters = {cfg.lora_path for cfg in configs}
    if len(adapters) != 1:
        raise ValueError(f"one process serves one model; got lora paths {sorted(adapters)}")
    adapter = adapters.pop()
    return VLLMQwen(resolve_model_path(model, adapter), allow_video=any(c.input_mode == "video" for c in configs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--split", required=True, help="dev, test or test_original")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", help="config overrides as key=value")
    parser.add_argument("--shard", default="0/1", help="i/n for parallel workers")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--splits-root", default="data/splits")
    parser.add_argument("--cache-root", default="cache")
    parser.add_argument("--results-root", default="results/quantitative/runs")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0, help="run only the first N samples")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    shard, total = parse_shard(args.shard)
    samples = load_agent_inputs(f"{args.splits_root}/questions_{args.split}.jsonl", args.splits_root)
    samples = samples[: args.limit] if args.limit else samples
    samples = samples[shard::total]

    overrides = parse_overrides(args.overrides)
    configs = [load_config(path, overrides) for path in args.config]
    services = build_services(_build_vlm(configs, args.model), args.cache_root)
    for cfg in configs:
        _run_config(cfg, samples, services, Path(args.results_root) / args.split, args.batch_size)


if __name__ == "__main__":
    main()
