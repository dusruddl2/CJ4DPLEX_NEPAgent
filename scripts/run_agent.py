#!/usr/bin/env python
"""Run the agent on one video: predict the next event, and save the result JSON and a timeline figure.

Video in, prediction out (no question or options needed). The whole file is the observed clip:
  python scripts/run_agent.py --video clip.mp4
Only the first 12 seconds are observed (a split point at or beyond the end of the file means the whole file):
  python scripts/run_agent.py --video clip.mp4 --t_split 12
With a question and answer options the predicted sentence is also matched onto the options:
  python scripts/run_agent.py --video clip.mp4 --t_split 12 --question "What happens next?" \
      --options "He sits down" "He opens the door" "He leaves" "He falls"
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.agent.config import load_config
from src.agent.figure import draw_timeline
from src.agent.output import write_result
from src.agent.pipeline import build_services, run_batch
from src.schemas import AgentInput
from src.data.splitter import observed_clip
from src.models.audio import AudioAnalyzer
from src.models.vlm import DEFAULT_MODEL, VLLMQwen, resolve_model_path
from src.stages.questions import GENERIC_QUESTION

LETTERS = "ABCDEFGH"


def make_sample(sample_id: str, prefix_path: str, question: str = GENERIC_QUESTION, options: list[str] | tuple[str, ...] = ()) -> AgentInput:
    """One input; without options the agent outputs the predicted sentence only."""
    return AgentInput(sample_id=sample_id, prefix_path=prefix_path, question=question, options=tuple(zip(LETTERS, options)))


def summary(result: dict) -> dict:
    """What is printed for one run: the reported prediction and the reasoning record behind it."""
    trace = result["trace"]["nep_agent"]
    entry = trace["reasoning"].get(trace["final_source"]) or {}
    return {"decision": decision(result), "prediction": result["prediction"], "final_source": trace["final_source"], "reasoning": entry.get("record"),
            "tool": {"selection": entry.get("selection"), "observation": entry.get("observation")} if "selection" in entry else None}


def decision(result: dict) -> str:
    """One line saying why the reported answer is the Direct one or a later candidate."""
    routing = (result["trace"].get("direct") or {}).get("routing") or {}
    source = result["trace"]["nep_agent"]["final_source"]
    confidence, tau = routing.get("generation_confidence"), routing.get("gate_tau")
    if confidence is None or tau is None:
        return f"reported answer: {source}"
    if not routing.get("routed"):
        return f"Direct answer kept: its confidence {confidence:.2f} is at or above the threshold {tau:g}; no tool or reasoning call was made"
    if source == "direct_fallback":
        return f"Direct answer kept as fallback: confidence {confidence:.2f} is below the threshold {tau:g}, but no later stage produced an answer"
    return f"Direct confidence {confidence:.2f} is below the threshold {tau:g}; reported answer: {source}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--t_split", type=float, default=None, help="seconds observed by the agent; default: the whole file")
    parser.add_argument("--question", default=GENERIC_QUESTION)
    parser.add_argument("--options", nargs="*", default=[], help="answer options; omit to get the predicted sentence only")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--config", default="configs/nep_agent.yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--work-dir", default="data/splits")
    parser.add_argument("--cache-root", default="cache")
    parser.add_argument("--output-dir", default="results/single")
    args = parser.parse_args()

    if args.options and not 2 <= len(args.options) <= len(LETTERS):
        parser.error(f"--options needs 2 to {len(LETTERS)} entries, or none")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    sample_id = args.sample_id or Path(args.video).stem
    prefix = observed_clip(args.video, args.t_split, Path(args.work_dir) / sample_id)
    sample = make_sample(sample_id, str(prefix), args.question, args.options)
    cfg = load_config(args.config)
    audio = AudioAnalyzer() if cfg.use_audio else None
    services = build_services(VLLMQwen(resolve_model_path(args.model, cfg.lora_path)), args.cache_root, audio)
    result = run_batch([sample], cfg, services)[0]
    path = write_result(result, args.output_dir)
    logging.info("result: %s", path)
    if result["prediction"]:
        figure = path.with_suffix(".png")
        draw_timeline(result, figure)
        logging.info("figure: %s", figure)
    logging.info(decision(result))
    logging.info(json.dumps(summary(result), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
