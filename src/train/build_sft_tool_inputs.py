"""Reasoning SFT inputs with tool evidence (the tool-evidence training set).

Same 1,500 clips and the same Teacher targets as cache/sft/reasoning_sft_no_chain. For each clip the input is built with the
inference code path: 32 sampled frames, and for about half of the clips one tool is applied first (terminal_state 80%,
dense_resample on the last 8 frames 20%), so the prompt and the image list are what the reasoning call receives after a tool.
The other half keeps the clip-only input. Targets are unchanged.
Usage: PYTHONPATH=. python -m src.train.build_sft_tool_inputs --workers 16
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SRC = Path("cache/sft/reasoning_sft_no_chain")
OUT = Path("cache/sft/reasoning_sft_tool_inputs")
CACHE = Path("cache/sft/frames_tool")
SEED = 0


def build(job: tuple[int, dict, str, str]) -> dict | None:
    index, item, sample_id, video = job
    from src.agent import observer, prompting, tools
    from src.agent.config import load_config
    from src.agent.observer import FrameIO
    from src.stages.evidence import evidence_registry
    from src.nep_agent import event_reasoning as er
    from src.schemas import AgentInput

    cfg = load_config("configs/nep_agent.yaml")
    io = FrameIO(CACHE)
    user = item["messages"][0]["content"]
    question = re.search(r"\nQuestion: (.*)\n", user).group(1)
    state = observer.sample_frames(AgentInput(sample_id, video, question, ()), cfg, io)
    if not state.alive or len(state.frames) != 32:
        return None
    rng = random.Random(SEED * 100003 + index)
    mode = "clip"
    if rng.random() < 0.5:
        ids = observer.frame_ids_of(state)
        if rng.random() < 0.8:
            state, new = tools.apply_terminal_frames(state, cfg, io); mode = "terminal_state"
        else:
            state, new = tools.apply_dense_resample(state, {"where": [ids[-8], ids[-1]]}, cfg, io); mode = "dense_resample"
        if not new:
            mode = "clip"
    common = {"preamble": prompting.frame_preamble(observer.frame_ids_of(state), cfg.use_vp), "audio": "Audio: not available.",
              "question_block": f"Question: {question}"}
    menu = "\n".join(tools.TOOL_DESCRIPTIONS[t] for t in tools.available_tools(state, cfg))
    text = er.render_reason("causal_sft_nc", common, state.frames[0]["frame_id"], state.frames[-1]["frame_id"], menu)
    if mode != "clip":
        text += "\n" + prompting.extra_block(state.extra_evidence)
    images = [e["path"] for e in evidence_registry(state).values()]
    return {"messages": [{"role": "user", "content": "<image>" * len(images) + text}, item["messages"][1]], "images": images,
            "_mode": mode, "_same_text_as_before": (mode == "clip" and ("<image>" * 32 + text) == user)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    items = json.loads((SRC / "train.json").read_text(encoding="utf-8"))
    videos = {}
    for line in Path("cache/sft/stage1_nep_sft/records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line); videos[r["sample_id"]] = r["video"]
    jobs = []
    for i, item in enumerate(items):
        sid = re.search(r"/frames_nep/([^/]+)/", item["images"][0]).group(1)
        jobs.append((i, item, sid, videos[sid]))
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(build, jobs, chunksize=8))
    kept = [r for r in rows if r]
    modes = {m: sum(r["_mode"] == m for r in kept) for m in ("clip", "terminal_state", "dense_resample")}
    same = sum(r.pop("_same_text_as_before") for r in kept)
    for r in kept:
        r.pop("_mode")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.json").write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    shutil.copy(SRC / "dataset_info.json", OUT / "dataset_info.json")
    (OUT / "manifest.json").write_text(json.dumps({"source": str(SRC), "items": len(kept), "dropped": len(rows) - len(kept), "modes": modes,
                                                    "clip_only_prompts_identical_to_source": same, "seed": SEED}, indent=1), encoding="utf-8")
    print(json.dumps({"items": len(kept), "dropped": len(rows) - len(kept), "modes": modes, "clip_only_prompts_identical_to_source": same}))


if __name__ == "__main__":
    main()
