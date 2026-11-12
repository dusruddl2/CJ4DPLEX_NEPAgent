"""Answer-only SFT data following the NEP paper's SFT recipe (Wang et al., "Fostering Video Reasoning via Next-Event Prediction").

Source: sail-sg/Video-Next-Event-Prediction, `video_data_generation.py::convert_to_sft` (MIT). Copied as-is: the question
pool, the input = first-part clip, the target = `video_split.split_caption.caption_part2` (caption of the future segment),
random question per sample, shuffled output. Changed from source: frames are our evaluation format
(uniform frames of the first-part clip, VP-numbered, 448^2 cap) instead of LLaMA-Factory `<video>` loading at
video_max_pixels=16384; sources Charades/Ego4D are excluded (license, as for evaluation); every video that appears in any
FutureBench evaluation file is excluded (all 866 FutureBench videos are inside V1-33K), and the build stops if those
evaluation files are missing; `--limit` items are built (the paper's comparison runs use 3K) and the first `--keep-first`
rows of the shuffled output are written as the training set (default 1,500).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.agent import observer, prompting
from src.agent.observer import FrameIO
from src.data.futurebench import EXCLUDED_SOURCES
from src.train.sft_format import fit_images, has_reserved_token, sft_record, write_dataset, write_manifest

logger = logging.getLogger("build_sft_nep")

DATASET_NAME = "futurebench_nep_sft"
PROGRESS_EVERY = 200
V1_33K_ROOT = Path(os.environ.get("V1_33K_ROOT", "data/raw/v1_33k"))  # V1-33K release: annotations and first-part clips
EVAL_QUESTION_FILES = ("data/splits/questions_fb_full.jsonl", "data/splits/questions_fb_test.jsonl",
                       "data/splits/questions_fb_dev.jsonl")
# Verbatim system prompts from the official repository (convert_to_distill / convert_to_CFT).
DISTILL_SYSTEM = ("You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the "
                  "reasoning process as an internal monologue and then provide the user with the answer. Respond in the following "
                  "format: <think>\n...\n</think>\n<answer>\n...\n</answer>")
CFT_SYSTEM = "Please critique whether the following solution to the question is correct."
CFT_WRONG_RATIO = 2.1  # repository default wrong/correct ratio
STRATEGIES = ("sft", "distill", "cft")
# Verbatim from the official repository (including its missing comma after the first entry, which merges the first two
# strings into one question "what will happen next?What comes next?"). We keep the pool but fix that typo.
QUESTION_POOL = [
    "what will happen next?", "What comes next?", "What is going to happen next?", "What follows from here?",
    "What's on the horizon?", "What happens after this?", "What comes after this?", "What is the next step?",
    "What is about to occur?", "What will follow?", "What is the subsequent event?", "What unfolds next?",
    "What is next in line?", "What is in store?", "What does the future hold for this?", "What's coming up next?",
    "What will transpire next?", "What is the next occurrence?", "What should we expect next?",
    "What will happen subsequently?", "What does the next chapter bring?", "What’s the next move?",
    "What comes around the corner?", "What lies ahead?", "What is planned next?", "What will the future reveal?",
]


@dataclass(frozen=True)
class BuildConfig:
    video_root: Path
    frame_root: Path
    num_frames: int = 16
    seed: int = 42


def video_key(path: str) -> str:
    return Path(path).name


def eval_videos(paths: Sequence[str | Path] = EVAL_QUESTION_FILES) -> set[str]:
    """Videos of every evaluation question file; these never enter the training set."""
    keys: set[str] = set()
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:  # without these files evaluation videos could enter the training set unnoticed
        raise FileNotFoundError(f"evaluation question files are missing: {missing}; build the splits first (scripts/build_full_splits.py)")
    for path in paths:
        path = Path(path)
        for line in path.open(encoding="utf-8"):
            if line.strip():
                keys.add(video_key(json.loads(line)["video"]))
    return keys


def load_nep_items(annotations: str | Path, video_root: Path, eval_paths: Sequence[str | Path] = EVAL_QUESTION_FILES,
                   seed: int = 42, limit: int = 3000, excluded_sources: frozenset[str] = EXCLUDED_SOURCES) -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw = json.loads(Path(annotations).read_text(encoding="utf-8"))
    held = {Path(v).stem for v in eval_videos(eval_paths)}
    counts = {"raw": len(raw), "excluded_source": 0, "eval_video": 0, "no_target": 0, "video_missing": 0}
    usable = []
    for row in raw:
        if row["data_source"] in excluded_sources:
            counts["excluded_source"] += 1
            continue
        if row["id"] in held:
            counts["eval_video"] += 1
            continue
        target = (((row.get("video_split") or {}).get("split_caption") or {}).get("caption_part2") or "").strip()
        if not target:
            counts["no_target"] += 1
            continue
        video = video_root / "first_part_video" / row["data_source"] / f"{row['id']}.mp4"
        if not video.exists():
            counts["video_missing"] += 1
            continue
        ds = row.get("ds_reasoning") or {}
        crit = row.get("o3_critique") or {}
        usable.append({"sample_id": f"nep_{row['id']}", "video": str(video), "target": target, "source": row["data_source"],
                       "reasoning": str(ds.get("rewritten_reasoning_content", "")).strip(),
                       "prediction": str(ds.get("rewritten_prediction_content", "")).strip(),
                       "critique": str(crit.get("Critique", "")).strip(), "conclusion": str(crit.get("Conclusion", "")).strip()})
    counts["usable"] = len(usable)
    rng = random.Random(seed)
    rng.shuffle(usable)
    counts["usable"] = len(usable)
    return usable, counts


def select_items(usable: list[dict[str, Any]], strategy: str, limit: int, offset: int, seed: int) -> list[dict[str, Any]]:
    """Window [offset, offset+limit) of the seeded shuffle. mix = equal thirds of sft/distill/cft (disjoint items, paper's
    'equal combination'); cft items are resampled toward the repository's wrong/correct ratio of 2.1 and need a critique."""
    if strategy != "mix":
        pool = usable[offset:]
        if strategy in {"distill", "cft"}:
            pool = [i for i in pool if i["reasoning"] and i["prediction"] and (strategy == "distill" or i["critique"])]
        if strategy == "cft":
            pool = _cft_ratio(pool, limit, seed)
        return [{**i, "strategy": strategy} for i in pool[:limit]]
    third = limit // 3
    taken: set[str] = set()
    out: list[dict[str, Any]] = []
    for name in STRATEGIES:
        pool = [i for i in usable[offset:] if i["sample_id"] not in taken]
        part = select_items(pool, name, third, 0, seed)
        taken.update(i["sample_id"] for i in part)
        out += part
    return out


def _cft_ratio(pool: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    wrong = [i for i in pool if i["conclusion"].lower() == "wrong"]
    other = [i for i in pool if i["conclusion"].lower() != "wrong"]
    n_wrong = min(len(wrong), int(round(limit * CFT_WRONG_RATIO / (1 + CFT_WRONG_RATIO))))
    n_other = min(len(other), limit - n_wrong)
    rng = random.Random(f"cft:{seed}")
    picked = rng.sample(wrong, n_wrong) + rng.sample(other, n_other)
    rng.shuffle(picked)
    return picked


def make_target(item: dict[str, Any], question: str) -> tuple[str, str, str]:
    """Return (user_text, target, system) for the item's strategy, copying the repository's formats."""
    strategy = item.get("strategy", "sft")
    if strategy == "sft":
        return question, item["target"], ""
    solution = f"<think>\n{item['reasoning']}\n</think>\n<answer>\n{item['prediction']}\n</answer>"
    if strategy == "distill":
        return question, solution, DISTILL_SYSTEM
    user = f"Question:\n{question}\n\nSolution:\n\n{solution}"
    target = f"Critique:\n{item['critique']}\n\nConclusion:\n{item['conclusion']}."
    return user, target, CFT_SYSTEM


def build_one(item: dict[str, Any], cfg: BuildConfig, rng_seed: int) -> dict[str, Any]:
    from decord import VideoReader, cpu

    try:
        reader = VideoReader(item["video"], ctx=cpu(0), num_threads=1)
        indices = observer.uniform_indices(len(reader), cfg.num_frames)
        del reader
        io = FrameIO(cfg.frame_root)
        stored = observer._store_frames(item["sample_id"], item["video"], indices, io)
    except (RuntimeError, ValueError, OSError) as exc:
        return {"sample_id": item["sample_id"], "status": "decode_error", "detail": str(exc)}
    frames = observer.number_frames(item["sample_id"], [{"t_sec": t, "raw_path": p, "caption": ""} for t, p in stored], use_vp=True, io=io)
    question = random.Random(f"{rng_seed}:{item['sample_id']}").choice(QUESTION_POOL)
    user_text, target, system = make_target(item, question)
    preamble = prompting.frame_preamble([f["frame_id"] for f in frames], True)
    prompt = f"{preamble}\n{user_text}"
    if has_reserved_token(prompt) or has_reserved_token(target):
        return {"sample_id": item["sample_id"], "status": "reserved_token_in_text"}
    images = fit_images([f["path"] for f in frames], cfg.frame_root / item["sample_id"] / "fit")
    return {"sample_id": item["sample_id"], "status": "ok", "video": item["video"], "source": item["source"],
            "strategy": item.get("strategy", "sft"), "record": sft_record(prompt, target, images, system)}


def _star(payload: tuple[dict[str, Any], BuildConfig, int]) -> dict[str, Any]:
    return build_one(*payload)


def build_dataset(annotations: str | Path, video_root: Path, out_dir: Path, cfg: BuildConfig, workers: int, limit: int,
                  strategy: str = "sft", offset: int = 0, dataset_name: str = DATASET_NAME,
                  excluded_sources: frozenset[str] = EXCLUDED_SOURCES, keep_first: int = 0) -> dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor

    usable, counts = load_nep_items(annotations, video_root, seed=cfg.seed, excluded_sources=excluded_sources)
    items = select_items(usable, strategy, limit, offset, cfg.seed)
    counts["chosen"] = len(items)
    counts["by_strategy"] = {k: sum(i.get("strategy") == k for i in items) for k in STRATEGIES}
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "records.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if progress.exists():
        for line in progress.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                done[row["sample_id"]] = row
    pending = [item for item in items if item["sample_id"] not in done]
    logger.info("%d chosen, %d already built, %d to build; counts %s", len(items), len(items) - len(pending), len(pending), counts)
    failures: dict[str, int] = {}
    with progress.open("a", encoding="utf-8") as handle, ProcessPoolExecutor(max_workers=workers) as pool:
        for number, result in enumerate(pool.map(_star, [(item, cfg, cfg.seed) for item in pending], chunksize=4), 1):
            if result["status"] == "ok":
                done[result["sample_id"]] = result
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
            else:
                failures[result["status"]] = failures.get(result["status"], 0) + 1
            if number % PROGRESS_EVERY == 0:
                logger.info("%d/%d, failures %s", number, len(pending), failures)
    wanted = [item["sample_id"] for item in items]
    rows = [done[s] for s in wanted if s in done]
    random.Random(cfg.seed).shuffle(rows)
    if not rows:
        raise RuntimeError(f"no usable samples; failures {failures}")
    rows = rows[:keep_first] if keep_first else rows
    write_dataset(out_dir, dataset_name, [row["record"] for row in rows])
    manifest = {"dataset": dataset_name, "strategy": strategy, "offset": offset, "keep_first": keep_first,
                "recipe": "NEP paper video_data_generation.py (convert_to_sft / convert_to_distill / convert_to_CFT)", "counts": counts,
                "written": len(rows), "failures_this_run": failures, "seed": cfg.seed, "num_frames": cfg.num_frames,
                "sources": {s: sum(r["source"] == s for r in rows) for s in sorted({r["source"] for r in rows})},
                "mean_target_chars": sum(len(r["record"]["messages"][1]["content"]) for r in rows) / len(rows)}
    write_manifest(out_dir, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(V1_33K_ROOT / "video_reasoning_future_prediction_v1.json"))
    parser.add_argument("--video-root", default=str(V1_33K_ROOT / "first_segment_video"))
    parser.add_argument("--out-dir", default="cache/sft/stage1_nep_sft")
    parser.add_argument("--frame-root", default="cache/sft/frames_nep")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=3000, help="items built from the seeded shuffle")
    parser.add_argument("--keep-first", type=int, default=1500, help="rows of the shuffled output written to train.json (0 = all)")
    parser.add_argument("--strategy", choices=[*STRATEGIES, "mix"], default="sft")
    parser.add_argument("--offset", type=int, default=0, help="start of the window in the seeded shuffle (disjoint datasets)")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = BuildConfig(Path(args.video_root), Path(args.frame_root), args.num_frames, args.seed)
    manifest = build_dataset(args.annotations, Path(args.video_root), Path(args.out_dir), cfg, args.workers, args.limit,
                             strategy=args.strategy, offset=args.offset, dataset_name=args.dataset_name, keep_first=args.keep_first)
    logger.info("manifest: %s", json.dumps(manifest, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
