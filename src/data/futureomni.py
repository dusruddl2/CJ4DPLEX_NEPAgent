"""FutureOmni annotations: option shuffling, stratified split, and question/answer separation.

Source: FutureOmni [4]. Changed from source: options are shuffled per item.
Agent-facing question files never contain the correct option; answers go to a separate file.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from src.schemas import AgentInput

_OPTION_PREFIX = re.compile(r"^\s*[A-Z][.)]\s*")
LETTERS = "ABCDEFGH"


@dataclass(frozen=True)
class Item:
    sample_id: str
    video: str
    t_split: float | None
    question: str
    options: tuple[str, ...]
    correct_index: int
    domain: str
    audio_type: str
    pattern: str


def parse_item(raw: dict[str, Any]) -> Item:
    letter = raw["answer"].strip()
    return Item(
        sample_id=f"fo_{int(raw['qid']):04d}",
        video=raw["video"],
        t_split=float(raw["seconds"]),
        question=raw["question"],
        options=tuple(_OPTION_PREFIX.sub("", option).strip() for option in raw["options"]),
        correct_index=LETTERS.index(letter),
        domain=raw["video_domain"],
        audio_type=raw["audio_type"],
        pattern=raw["forecasting_pattern"],
    )


def load_items(path: str | Path) -> list[Item]:
    with Path(path).open(encoding="utf-8") as handle:
        return [parse_item(raw) for raw in json.load(handle)]


def shuffled_order(sample_id: str, count: int, seed: int) -> list[int]:
    """Deterministic option permutation for one question."""
    order = list(range(count))
    random.Random(f"{seed}:{sample_id}").shuffle(order)
    return order


def _quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    """Largest-remainder allocation of `target` proportional to `counts`."""
    total = sum(counts.values())
    exact = {key: target * value / total for key, value in counts.items()}
    quota = {key: int(value) for key, value in exact.items()}
    remainder = sorted(exact, key=lambda key: (exact[key] - quota[key], key), reverse=True)
    for key in remainder[: target - sum(quota.values())]:
        quota[key] += 1
    return quota


def stratified_split(
    items: Sequence[Item],
    dev_size: int,
    test_size: int,
    seed: int,
    key: Callable[[Item], str] = lambda item: item.domain,
) -> tuple[list[Item], list[Item]]:
    """Split by source video so that no video appears in both dev and test.

    `key` names the stratum whose proportions are kept. A video belongs to one stratum, taken
    from its first item, so videos whose items differ in `key` are never split across dev/test.
    """
    by_video: dict[str, list[Item]] = defaultdict(list)
    for item in sorted(items, key=lambda i: i.sample_id):
        by_video[item.video].append(item)
    by_stratum_video: dict[str, dict[str, list[Item]]] = defaultdict(dict)
    for video, group in by_video.items():
        by_stratum_video[key(group[0])][video] = group

    counts = {stratum: sum(len(v) for v in videos.values()) for stratum, videos in by_stratum_video.items()}
    dev_quota = _quotas(counts, dev_size)
    test_quota = _quotas(counts, test_size)
    rng = random.Random(seed)
    dev: list[Item] = []
    test: list[Item] = []
    for stratum in sorted(by_stratum_video):
        videos = sorted(by_stratum_video[stratum])
        rng.shuffle(videos)
        stratum_dev: list[Item] = []
        stratum_test: list[Item] = []
        for video in videos:
            group = by_stratum_video[stratum][video]
            if len(stratum_dev) + len(group) <= dev_quota[stratum]:
                stratum_dev.extend(group)
            elif len(stratum_test) + len(group) <= test_quota[stratum]:
                stratum_test.extend(group)
        dev.extend(stratum_dev)
        test.extend(stratum_test)
    return sorted(dev, key=lambda i: i.sample_id), sorted(test, key=lambda i: i.sample_id)


def to_records(item: Item, seed: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (question record, label record). seed=None keeps the published option order."""
    count = len(item.options)
    order = list(range(count)) if seed is None else shuffled_order(item.sample_id, count, seed)
    options = [[LETTERS[new], item.options[old]] for new, old in enumerate(order)]
    question = {
        "sample_id": item.sample_id,
        "video": item.video,
        "t_split": item.t_split,
        "question": item.question,
        "options": options,
    }
    label = {
        "sample_id": item.sample_id,
        "correct": LETTERS[order.index(item.correct_index)],
        "domain": item.domain,
        "audio_type": item.audio_type,
        "pattern": item.pattern,
        "video": item.video,
    }
    return question, label


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_agent_inputs(questions_path: str | Path, splits_root: str | Path) -> list[AgentInput]:
    """Build agent inputs from the answer-free question file."""
    root = Path(splits_root)
    return [
        AgentInput(
            sample_id=row["sample_id"],
            prefix_path=str(root / row["sample_id"] / "prefix.mp4"),
            question=row["question"],
            options=tuple((letter, text) for letter, text in row["options"]),
        )
        for row in read_jsonl(questions_path)
    ]
