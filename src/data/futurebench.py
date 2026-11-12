"""FutureBench (NEP [1]) as a secondary benchmark: loader, license filter, split, and leak heuristics.

Source: Wang et al., "Fostering Video Reasoning via Next-Event Prediction" [1], FutureBench.
Changed from source: charades and ego4d clips are excluded (non-commercial / signed-access
licenses), options are shuffled per item with a fixed seed, and the split keeps the hop and
data-source mix while separating videos. Question text may describe later scenes by design;
see `leak_summary` and section 5 of README.md.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.data.futureomni import LETTERS, Item, stratified_split

EXCLUDED_SOURCES = frozenset({"charades", "ego4d"})
AUDIO_TYPE = "n/a"
VIDEO_PATH_ANCHOR = "first_part_video"
OPTION_COUNT = 4

# Phrases that describe scenes after the observed prefix, grouped by what they reveal.
LEAK_PATTERNS = {
    "fill_in_gap": re.compile(r"\bfill(?:ing)? in\b|\bmissing (?:intermediate )?(?:event|step)", re.IGNORECASE),
    "numbered_slots": re.compile(r"\b\d\.\s*\[\?\]"),
    "final_scene_described": re.compile(
        r"\b(?:final|last|closing|concluding|ending) (?:scene|clip|shot|frame|event|step|moment|image|reveal)s?\b"
        r"|\bfinale\b|\bwraps up\b|\bconclud(?:es|ing) with\b",
        re.IGNORECASE,
    ),
    "described_future_step": re.compile(r"\b\d\.\s+(?!\[\?\])[A-Z]"),
}


def stratum(item: Item) -> str:
    """Stratify on data source and hop jointly."""
    return f"{item.domain}|{item.pattern}"


def relative_video_path(video_path: str) -> str:
    """Return the `first_part_video/<source>/<file>` suffix used to locate the clip."""
    index = video_path.find(VIDEO_PATH_ANCHOR)
    if index < 0:
        raise ValueError(f"video_path has no '{VIDEO_PATH_ANCHOR}' component: {video_path}")
    return video_path[index:]


def parse_futurebench_item(raw: dict[str, Any], index: int) -> Item:
    qa = raw["qa"]
    options = qa["Options"]
    letters = tuple(sorted(options))
    if letters != tuple(LETTERS[: len(letters)]) or len(letters) != OPTION_COUNT:
        raise ValueError(f"{raw.get('id')}: unexpected option letters {letters}")
    answer = qa["Answer"].strip()
    if answer not in options:
        raise ValueError(f"{raw.get('id')}: answer {answer} not among options")
    return Item(
        sample_id=f"fb_{index:04d}",
        video=relative_video_path(raw["video_path"]),
        t_split=None,
        question=qa["Question"].strip(),
        options=tuple(options[letter].strip() for letter in letters),
        correct_index=LETTERS.index(answer),
        domain=raw["data_source"],
        audio_type=AUDIO_TYPE,
        pattern=raw["question_type"],
    )


def load_futurebench_items(
    path: str | Path, excluded_sources: frozenset[str] = EXCLUDED_SOURCES
) -> list[Item]:
    """Parse every record (index = position in the file) and drop excluded sources."""
    with Path(path).open(encoding="utf-8") as handle:
        raw_items = json.load(handle)
    items = [parse_futurebench_item(raw, index) for index, raw in enumerate(raw_items)]
    return [item for item in items if item.domain not in excluded_sources]


def split_futurebench(
    items: Sequence[Item], dev_size: int, test_size: int, seed: int
) -> tuple[list[Item], list[Item]]:
    return stratified_split(items, dev_size, test_size, seed, key=stratum)


def answer_letter_counts(items: Iterable[Item]) -> dict[str, int]:
    return dict(sorted(Counter(LETTERS[item.correct_index] for item in items).items()))


def longest_option_is_correct_rate(items: Sequence[Item]) -> float:
    """Share of items whose longest option is the correct one (a known FutureBench bias)."""
    if not items:
        return 0.0
    hits = sum(max(range(len(i.options)), key=lambda k: len(i.options[k])) == i.correct_index for i in items)
    return hits / len(items)


def leak_flags(question: str) -> dict[str, bool]:
    return {name: bool(pattern.search(question)) for name, pattern in LEAK_PATTERNS.items()}


def leak_summary(items: Sequence[Item]) -> dict[str, dict[str, Any]]:
    """Per hop: item count, share matching any pattern, and share per pattern."""
    by_hop: dict[str, list[Item]] = {}
    for item in items:
        by_hop.setdefault(item.pattern, []).append(item)
    summary: dict[str, dict[str, Any]] = {}
    for hop in sorted(by_hop):
        group = by_hop[hop]
        flags = [leak_flags(item.question) for item in group]
        summary[hop] = {
            "n": len(group),
            "any": round(sum(any(f.values()) for f in flags) / len(group), 4),
            **{name: round(sum(f[name] for f in flags) / len(group), 4) for name in LEAK_PATTERNS},
        }
    return summary


def distribution_report(items: Sequence[Item], dev: Sequence[Item], test: Sequence[Item]) -> dict[str, Any]:
    def mix(group: Sequence[Item]) -> dict[str, dict[str, int]]:
        return {
            "source": dict(sorted(Counter(i.domain for i in group).items())),
            "hop": dict(sorted(Counter(i.pattern for i in group).items())),
        }

    return {
        "n_after_license_filter": len(items),
        "answer_letters_original": answer_letter_counts(items),
        "longest_option_is_correct": round(longest_option_is_correct_rate(items), 4),
        "mix_all": mix(items),
        "mix_dev": mix(dev),
        "mix_test": mix(test),
        "leak_heuristics_by_hop": leak_summary(items),
    }
