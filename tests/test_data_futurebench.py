"""FutureBench loader, license filter, split, bias measures, and leak heuristics."""
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from src.data.futureomni import LETTERS, read_jsonl, to_records
from src.data.futurebench import (
    EXCLUDED_SOURCES,
    answer_letter_counts,
    leak_flags,
    leak_summary,
    load_futurebench_items,
    longest_option_is_correct_rate,
    parse_futurebench_item,
    relative_video_path,
    split_futurebench,
)
from src.data.prefix_prep import prepare_linked_prefixes
from src.utils.media import ffmpeg_binary

ROOT = Path(__file__).parents[1]
REAL_FILE = Path(os.environ.get("V1_33K_ROOT", ROOT / "data/raw/v1_33k")) / "futurebench.json"
FB_DEV = ROOT / "data/splits/questions_fb_dev.jsonl"
FB_TEST = ROOT / "data/splits/questions_fb_test.jsonl"


def _raw(index: int, source: str = "youtube", hop: str = "hop1", answer: str = "A", video: str = "v1.mp4") -> dict:
    return {
        "id": f"id{index}", "data_source": source, "question_type": hop,
        "video_path": f"/x/first_segment_video/first_part_video/{source}/{video}",
        "qa": {
            "Question": "Based on the given video, fill in 1. [?] 2. The final scene shows a dog. Which option?",
            "Options": {"A": "a long long option", "B": "b", "C": "c", "D": "d"},
            "Answer": answer, "Explanation": {},
        },
    }


def test_parse_item_fields() -> None:
    item = parse_futurebench_item(_raw(3, answer="C"), 3)
    assert item.sample_id == "fb_0003"
    assert item.video == "first_part_video/youtube/v1.mp4"
    assert item.t_split is None
    assert item.correct_index == 2
    assert (item.domain, item.audio_type, item.pattern) == ("youtube", "n/a", "hop1")


def test_relative_video_path_requires_anchor() -> None:
    with pytest.raises(ValueError):
        relative_video_path("/no/anchor/here.mp4")


def test_rejects_answer_outside_options() -> None:
    with pytest.raises(ValueError):
        parse_futurebench_item(_raw(0, answer="E"), 0)


def test_load_excludes_licensed_sources(tmp_path: Path) -> None:
    rows = [_raw(0), _raw(1, source="charades"), _raw(2, source="ego4d"), _raw(3, source="nextqa")]
    path = tmp_path / "fb.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    items = load_futurebench_items(path)
    assert [i.sample_id for i in items] == ["fb_0000", "fb_0003"]
    assert EXCLUDED_SOURCES == {"charades", "ego4d"}


def test_shuffle_moves_the_label_consistently() -> None:
    item = parse_futurebench_item(_raw(0, answer="A"), 0)
    question, label = to_records(item, seed=42)
    assert "correct" not in question and question["t_split"] is None
    assert dict(question["options"])[label["correct"]] == "a long long option"
    assert to_records(item, seed=42) == (question, label)


def test_bias_measures() -> None:
    items = [parse_futurebench_item(_raw(i, answer="A"), i) for i in range(3)]
    items.append(parse_futurebench_item(_raw(3, answer="B"), 3))
    assert answer_letter_counts(items) == {"A": 3, "B": 1}
    assert longest_option_is_correct_rate(items) == 0.75


def test_leak_flags_and_summary() -> None:
    flags = leak_flags(_raw(0)["qa"]["Question"])
    assert flags["fill_in_gap"] and flags["numbered_slots"] and flags["final_scene_described"]
    assert flags["described_future_step"]
    assert not any(leak_flags("What happens next?").values())
    summary = leak_summary([parse_futurebench_item(_raw(0), 0)])
    assert summary["hop1"]["n"] == 1 and summary["hop1"]["any"] == 1.0


def test_split_separates_videos_and_keeps_mix() -> None:
    rows = []
    index = 0
    for source in ("youtube", "nextqa"):
        for hop in ("hop1", "hop5"):
            for video in range(12):
                rows.append(_raw(index, source=source, hop=hop, video=f"{source}{hop}{video}.mp4"))
                index += 1
    rows.append(_raw(index, video="youtubehop10.mp4", hop="hop5"))  # same video, different hop
    items = [parse_futurebench_item(raw, i) for i, raw in enumerate(rows)]

    dev, test = split_futurebench(items, 8, 24, 42)

    assert len(dev) == 8 and len(test) == 24
    assert not {i.video for i in dev} & {i.video for i in test}
    assert Counter(i.domain for i in test) == {"youtube": 12, "nextqa": 12}
    assert dev == split_futurebench(items, 8, 24, 42)[0]


def test_prepare_linked_prefixes_links_and_reports(tmp_path: Path) -> None:
    root = tmp_path / "videos" / "first_part_video" / "youtube"
    root.mkdir(parents=True)
    subprocess.run(
        [ffmpeg_binary(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=2:size=64x48:rate=5",
         "-pix_fmt", "yuv420p", str(root / "v1.mp4")],
        check=True, timeout=120,
    )
    questions = tmp_path / "q.jsonl"
    rows = [
        {"sample_id": "fb_0000", "video": "first_part_video/youtube/v1.mp4", "t_split": None, "question": "q", "options": []},
        {"sample_id": "fb_0001", "video": "first_part_video/youtube/none.mp4", "t_split": None, "question": "q", "options": []},
    ]
    questions.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    splits, report_path = tmp_path / "splits", tmp_path / "r.json"

    report = prepare_linked_prefixes([questions], tmp_path / "videos", splits, report_path)

    assert (splits / "fb_0000" / "prefix.mp4").is_symlink()
    assert report["linked"] == 1 and report["without_audio"] == 1
    assert [f["sample_id"] for f in report["failures"]] == ["fb_0001"]
    assert prepare_linked_prefixes([questions], tmp_path / "videos", splits, report_path)["resumed_existing"] == 1


def test_cli_futurebench_mode_reports_missing_root(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/prepare_prefix.py"), "--benchmark", "futurebench",
         "--video-root", str(tmp_path / "nothing"), "--splits-root", str(tmp_path / "s"), "--report", str(tmp_path / "r.json")],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 2 and "not ready" in result.stderr


def test_real_file_loads_and_filters() -> None:
    if not REAL_FILE.exists():
        pytest.skip(f"missing {REAL_FILE}; set V1_33K_ROOT to the V1-33K release")
    items = load_futurebench_items(REAL_FILE)
    assert len(items) == 1056 - 95
    assert not {i.domain for i in items} & EXCLUDED_SOURCES
    assert all(len(i.options) == 4 for i in items)


def test_real_split_files_exist_and_are_consistent() -> None:
    if not (ROOT / "data").exists():
        pytest.skip("data/ is not part of this tree; build it with scripts/build_splits_fb.py")
    for path in (FB_DEV, FB_TEST):
        assert path.exists(), f"missing {path}; run scripts/build_splits_fb.py"
    dev, test = read_jsonl(FB_DEV), read_jsonl(FB_TEST)
    assert (len(dev), len(test)) == (40, 200)
    assert not {r["video"] for r in dev} & {r["video"] for r in test}
    assert all(set(r) == {"sample_id", "video", "t_split", "question", "options"} for r in dev + test)
    labels = read_jsonl(ROOT / "data/labels/labels_fb_test.jsonl")
    assert all(l["correct"] in LETTERS[:4] and l["audio_type"] == "n/a" for l in labels)
