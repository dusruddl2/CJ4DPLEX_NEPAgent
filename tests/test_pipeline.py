"""Run the pipeline end to end with a scripted VLM on a synthetic video, without a GPU, plus small utility checks."""
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from src.agent.observer import range_midpoints, uniform_indices
from src.agent.parsing import extract_json
from src.agent.pipeline import AUDIO_CACHE_KEY, build_services, run_batch
from src.schemas import AgentInput, RunConfig
from src.utils.media import ffmpeg_binary
from src.data.futureomni import Item, shuffled_order, stratified_split
from src.models.vlm import MockVLM as ScriptedVLM


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("clip") / "prefix.mp4"
    subprocess.run(
        [ffmpeg_binary(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=8:size=320x240:rate=10", str(path)],
        check=True, timeout=120,
    )
    return str(path)


def _sample(clip: str) -> AgentInput:
    return AgentInput("s1", clip, "What next?", (("A", "one"), ("B", "two"), ("C", "three")))


def _config(**changes) -> RunConfig:
    return replace(RunConfig(run_name="t", num_frames=8, use_audio=False, reasoning_variants=""), **changes)


def test_direct_step_returns_a_prediction_from_real_sampled_frames(clip: str, tmp_path: Path) -> None:
    services = build_services(ScriptedVLM(), tmp_path)
    result = run_batch([_sample(clip)], _config(), services)[0]
    assert result["status"] == "ok" and result["prediction"]["answer"] in "ABC"
    assert result["trace"]["frames_used"] == 8 and result["trace"]["vlm_calls"] == 2
    assert len(services.vlm.requests[0].images) == 8 and services.vlm.requests[-1].images == ()


def test_undecodable_video_is_reported_not_raised(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    result = run_batch([_sample(str(broken))], _config(), build_services(ScriptedVLM(), tmp_path))[0]
    assert result["status"] == "video_error"


def test_audio_cache_is_required_when_no_model_is_loaded(clip: str, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        run_batch([_sample(clip)], _config(use_audio=True), build_services(ScriptedVLM(), tmp_path))


def test_cached_audio_is_used(clip: str, tmp_path: Path) -> None:
    services = build_services(ScriptedVLM(), tmp_path)
    audio = {"asr": [{"t_start": 1.0, "t_end": 2.0, "text": "hello"}], "tags": []}
    services.cache.put("s1", "audio", AUDIO_CACHE_KEY, {"audio": audio})
    run_batch([_sample(clip)], _config(use_audio=True), services)
    assert any('speech: "hello"' in r.text for r in services.vlm.requests)


def test_uniform_indices_include_last_frame() -> None:
    assert uniform_indices(100, 4) == [0, 33, 66, 99]


def test_range_midpoints_cover_the_range_and_respect_the_limit() -> None:
    frames = [{"frame_id": f"#{i:02d}", "t_sec": float(i)} for i in (1, 2, 3, 4)]
    assert range_midpoints(frames, "#02", "#04", 10) == [2.5, 3.5]
    assert len(range_midpoints(frames, "#01", "#04", 2)) == 2


def test_extract_json_handles_fences_and_garbage() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json("nothing") is None
    assert extract_json(None) is None


def _item(index: int, domain: str, video: str) -> Item:
    return Item(f"fo_{index:04d}", video, 1.0, "q", ("a", "b"), 0, domain, "Sound", "Causal")


def test_split_keeps_each_video_on_one_side_and_preserves_domain_share() -> None:
    items = [_item(i, "x" if i < 60 else "y", f"v{i // 2}") for i in range(100)]
    dev, test = stratified_split(items, 10, 40, seed=42)
    assert not {i.video for i in dev} & {i.video for i in test}
    assert sum(i.domain == "x" for i in dev) == 6


def test_option_shuffle_is_deterministic_per_sample() -> None:
    assert shuffled_order("a", 5, 42) == shuffled_order("a", 5, 42)
    assert sorted(shuffled_order("a", 5, 42)) == [0, 1, 2, 3, 4]


def test_shuffled_preamble_does_not_claim_temporal_order():
    from src.agent import prompting
    text = prompting.frame_preamble(["#01", "#02", "#03"], False, shuffled=True)
    assert "random order" in text and "temporal order" not in text
    text_vp = prompting.frame_preamble(["#01", "#02", "#03"], True, shuffled=True)
    assert "random order" in text_vp and "#01" in text_vp


def test_extra_dev_split_is_a_seeded_sample_outside_the_test_split(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    from scripts.build_dev_extra_splits import select

    ids = [f"fo_{i:04d}" for i in range(700)]
    held_out = set(ids[::2])
    chosen = select(ids, held_out)
    assert len(chosen) == 300 and not chosen & held_out and chosen == select(ids, held_out)
    with pytest.raises(ValueError):
        select(ids[:400], held_out)
    for name, held in (("full", "test"), ("fb_full", "fb_test")):
        rows = [{"sample_id": i, "video": f"{i}.mp4"} for i in ids]
        (tmp_path / "splits").mkdir(exist_ok=True)
        (tmp_path / "labels").mkdir(exist_ok=True)
        (tmp_path / f"splits/questions_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (tmp_path / f"labels/labels_{name}.jsonl").write_text("".join(json.dumps({**r, "correct": "A"}) + "\n" for r in rows))
        (tmp_path / f"splits/questions_{held}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[::2]))
    script = Path(__file__).parents[1] / "scripts/build_dev_extra_splits.py"
    subprocess.run([sys.executable, str(script), "--data-root", str(tmp_path)], check=True, timeout=120)
    questions = [json.loads(line) for line in (tmp_path / "splits/questions_fo_devx.jsonl").read_text().splitlines()]
    labels = [json.loads(line) for line in (tmp_path / "labels/labels_fb_devx.jsonl").read_text().splitlines()]
    assert {q["sample_id"] for q in questions} == chosen and [q["sample_id"] for q in questions] == sorted(chosen)  # full-set order
    assert len(labels) == 300 and "correct" not in questions[0] and labels[0]["correct"] == "A"


def test_audio_type_scope_filters_futureomni_only(monkeypatch) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import eval_common
    import subtype_table

    rows = {"fo_1": {"audio_type": "Speech"}, "fo_2": {"audio_type": "Music"}, "fo_3": {"audio_type": "Sound"}, "fb_1": {"audio_type": "n/a"}, "fb_2": {}}
    monkeypatch.setattr(eval_common, "label_rows", lambda split: dict(rows))
    try:
        assert set(subtype_table.label_rows("test")) == set(rows) and subtype_table.phase_title("test") == "test"
        subtype_table.keep_audio(["Speech"])
        assert set(subtype_table.label_rows("test")) == {"fo_1", "fb_1", "fb_2"}  # FutureBench items always stay
        assert subtype_table.phase_title("test") == "test, FutureOmni는 Speech 문항만" and subtype_table.phase_title("dev").startswith("개발 세트")
        subtype_table.keep_audio(["Music", "Speech"])
        assert set(subtype_table.label_rows("test")) == {"fo_1", "fo_2", "fb_1", "fb_2"}
        with pytest.raises(SystemExit):
            subtype_table.keep_audio(["Noise"])
    finally:
        subtype_table.keep_audio([])


def test_heldout_split_is_the_full_set_without_the_dev_side_and_test_items(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    ids = [f"x_{i:03d}" for i in range(30)]
    (tmp_path / "splits").mkdir()
    (tmp_path / "labels").mkdir()
    dump = lambda path, rows: path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for full, extra, held, _ in (("full", "fo_devx", "test", "fo_rest"), ("fb_full", "fb_devx", "fb_test", "fb_rest")):
        dump(tmp_path / f"splits/questions_{full}.jsonl", [{"sample_id": i} for i in ids])
        dump(tmp_path / f"labels/labels_{full}.jsonl", [{"sample_id": i, "correct": "A"} for i in ids])
        dump(tmp_path / f"splits/questions_{extra}.jsonl", [{"sample_id": i} for i in ids[:10]])
        dump(tmp_path / f"splits/questions_{held}.jsonl", [{"sample_id": i} for i in ids[10:18]])
    script = Path(__file__).parents[1] / "scripts/build_heldout_splits.py"
    subprocess.run([sys.executable, str(script), "--data-root", str(tmp_path)], check=True, timeout=120)
    for name in ("fo_rest", "fb_rest"):
        questions = [json.loads(line)["sample_id"] for line in (tmp_path / f"splits/questions_{name}.jsonl").read_text().splitlines()]
        labels = [json.loads(line)["sample_id"] for line in (tmp_path / f"labels/labels_{name}.jsonl").read_text().splitlines()]
        assert questions == ids[18:] == labels
