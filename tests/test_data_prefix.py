"""prepare_prefix on synthetic clips and a synthetic split zip archive."""
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from src.data.prefix_prep import (
    ConcatFile,
    VideosNotReady,
    classify_clip,
    prepare_prefixes,
)
from src.utils.media import ffmpeg_binary, probe_duration

TOLERANCE = 0.3


def _make_clip(path: Path, seconds: float) -> None:
    subprocess.run(
        [ffmpeg_binary(), "-y", "-v", "error", "-f", "lavfi", "-i",
         f"testsrc=duration={seconds}:size=64x48:rate=5", "-pix_fmt", "yuv420p", str(path)],
        check=True, timeout=120,
    )


def _write_split_zip(archive_dir: Path, members: dict[str, Path], extra: dict[str, bytes]) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, source in members.items():
            archive.write(source, name)
        for name, payload in extra.items():
            archive.writestr(name, payload)
    data = buffer.getvalue()
    archive_dir.mkdir(parents=True, exist_ok=True)
    cut = len(data) // 2
    (archive_dir / "test_splitted_videos_part_aa").write_bytes(data[:cut])
    (archive_dir / "test_splitted_videos_part_ab").write_bytes(data[cut:])


@pytest.fixture()
def workspace(tmp_path: Path) -> dict[str, Path]:
    clips = tmp_path / "clips"
    clips.mkdir()
    for name, seconds in (("0.mp4", 3.0), ("1.mp4", 6.0), ("2.mp4", 2.0)):
        _make_clip(clips / name, seconds)
    questions = tmp_path / "questions.jsonl"
    rows = [("fo_0000", 3.0), ("fo_0001", 2.5), ("fo_0002", 5.0), ("fo_0003", 4.0)]
    questions.write_text(
        "".join(json.dumps({"sample_id": s, "t_split": t, "question": "q", "options": []}) + "\n" for s, t in rows),
        encoding="utf-8",
    )
    return {
        "clips": clips, "questions": questions, "archive": tmp_path / "archive",
        "videos": tmp_path / "videos", "splits": tmp_path / "splits", "report": tmp_path / "logs/report.json",
    }


def _run(ws: dict[str, Path]) -> dict:
    return prepare_prefixes([ws["questions"]], ws["archive"], ws["videos"], ws["splits"], ws["report"], tolerance=TOLERANCE)


def _prefix(ws: dict[str, Path], sample_id: str) -> Path:
    return ws["splits"] / sample_id / "prefix.mp4"


def _populate(ws: dict[str, Path], extra: dict[str, bytes] | None = None) -> None:
    _write_split_zip(ws["archive"], {p.name: p for p in ws["clips"].glob("*.mp4")}, extra or {})


def test_concat_file_reads_across_boundaries(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"hello ")
    (tmp_path / "b").write_bytes(b"world")
    with io.BufferedReader(ConcatFile([tmp_path / "a", tmp_path / "b"])) as reader:
        reader.seek(3)
        assert reader.read() == b"lo world"
        reader.seek(-2, io.SEEK_END)
        assert reader.read() == b"ld"


@pytest.mark.parametrize(
    "duration,t_split,expected",
    [(10.0, 10.0, "already_prefix"), (10.2, 10.0, "already_prefix"), (15.0, 10.0, "cut"), (4.0, 10.0, "shorter_than_t_split")],
)
def test_classify_clip(duration: float, t_split: float, expected: str) -> None:
    assert classify_clip(duration, t_split, TOLERANCE) == expected


def test_prepares_links_cuts_and_reports(workspace: dict[str, Path]) -> None:
    _populate(workspace)
    report = _run(workspace)

    assert report["counts"] == {
        "already_prefix": 1, "cut": 1, "shorter_than_t_split": 1, "missing_source": 1, "probe_error": 0,
    }
    assert _prefix(workspace, "fo_0000").is_symlink()
    assert not _prefix(workspace, "fo_0001").is_symlink()
    assert probe_duration(_prefix(workspace, "fo_0001")) <= 2.5 + TOLERANCE
    assert _prefix(workspace, "fo_0002").is_symlink()
    assert not _prefix(workspace, "fo_0003").exists()
    assert {r["sample_id"] for r in report["failures"]} == {"fo_0003"}
    assert {r["sample_id"] for r in report["length_mismatches"]} == {"fo_0001", "fo_0002"}
    assert json.loads(workspace["report"].read_text(encoding="utf-8"))["n_samples"] == 4


def test_rerun_resumes_without_redoing_work(workspace: dict[str, Path]) -> None:
    _populate(workspace)
    _run(workspace)
    cut_prefix = _prefix(workspace, "fo_0001")
    before = cut_prefix.stat().st_mtime_ns

    report = _run(workspace)

    assert cut_prefix.stat().st_mtime_ns == before
    assert report["resumed_existing"] == 3
    assert report["counts"]["cut"] == 1


def test_interrupted_prefix_is_rebuilt(workspace: dict[str, Path]) -> None:
    _populate(workspace)
    _run(workspace)
    _prefix(workspace, "fo_0000").unlink()

    report = _run(workspace)

    assert _prefix(workspace, "fo_0000").is_symlink()
    assert report["resumed_existing"] == 2


def test_only_flat_numeric_mp4_members_are_extracted(workspace: dict[str, Path]) -> None:
    _populate(workspace, {"../evil.mp4": b"x", "nested/5.mp4": b"x", "conftest.py": b"import os"})
    _run(workspace)
    assert sorted(p.name for p in workspace["videos"].iterdir()) == ["0.mp4", "1.mp4", "2.mp4"]
    assert not (workspace["videos"].parent / "evil.mp4").exists()


def test_no_videos_raises_clear_error(workspace: dict[str, Path]) -> None:
    with pytest.raises(VideosNotReady, match="not finished"):
        _run(workspace)


def test_truncated_archive_raises_clear_error(workspace: dict[str, Path]) -> None:
    workspace["archive"].mkdir()
    (workspace["archive"] / "test_splitted_videos_part_aa").write_bytes(b"PK\x03\x04partial")
    with pytest.raises(VideosNotReady, match="incomplete"):
        _run(workspace)


def test_uses_existing_extraction_when_archive_is_absent(workspace: dict[str, Path]) -> None:
    workspace["videos"].mkdir()
    _make_clip(workspace["videos"] / "0.mp4", 3.0)
    report = _run(workspace)
    assert report["counts"]["already_prefix"] == 1


def test_cli_exits_with_message_when_videos_are_missing(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts/prepare_prefix.py"
    questions = tmp_path / "q.jsonl"
    questions.write_text(json.dumps({"sample_id": "fo_0000", "t_split": 1.0}) + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-I", str(script), "--questions", str(questions), "--archive-dir", str(tmp_path / "a"),
         "--video-dir", str(tmp_path / "v"), "--splits-root", str(tmp_path / "s"), "--report", str(tmp_path / "r.json")],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "not ready" in result.stderr


def test_find_parts_follows_symlinked_archive_dir(tmp_path: Path) -> None:
    from src.data.prefix_prep import find_parts

    real = tmp_path / "real"
    real.mkdir()
    (real / "test_splitted_videos_part_aa").write_bytes(b"x")
    link = tmp_path / "link"
    link.symlink_to(real)
    assert [p.name for p in find_parts(link)] == ["test_splitted_videos_part_aa"]


def test_cli_default_archive_dir_is_the_repository_symlink() -> None:
    script = (Path(__file__).parents[1] / "scripts/prepare_prefix.py").read_text(encoding="utf-8")
    assert 'default=str(ROOT / "data/raw/futureomni")' in script
