"""Extract FutureOmni test clips and create data/splits/{sample_id}/prefix.mp4.

Source: FutureOmni [4] test clips, cut at `seconds`. Changed from source: the cut is verified
against the real clip length instead of being assumed, and clips that are too long are cut here.
The archive is untrusted: only flat `<digits>.mp4` members are ever written out.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.data.futureomni import read_jsonl
from src.data.splitter import split_video
from src.utils.media import has_audio_stream, probe_duration

logger = logging.getLogger(__name__)

PART_PATTERN = "test_splitted_videos_part_*"
DURATION_TOLERANCE_SEC = 1.0
MAX_MEMBER_BYTES = 8 * 1024**3
COPY_BUFFER_BYTES = 8 * 1024**2
_MEMBER_NAME = re.compile(r"^\d+\.mp4$")
_SAMPLE_ID = re.compile(r"^fo_(\d+)$")

STATUS_PREFIX = "already_prefix"
STATUS_CUT = "cut"
STATUS_SHORTER = "shorter_than_t_split"
STATUS_MISSING = "missing_source"
STATUS_ERROR = "probe_error"


class VideosNotReady(RuntimeError):
    """The test videos have not finished downloading."""


class ConcatFile(io.RawIOBase):
    """Read-only seekable view over several files, so a split zip needs no merged copy."""

    def __init__(self, paths: Sequence[Path]) -> None:
        super().__init__()
        self._handles = [path.open("rb") for path in paths]
        self._sizes = [path.stat().st_size for path in paths]
        self._size = sum(self._sizes)
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._size}[whence]
        self._position = max(0, base + offset)
        return self._position

    def readinto(self, buffer: Any) -> int:
        offset = self._position
        for handle, size in zip(self._handles, self._sizes):
            if offset >= size:
                offset -= size
                continue
            handle.seek(offset)
            count = handle.readinto(memoryview(buffer)[: size - offset])
            self._position += count or 0
            return count or 0
        return 0

    def close(self) -> None:
        for handle in self._handles:
            handle.close()
        super().close()


@dataclass(frozen=True)
class Question:
    sample_id: str
    t_split: float | None
    video: str = ""

    @property
    def member(self) -> str:
        match = _SAMPLE_ID.match(self.sample_id)
        if match is None:
            raise ValueError(f"unexpected FutureOmni sample id: {self.sample_id}")
        return f"{int(match.group(1))}.mp4"


def load_questions(paths: Iterable[str | Path]) -> list[Question]:
    """Unique questions from the agent-facing files, which carry no labels."""
    seen: dict[str, Question] = {}
    for path in paths:
        for row in read_jsonl(path):
            t_split = row.get("t_split")
            question = Question(row["sample_id"], None if t_split is None else float(t_split), row.get("video", ""))
            seen.setdefault(row["sample_id"], question)
    return list(seen.values())


def find_parts(archive_dir: str | Path) -> list[Path]:
    return sorted(Path(archive_dir).glob(PART_PATTERN))


def _open_archive(reader: io.BufferedReader, parts: Sequence[Path]) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(reader)
    except zipfile.BadZipFile as error:
        names = ", ".join(part.name for part in parts)
        raise VideosNotReady(f"archive parts are incomplete or invalid ({names}): {error}") from error


def _safe_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    return {
        info.filename: info
        for info in archive.infolist()
        if _MEMBER_NAME.match(info.filename) and info.file_size <= MAX_MEMBER_BYTES
    }


def _extract_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> None:
    partial = target.with_name(f"{target.name}.part")
    with archive.open(info) as source, partial.open("wb") as sink:
        shutil.copyfileobj(source, sink, COPY_BUFFER_BYTES)
    os.replace(partial, target)


def extract_videos(parts: Sequence[Path], members: Iterable[str], video_dir: Path) -> None:
    """Extract the requested clips, skipping any already present at full size."""
    video_dir.mkdir(parents=True, exist_ok=True)
    with io.BufferedReader(ConcatFile(parts), COPY_BUFFER_BYTES) as reader, _open_archive(reader, parts) as archive:
        available = _safe_members(archive)
        for name in sorted(set(members)):
            info = available.get(name)
            if info is None:
                continue
            target = video_dir / name
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            logger.info("extracting %s", name)
            _extract_member(archive, info, target)


def ensure_videos(archive_dir: Path, video_dir: Path, members: Sequence[str]) -> None:
    """Extract from the archive, or accept an existing extraction when the archive is gone."""
    parts = find_parts(archive_dir)
    if parts:
        extract_videos(parts, members, video_dir)
        return
    if not (video_dir.is_dir() and any(video_dir.glob("*.mp4"))):
        raise VideosNotReady(
            f"no '{PART_PATTERN}' files in {archive_dir} and no extracted clips in {video_dir}. "
            "The test video download has not finished; see logs/download_futureomni.log."
        )
    logger.warning("no archive parts found; using the clips already in %s", video_dir)


def _is_done(prefix: Path) -> bool:
    return prefix.exists() and prefix.stat().st_size > 0


def _link_prefix(source: Path, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    temporary = prefix.with_name(f".{prefix.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(source.resolve(), prefix.parent.resolve()))
    os.replace(temporary, prefix)


def _cut_prefix(source: Path, t_split: float, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    work = prefix.parent / ".cutting"
    written, _ = split_video(source, t_split, work, write_future=False)
    os.replace(written, prefix)
    work.rmdir()


def classify_clip(duration: float, t_split: float, tolerance: float) -> str:
    if abs(duration - t_split) <= tolerance:
        return STATUS_PREFIX
    return STATUS_CUT if duration > t_split else STATUS_SHORTER


def _process_question(
    question: Question, video_dir: Path, splits_root: Path, tolerance: float
) -> dict[str, Any]:
    source = video_dir / question.member
    record: dict[str, Any] = {"sample_id": question.sample_id, "t_split": question.t_split}
    if not source.exists():
        return {**record, "status": STATUS_MISSING}
    try:
        duration = probe_duration(source)
    except (OSError, ValueError) as error:
        return {**record, "status": STATUS_ERROR, "error": str(error)}

    if question.t_split is None:
        return {**record, "status": STATUS_ERROR, "error": "t_split is required for FutureOmni clips"}
    status = classify_clip(duration, question.t_split, tolerance)
    prefix = splits_root / question.sample_id / "prefix.mp4"
    resumed = _is_done(prefix)
    if not resumed:
        if status == STATUS_CUT:
            _cut_prefix(source, question.t_split, prefix)
        else:
            _link_prefix(source, prefix)
    return {**record, "status": status, "duration": duration, "resumed": resumed}


def build_report(records: Sequence[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    counts = {status: 0 for status in (STATUS_PREFIX, STATUS_CUT, STATUS_SHORTER, STATUS_MISSING, STATUS_ERROR)}
    for record in records:
        counts[record["status"]] += 1
    probed = [r for r in records if "duration" in r]
    differences = [abs(r["duration"] - r["t_split"]) for r in probed]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tolerance_sec": tolerance,
        "n_samples": len(records),
        "counts": counts,
        "resumed_existing": sum(1 for r in probed if r["resumed"]),
        "max_abs_duration_diff_sec": max(differences, default=0.0),
        "length_mismatches": [r for r in records if r["status"] in (STATUS_CUT, STATUS_SHORTER)],
        "failures": [r for r in records if r["status"] in (STATUS_MISSING, STATUS_ERROR)],
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_prefixes(
    question_files: Sequence[str | Path],
    archive_dir: str | Path,
    video_dir: str | Path,
    splits_root: str | Path,
    report_path: str | Path,
    tolerance: float = DURATION_TOLERANCE_SEC,
) -> dict[str, Any]:
    """Create every prefix.mp4 and write the length report. Safe to re-run after interruption."""
    questions = load_questions(question_files)
    videos, splits = Path(video_dir), Path(splits_root)
    ensure_videos(Path(archive_dir), videos, [q.member for q in questions])
    records = [_process_question(q, videos, splits, tolerance) for q in questions]
    report = build_report(records, tolerance)
    write_report(Path(report_path), report)
    return report


def _link_question(question: Question, video_root: Path, splits_root: Path) -> dict[str, Any]:
    """Link an already-prefix clip (FutureBench first segment) and record its duration and audio."""
    source = video_root / question.video
    record: dict[str, Any] = {"sample_id": question.sample_id, "video": question.video}
    if not source.exists():
        return {**record, "status": STATUS_MISSING}
    try:
        duration, audio = probe_duration(source), has_audio_stream(source)
    except (OSError, ValueError) as error:
        return {**record, "status": STATUS_ERROR, "error": str(error)}
    prefix = splits_root / question.sample_id / "prefix.mp4"
    resumed = _is_done(prefix)
    if not resumed:
        _link_prefix(source, prefix)
    return {**record, "status": STATUS_PREFIX, "duration": duration, "has_audio": audio, "resumed": resumed}


def build_link_report(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    linked = [r for r in records if r["status"] == STATUS_PREFIX]
    durations = [r["duration"] for r in linked]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(records),
        "linked": len(linked),
        "resumed_existing": sum(1 for r in linked if r["resumed"]),
        "with_audio": sum(1 for r in linked if r["has_audio"]),
        "without_audio": sum(1 for r in linked if not r["has_audio"]),
        "duration_sec": {
            "min": min(durations, default=0.0),
            "median": sorted(durations)[len(durations) // 2] if durations else 0.0,
            "max": max(durations, default=0.0),
        },
        "failures": [r for r in records if r["status"] in (STATUS_MISSING, STATUS_ERROR)],
    }


def prepare_linked_prefixes(
    question_files: Sequence[str | Path], video_root: str | Path, splits_root: str | Path, report_path: str | Path
) -> dict[str, Any]:
    """FutureBench: every clip is already a prefix, so link it and report length and audio."""
    root = Path(video_root)
    if not root.is_dir():
        raise VideosNotReady(f"video root {root} does not exist")
    questions = load_questions(question_files)
    records = [_link_question(q, root, Path(splits_root)) for q in questions]
    report = build_link_report(records)
    write_report(Path(report_path), report)
    return report
