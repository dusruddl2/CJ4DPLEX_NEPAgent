"""The agent cannot reach labels or post-split video."""
import ast
import dataclasses
import subprocess
from pathlib import Path

import pytest

from src.schemas import AgentInput
from src.utils.media import ffmpeg_binary
from src.data.futureomni import Item, to_records
from src.data.splitter import probe_duration, split_video

SRC = Path(__file__).parents[1] / "src"
AGENT_DIRS = ("agent", "stages", "nep_agent")  # everything that runs at prediction time
FORBIDDEN_IMPORTS = ("src.eval", "src.data", "src.train")
FORBIDDEN_SUBSTRINGS = ("future", "ground_truth", "groundtruth", "gt_")
LABEL_WORDS = ("answer", "correct", "correct_index", "label_path")
# The only permitted use: the output schema key holding the model's own prediction.
ALLOWED_LABEL_WORDS = {("agent/output.py", "answer"), ("nep_agent/core.py", "answer")}
# Identifiers through which the answer options can be read, and the only files that may use them:
#   agent/prompting.py  defines question_block (question + options text)
#   stages/matcher.py   the text-only matcher, which maps a frozen sentence onto the options (the one legitimate reader)
#   stages/direct.py    only checks whether options were given, to skip the matcher when there are none
#   agent/output.py     copies the option texts of the ranked answers into the result
#   agent/vlm_ops.py    probs_charged, the call wrapper the matcher uses, passes the option letters to the backend
OPTION_ACCESSORS = ("options", "options_text", "letters", "question_block")
OPTION_READERS = {"agent/prompting.py", "stages/matcher.py", "stages/direct.py", "agent/output.py", "agent/vlm_ops.py"}
DURATION_TOLERANCE_SEC = 0.15


def _agent_files() -> list[Path]:
    return sorted(path for name in AGENT_DIRS for path in (SRC / name).glob("*.py"))


def _relative(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def _docstring_nodes(tree: ast.AST) -> set[int]:
    nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                nodes.add(id(first.value))
    return nodes


def _words(node: ast.AST, docstrings: set[int]) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
        return [node.value]
    return []


@pytest.mark.parametrize("path", _agent_files(), ids=_relative)
def test_agent_does_not_import_eval_or_data(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(FORBIDDEN_IMPORTS), f"{path.name}: {node.module}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(FORBIDDEN_IMPORTS), f"{path.name}: {alias.name}"


@pytest.mark.parametrize("path", _agent_files(), ids=_relative)
def test_agent_code_has_no_label_or_post_split_references(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        for word in _words(node, docstrings):
            lowered = word.lower()
            assert not any(s in lowered for s in FORBIDDEN_SUBSTRINGS), f"{path.name}: {word!r}"
            if lowered in LABEL_WORDS:
                assert (_relative(path), lowered) in ALLOWED_LABEL_WORDS, f"{path.name}: {word!r}"


@pytest.mark.parametrize("path", _agent_files(), ids=_relative)
def test_only_the_matcher_side_reads_the_options(path: Path) -> None:
    if _relative(path) in OPTION_READERS:
        return
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        assert name not in OPTION_ACCESSORS, f"{_relative(path)}: {name}"


def test_scan_covers_every_prediction_package() -> None:
    covered = {_relative(path).split("/")[0] for path in _agent_files()}
    assert covered == set(AGENT_DIRS) and len(_agent_files()) >= 20


def test_agent_input_has_no_label_field() -> None:
    names = {field.name for field in dataclasses.fields(AgentInput)}
    assert names == {"sample_id", "prefix_path", "question", "options"}


def test_question_record_contains_no_label() -> None:
    item = Item("fo_0001", "v.mp4", 10.0, "q?", ("a", "b", "c", "d", "e"), 2, "movie", "Sound", "Causal")
    question, label = to_records(item, seed=42)
    assert set(question) == {"sample_id", "video", "t_split", "question", "options"}
    assert dict(question["options"])[label["correct"]] == "c"


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("video") / "source.mp4"
    subprocess.run(
        [ffmpeg_binary(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=6:size=160x120:rate=10", str(path)],
        check=True, timeout=120,
    )
    return path


def test_prefix_does_not_exceed_split_point(synthetic_video: Path, tmp_path: Path) -> None:
    prefix, future = split_video(synthetic_video, 2.5, tmp_path)
    assert probe_duration(prefix) <= 2.5 + DURATION_TOLERANCE_SEC
    assert future is not None and probe_duration(future) > 3.0


def test_observed_clip_is_the_whole_file_without_a_split_point_inside_it(synthetic_video: Path, tmp_path: Path) -> None:
    from src.data.splitter import observed_clip

    for t_split in (None, 6.0, 60.0):  # no value, exactly the end, beyond the end
        assert observed_clip(synthetic_video, t_split, tmp_path / "whole") == synthetic_video
    assert not (tmp_path / "whole").exists()  # nothing is written
    prefix = observed_clip(synthetic_video, 2.5, tmp_path / "cut")
    assert prefix.name == "prefix.mp4" and probe_duration(prefix) <= 2.5 + DURATION_TOLERANCE_SEC
    assert not (tmp_path / "cut" / "future.mp4").exists()  # the part after the split point is never written here
    with pytest.raises(ValueError):
        observed_clip(synthetic_video, 0, tmp_path)


def test_split_rejects_point_after_end(synthetic_video: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        split_video(synthetic_video, 60.0, tmp_path)
