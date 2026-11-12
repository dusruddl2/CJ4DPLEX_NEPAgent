"""Frame ids are fixed once assigned, and id order equals time order."""
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from src.agent import frame_ids
from src.agent.observer import FrameIO, add_frames, sample_frames
from src.schemas import AgentInput, RunConfig
from src.utils.media import ffmpeg_binary


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("clip") / "prefix.mp4"
    subprocess.run(
        [ffmpeg_binary(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=8:size=320x240:rate=10", str(path)],
        check=True, timeout=120,
    )
    return str(path)


def _state(clip: str, tmp_path: Path):
    cfg = replace(RunConfig(run_name="t", num_frames=8, use_audio=False))
    io = FrameIO(tmp_path)
    return sample_frames(AgentInput("s1", clip, "q", (("A", "a"),)), cfg, io), io


def test_sort_key_orders_sub_numbers_between_their_neighbours() -> None:
    ids = ["#03", "#02.2", "#02", "#02.1", "#16.1", "#07z", "#07"]
    assert sorted(ids, key=frame_ids.sort_key) == ["#02", "#02.1", "#02.2", "#03", "#07", "#07z", "#16.1"]


def test_sub_ids_follow_the_preceding_frame() -> None:
    frames = [{"frame_id": "#01", "t_sec": 0.0}, {"frame_id": "#02", "t_sec": 1.0}, {"frame_id": "#02.1", "t_sec": 1.5}]
    assert frame_ids.next_sub_ids(frames, [1.8, 2.5]) == [("#02.2", 1.8), ("#02.3", 2.5)]


def test_insertion_before_an_existing_sub_frame_gets_a_deeper_id_and_keeps_order() -> None:
    frames = [{"frame_id": "#02", "t_sec": 1.0}, {"frame_id": "#02.1", "t_sec": 1.5}, {"frame_id": "#03", "t_sec": 2.0}]
    assigned = frame_ids.next_sub_ids(frames, [1.2, 1.7])
    assert assigned == [("#02.0.1", 1.2), ("#02.2", 1.7)]
    grown = frames + [{"frame_id": i, "t_sec": t} for i, t in assigned]
    assert frame_ids.check_order(grown)


def test_resampling_keeps_existing_ids_and_times(clip: str, tmp_path: Path) -> None:
    state, io = _state(clip, tmp_path)
    before = {f["frame_id"]: f["t_sec"] for f in state.frames}
    grown, new_ids = add_frames(state, [1.05, 1.45, 7.95], frame_ids.SOURCE_DENSE, True, io)
    after = {f["frame_id"]: f["t_sec"] for f in grown.frames}
    assert all(after[k] == v for k, v in before.items())
    assert new_ids and all("." in i for i in new_ids)
    assert frame_ids.check_order(grown.frames)
    assert [f["t_sec"] for f in grown.frames] == sorted(f["t_sec"] for f in grown.frames)
    assert all(f["source"] == frame_ids.SOURCE_DENSE for f in grown.frames if f["frame_id"] in new_ids)


def test_labelled_image_path_encodes_the_id(clip: str, tmp_path: Path) -> None:
    state, io = _state(clip, tmp_path)
    grown, new_ids = add_frames(state, [1.05], frame_ids.SOURCE_DENSE, True, io)
    path = next(f["path"] for f in grown.frames if f["frame_id"] == new_ids[0])
    assert Path(path).name.startswith(frame_ids.file_stem(new_ids[0]) + "_")


def test_positional_citations_map_to_time_ordered_ids_without_labels():
    from src.agent.frame_ids import valid_ids

    known = ["#01", "#02", "#02.1", "#03"]
    assert valid_ids([3, "4", "#01", "#09", 0], known) == (["#01", "#02.1", "#03"], 2)


def test_no_vp_preamble_does_not_claim_labels():
    from src.agent.prompting import frame_preamble

    text = frame_preamble(["#01", "#02", "#03"], use_vp=False)
    assert "label" not in text and "position" in text and "3 for the last" in text
    assert "bottom-left" in frame_preamble(["#01", "#02"], use_vp=True)
