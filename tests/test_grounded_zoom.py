"""Object-grounded zoom: cue to query text, crop box expansion, and the fallback path (no detector needed)."""
from src.agent import grounding


def test_query_text_is_short_lower_case_phrase_with_period():
    assert grounding.query_text("The Red Cup on the table.") == "the red cup on the table."
    assert grounding.query_text("   ") == ""
    assert len(grounding.query_text(" ".join(["word"] * 30)).split()) == grounding.MAX_CUE_WORDS


def test_expand_box_adds_margin_keeps_minimum_and_stays_inside():
    left, top, right, bottom = grounding.expand_box((100, 100, 120, 110), 400, 200)
    assert right - left >= 400 * grounding.MIN_SHARE and bottom - top >= 200 * grounding.MIN_SHARE
    assert 0 <= left < right <= 400 and 0 <= top < bottom <= 200
    left, top, right, bottom = grounding.expand_box((350, 150, 400, 200), 400, 200)
    assert right == 400 and bottom == 200


def test_zoom_falls_back_to_region_when_nothing_is_detected(tmp_path, monkeypatch):
    from PIL import Image
    from src.agent import tools
    from src.agent.observer import FrameIO
    from src.schemas import AgentInput, SampleState
    image = tmp_path / "f.jpg"; Image.new("RGB", (64, 48), "red").save(image)
    frames = ({"frame_id": "#01", "t_sec": 1.0, "source": "uniform", "path": str(image), "raw_path": str(image), "caption": ""},)
    state = SampleState(input=AgentInput("s", "p.mp4", "q", ()), frames=frames)
    monkeypatch.setattr(grounding, "detect", lambda img, cue: None)
    out = tools.apply_zoom(state, {"where": ["#01", "#01"], "region": "center", "missing": "the cup"}, FrameIO(tmp_path), "grounded")
    entry = out.extra_evidence[-1]
    assert entry["kind"] == tools.ZOOM and entry["grounded"] is False and entry["fallback"] == "no_detection"
    monkeypatch.setattr(grounding, "detect", lambda img, cue: {"box": [10.0, 10.0, 20.0, 20.0], "score": 0.9, "label": "cup"})
    out = tools.apply_zoom(state, {"where": ["#01", "#01"], "region": "center", "missing": "the saucer"}, FrameIO(tmp_path / "b"), "grounded")
    assert out.extra_evidence[-1]["grounded"] is True and out.extra_evidence[-1]["detector_label"] == "cup"
