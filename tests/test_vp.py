"""The frame-number visual prompt must sit in the bottom-left corner (ViKey paper setting)."""
from pathlib import Path

from PIL import Image

from src.agent.vp import render_number


def test_number_box_is_drawn_bottom_left(tmp_path: Path) -> None:
    source = tmp_path / "black.png"
    target = tmp_path / "numbered.png"
    Image.new("RGB", (320, 240), (0, 0, 0)).save(source)

    render_number(source, target, 7)

    with Image.open(target) as image:
        rgb = image.convert("RGB")
    assert min(rgb.getpixel((2, 237))) > 150
    assert rgb.getpixel((2, 2)) == (0, 0, 0)
    assert rgb.getpixel((317, 237)) == (0, 0, 0)


def test_const_style_draws_the_same_label_and_uses_positional_wording(monkeypatch):
    from src.agent import prompting, vp
    monkeypatch.setattr(vp, "VP_STYLE", "const")
    assert vp.vp_text("#07") == vp.vp_text("#31") == "frame"
    prompting.frame_preamble.cache_clear() if hasattr(prompting.frame_preamble, "cache_clear") else None
    text = prompting.frame_preamble(["#01", "#02"], True)
    assert "position" in text and "bottom-left" not in text
