"""Burn a "frame #NN" visual prompt into a frame.

Source: ViKey [2], add_VP.py (red text on a semi-transparent white box, bottom-left as in the ViKey paper, font = short side / 10).
Changed from source: single-frame function; the label is the fixed frame id (for example "#02.1"), which later citations refer to.
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.agent.frame_ids import uniform_id, vp_label

# VP mechanism ablation: VP_STYLE=const draws the same text "frame" on every frame, so the label
# carries no index. Everything else (position, box, font) is unchanged. Default: numbered labels.
VP_STYLE = os.environ.get("VP_STYLE", "number")
CONST_LABEL = "frame"
AUTO_FONT_DIVISOR = 10
TEXT_COLOR = (255, 0, 0)
BOX_COLOR = (255, 255, 255, 200)
PADDING = 10


def vp_text(frame_id: int | str) -> str:
    """Label text; an int is treated as a uniform position for backward compatibility."""
    if VP_STYLE == "const":
        return CONST_LABEL
    return vp_label(uniform_id(frame_id) if isinstance(frame_id, int) else frame_id)


def render_number(source: str | Path, target: str | Path, frame_id: int | str) -> None:
    """Write a copy of `source` with the frame label drawn in the bottom-left corner."""
    with Image.open(source) as opened:
        image = opened.convert("RGBA")
    font = ImageFont.load_default(size=min(image.size) // AUTO_FONT_DIVISOR)
    layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)
    text = vp_text(frame_id)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    box_height = bottom - top + 2 * PADDING
    box_top = image.height - box_height
    draw.rectangle([0, box_top, right - left + 2 * PADDING, image.height], fill=BOX_COLOR)
    draw.text((PADDING - left, box_top + PADDING - top), text, font=font, fill=TEXT_COLOR)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, layer).convert("RGB").save(target, quality=92)
