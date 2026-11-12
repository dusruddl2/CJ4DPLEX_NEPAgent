"""Object-grounded zoom: find the object named by the tool request's cue with Grounding DINO and return a crop box around it.

The detector is loaded once per process on first use. A box is returned only when the detector's best match passes the
thresholds; the caller then falls back to the fixed region crop.
"""
from __future__ import annotations

import os
from typing import Any

from PIL import Image

DETECTOR_ID = os.environ.get("ZOOM_DETECTOR", "IDEA-Research/grounding-dino-tiny")
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
MARGIN = 0.15  # each side of the detected box grows by this share of the box size
MIN_SHARE = 0.25  # the crop is at least this share of the frame in each dimension
MAX_CUE_WORDS = 12

_detector: tuple[Any, Any, str] | None = None


def _load() -> tuple[Any, Any, str]:
    global _detector
    if _detector is None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(DETECTOR_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_ID).to(device).eval()
        _detector = (processor, model, device)
    return _detector


def query_text(cue: str) -> str:
    """Grounding DINO expects a short lower-case phrase ending with a period."""
    words = cue.strip().lower().replace("\n", " ").split()[:MAX_CUE_WORDS]
    return (" ".join(words).rstrip(".") + ".") if words else ""


def expand_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """Grow the detected box by MARGIN, keep at least MIN_SHARE of the frame, and clamp it inside the image."""
    x0, y0, x1, y1 = box
    bw = max((x1 - x0) * (1 + 2 * MARGIN), width * MIN_SHARE)
    bh = max((y1 - y0) * (1 + 2 * MARGIN), height * MIN_SHARE)
    bw, bh = min(bw, width), min(bh, height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(cx - bw / 2, 0), width - bw)
    top = min(max(cy - bh / 2, 0), height - bh)
    return int(round(left)), int(round(top)), int(round(left + bw)), int(round(top + bh))


def detect(image: Image.Image, cue: str) -> dict[str, Any] | None:
    """Best box for the cue as {"box", "score", "label"} in pixel coordinates, or None."""
    text = query_text(cue)
    if not text:
        return None
    import torch

    processor, model, device = _load()
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=BOX_THRESHOLD,
                                                              text_threshold=TEXT_THRESHOLD, target_sizes=[image.size[::-1]])[0]
    if len(result["scores"]) == 0:
        return None
    best = int(result["scores"].argmax())
    labels = result.get("text_labels") or result.get("labels") or []
    return {"box": [float(v) for v in result["boxes"][best].tolist()], "score": float(result["scores"][best]),
            "label": str(labels[best]) if len(labels) > best else ""}
