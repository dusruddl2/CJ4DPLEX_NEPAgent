"""Shared helpers that write LLaMA-Factory sharegpt datasets with inference-identical images.

Source: LlamaFactory [20] multimodal dataset format.
Changed from source: images are pre-resized with the inference loader, so training pixels and
token grids match what the served model receives.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from src.models.vlm import VLLMQwen

IMAGE_TOKEN = "<image>"
RESERVED_TOKENS = (IMAGE_TOKEN, "<video>", "<audio>")
FIT_JPEG_QUALITY = 95


def fit_image(source: str | Path, target: Path) -> str:
    """Save the frame exactly as VLLMQwen would feed it to the model; skip if already there."""
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.partial")
        VLLMQwen._load_image(str(source)).save(partial, format="JPEG", quality=FIT_JPEG_QUALITY)
        partial.replace(target)
    return str(target.resolve())


def fit_images(sources: Sequence[str], folder: Path) -> list[str]:
    return [fit_image(source, folder / f"{number:03d}.jpg") for number, source in enumerate(sources, 1)]


def has_reserved_token(text: str) -> bool:
    """LLaMA-Factory counts these placeholders, so literal ones in question text would break it."""
    return any(token in text for token in RESERVED_TOKENS)


def sft_record(prompt: str, target: str, images: Sequence[str], system: str = "") -> dict[str, Any]:
    """One conversation; image placeholders first, as in the inference chat template. `system` adds a leading system turn."""
    user = IMAGE_TOKEN * len(images) + prompt
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}, {"role": "assistant", "content": target}]
    return {"messages": messages, "images": list(images)}


def write_dataset(out_dir: Path, name: str, records: Sequence[dict[str, Any]]) -> None:
    """Write train.json and dataset_info.json under out_dir for `dataset: name`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.json").write_text(json.dumps(list(records), ensure_ascii=False), encoding="utf-8")
    info = {
        name: {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def write_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
