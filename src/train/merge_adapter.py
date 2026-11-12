#!/usr/bin/env python
"""Merge a trained LoRA adapter into the base model so vLLM can serve it as a plain model.

The saved config.json is compared with the base, because a transformers version
mismatch can silently change values such as rope_theta.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger("merge_adapter")

REQUIRED_TRANSFORMERS = "4.57.1"
TEMP_SUFFIX = ".merge_tmp"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
LANGUAGE_MODEL_KEY = "language_model"
VISION_MODEL_KEY = "visual"
IGNORED_CONFIG_KEYS = frozenset({"transformers_version", "torch_dtype", "dtype", "_name_or_path"})
NOT_COPIED_SUFFIXES = (".safetensors", ".bin", ".pt")
NOT_COPIED_NAMES = frozenset({"config.json", "model.safetensors.index.json", "README.md", "nep_agent_lora.md"})


def diff_configs(base: Mapping[str, Any], merged: Mapping[str, Any], prefix: str = "") -> list[str]:
    """List fields that the merged config lost or changed; version metadata is ignored."""
    problems: list[str] = []
    for key, value in base.items():
        if key in IGNORED_CONFIG_KEYS:
            continue
        name = f"{prefix}{key}"
        if key not in merged:
            problems.append(f"missing {name}")
        elif isinstance(value, dict) and isinstance(merged[key], dict):
            problems.extend(diff_configs(value, merged[key], f"{name}."))
        elif merged[key] != value:
            problems.append(f"changed {name}: {value!r} -> {merged[key]!r}")
    return problems


def check_adapter_scope(adapter_dir: Path) -> int:
    """Require every LoRA weight to sit in the language model; return the weight count."""
    from safetensors import safe_open

    with safe_open(str(adapter_dir / ADAPTER_WEIGHTS), framework="pt") as handle:
        keys = list(handle.keys())
    if not keys:
        raise ValueError(f"{adapter_dir} has no adapter weights")
    outside = [key for key in keys if LANGUAGE_MODEL_KEY not in key or f".{VISION_MODEL_KEY}." in key]
    if outside:
        raise ValueError(f"adapter touches modules outside the language model, e.g. {outside[:3]}")
    return len(keys)


def _copy_processor_files(base: Path, target: Path) -> None:
    """Copy tokenizer and processor files byte for byte so inference preprocessing is unchanged."""
    for path in base.iterdir():
        if path.is_file() and path.name not in NOT_COPIED_NAMES and not path.name.endswith(NOT_COPIED_SUFFIXES):
            shutil.copy2(path, target / path.name)


def _read_config(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "config.json").read_text(encoding="utf-8"))


def merge(base: Path, adapter: Path, out: Path, device: str, force: bool) -> None:
    """Write the merged model atomically to `out` after validating the adapter and the config."""
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if transformers.__version__ != REQUIRED_TRANSFORMERS:
        raise RuntimeError(f"merge with transformers {REQUIRED_TRANSFORMERS}, found {transformers.__version__}")
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; pass --force to replace it")
    logger.info("adapter has %d weights, all in the language model", check_adapter_scope(adapter))

    model = Qwen3VLForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, device_map={"": device})
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()

    temporary = out.with_name(out.name + TEMP_SUFFIX)
    if temporary.exists():
        shutil.rmtree(temporary)
    merged.save_pretrained(temporary, safe_serialization=True)
    _copy_processor_files(base, temporary)

    problems = diff_configs(_read_config(base), _read_config(temporary))
    if problems:
        raise ValueError(f"merged config differs from base: {problems}")
    AutoProcessor.from_pretrained(temporary)

    if out.exists():
        shutil.rmtree(out)
    temporary.rename(out)
    logger.info("merged model saved to %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True, help="use {lora_path}_merged so resolve_model_path finds it")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    merge(Path(args.base), Path(args.adapter), Path(args.out), args.device, args.force)


if __name__ == "__main__":
    main()
