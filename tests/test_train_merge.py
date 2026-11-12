"""Merge safeguards: configs must not drift, and adapters must stay in the language model."""
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.train.merge_adapter import check_adapter_scope, diff_configs

BASE = {
    "model_type": "qwen3_vl",
    "transformers_version": "4.57.0.dev0",
    "text_config": {"rope_theta": 5000000, "rope_scaling": {"mrope_section": [24, 20, 20]}, "dtype": "bfloat16"},
    "vision_config": {"depth": 27},
}


def _copy(**text_changes) -> dict:
    return {**BASE, "text_config": {**BASE["text_config"], **text_changes}}


def test_identical_configs_have_no_problems() -> None:
    assert diff_configs(BASE, BASE) == []


def test_version_metadata_is_ignored() -> None:
    assert diff_configs(BASE, {**BASE, "transformers_version": "4.57.1"}) == []


def test_changed_rope_theta_is_reported() -> None:
    problems = diff_configs(BASE, _copy(rope_theta=10000))
    assert problems == ["changed text_config.rope_theta: 5000000 -> 10000"]


def test_missing_rope_scaling_is_reported() -> None:
    merged = {**BASE, "text_config": {"rope_theta": 5000000, "dtype": "bfloat16"}}
    assert diff_configs(BASE, merged) == ["missing text_config.rope_scaling"]


def test_changed_nested_list_is_reported() -> None:
    problems = diff_configs(BASE, _copy(rope_scaling={"mrope_section": [24, 20, 21]}))
    assert problems and problems[0].startswith("changed text_config.rope_scaling.mrope_section")


def _adapter(path: Path, keys: list[str]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    save_file({key: torch.zeros(1) for key in keys}, str(path / "adapter_model.safetensors"))
    return path


def test_language_model_adapter_is_accepted(tmp_path: Path) -> None:
    keys = ["base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"]
    assert check_adapter_scope(_adapter(tmp_path, keys)) == 1


def test_vision_adapter_is_rejected(tmp_path: Path) -> None:
    keys = [
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.visual.blocks.0.attn.proj.lora_A.weight",
    ]
    with pytest.raises(ValueError):
        check_adapter_scope(_adapter(tmp_path, keys))


def test_adapter_outside_language_model_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        check_adapter_scope(_adapter(tmp_path, ["base_model.model.lm_head.lora_A.weight"]))


@pytest.mark.parametrize("script", ["scripts/train_stage1.sh", "scripts/download_data.sh", "src/train/run_sft.sh"])
def test_training_scripts_have_valid_shell_syntax(script: str) -> None:
    import subprocess

    subprocess.run(["bash", "-n", script], check=True, timeout=30)
