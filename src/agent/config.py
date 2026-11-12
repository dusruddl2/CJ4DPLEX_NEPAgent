"""Load run configurations; a YAML may inherit another file (`inherit: base.yaml`) and override its values."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.schemas import RunConfig


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    parent = data.pop("inherit", None)
    return {**_read(path.parent / parent), **data} if parent else data


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> RunConfig:
    """Return the RunConfig for a YAML file; the run name defaults to the file stem."""
    source = Path(path)
    values = {"run_name": source.stem, "config_path": str(source), **_read(source), **(overrides or {})}
    known = {field.name for field in fields(RunConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown config keys in {source}: {sorted(unknown)}")
    return RunConfig(**values)
