"""Per-sample JSON cache keyed by (sample_id, module, module_config_hash)."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


def config_hash(module_config: Mapping[str, Any]) -> str:
    """Return a short, order-independent hash of the settings a module depends on."""
    canonical = json.dumps(module_config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]


class JsonCache:
    """Store one JSON document per (sample, module, config hash)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, sample_id: str, module: str, module_config: Mapping[str, Any]) -> Path:
        return self.root / sample_id / f"{module}_{config_hash(module_config)}.json"

    def get(self, sample_id: str, module: str, module_config: Mapping[str, Any]) -> Any | None:
        path = self.path(sample_id, module, module_config)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def put(self, sample_id: str, module: str, module_config: Mapping[str, Any], value: Any) -> None:
        """Write atomically so that an interrupted run never leaves a truncated file."""
        path = self.path(sample_id, module, module_config)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=1)
        os.replace(tmp_name, path)
