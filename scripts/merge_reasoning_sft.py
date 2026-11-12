"""NEPAgent SFT training set at the size of the answer-only SFT set: the kept base items (same order) + the first kept extra items
(pool order, no selection by content) until the target size (default 1,500).
Usage: python3 scripts/merge_reasoning_sft.py <base dir> <extra dir> <out dir> [target]"""
import json
import shutil
import sys
from pathlib import Path

base, extra, out = map(Path, sys.argv[1:4])
target = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
key = lambda r: Path(r["images"][0]).parts[-3]
rows = json.loads((base / "train.json").read_text(encoding="utf-8"))
seen = {key(r) for r in rows}
add = [r for r in json.loads((extra / "train.json").read_text(encoding="utf-8")) if key(r) not in seen][: target - len(rows)]
if len(rows) + len(add) < target:
    raise SystemExit(f"only {len(rows) + len(add)} items, need {target}")
out.mkdir(parents=True, exist_ok=True)
(out / "train.json").write_text(json.dumps(rows + add, ensure_ascii=False), encoding="utf-8")
shutil.copy(base / "dataset_info.json", out / "dataset_info.json")
(out / "manifest.json").write_text(json.dumps({"base": str(base), "base_items": len(rows), "extra": str(extra),
    "extra_items_used": len(add), "total": len(rows) + len(add), "rule": "extras taken in pool order after the same Teacher filter"}, indent=1), encoding="utf-8")
print(len(rows), "+", len(add), "=", len(rows) + len(add))
