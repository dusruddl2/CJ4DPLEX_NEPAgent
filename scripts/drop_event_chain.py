"""Chain-free copy of an NEPAgent SFT dataset. Same items, same order; the input prompt asks for the `causal_sft_nc` schema and the
target drops `observed.events` (everything else unchanged). Usage: PYTHONPATH=. python scripts/drop_event_chain.py <src dir> <out dir>"""
import json
import shutil
import sys
from pathlib import Path

from src.nep_agent import event_reasoning as er

src, out = map(Path, sys.argv[1:3])
rows = json.loads((src / "train.json").read_text(encoding="utf-8"))
done = []
for r in rows:
    user, target = r["messages"][0]["content"], json.loads(r["messages"][-1]["content"])
    if er.TASKS["causal_sft"] not in user or er.SCHEMAS["causal_sft"] not in user:
        raise SystemExit("input prompt does not contain the causal_sft task/schema text")
    user = user.replace(er.TASKS["causal_sft"], er.TASKS["causal_sft_nc"]).replace(er.SCHEMAS["causal_sft"], er.SCHEMAS["causal_sft_nc"])
    target["observed"] = {k: v for k, v in target["observed"].items() if k != "events"}
    done.append({**r, "messages": [{**r["messages"][0], "content": user}, *r["messages"][1:-1],
                                   {**r["messages"][-1], "content": json.dumps(target, ensure_ascii=False)}]})
out.mkdir(parents=True, exist_ok=True)
(out / "train.json").write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
shutil.copy(src / "dataset_info.json", out / "dataset_info.json")
(out / "manifest.json").write_text(json.dumps({"source": str(src), "items": len(done),
                                               "change": "prompt = causal_sft_nc; target without observed.events"}, indent=1), encoding="utf-8")
print(len(done), "items ->", out)
