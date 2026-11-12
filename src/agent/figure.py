"""Timeline figure of one result: sampled frames, cited frames, what the tool added, and the prediction with its reasoning."""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

STRIP_FRAMES = 12  # sampled frames shown, evenly spaced
EXTRA_FRAMES = 6  # frames a tool added, or cited frames missing from the strip
WRAP = 150
CITED_COLOR, ADDED_COLOR = "#d62728", "#1f77b4"


def _even(items: list[Any], count: int) -> list[Any]:
    if len(items) <= count:
        return items
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


def _show(axis, frame: dict[str, Any], title: str, color: str | None) -> None:
    from PIL import Image

    path = Path(frame.get("path") or "")
    if path.is_file():
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
    else:  # the result can be read on a machine without the frame cache
        axis.set_facecolor("#dddddd")
        axis.text(0.5, 0.5, "no image", ha="center", va="center", fontsize=5, color="#666666", transform=axis.transAxes)
    axis.set_title(title, fontsize=7, color=color or "black", pad=2)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_edgecolor(color or "#888888")
        spine.set_linewidth(2.5 if color else 0.5)


def _line(label: str, value: Any) -> list[str]:
    text = "none" if value in (None, "", []) else str(value)
    return textwrap.wrap(f"{label}: {text}", WRAP, subsequent_indent="    ") or [f"{label}:"]


def describe(result: dict[str, Any]) -> list[str]:
    """Text of the figure: Direct sentence and confidence, the tool, the reasoning record, the reported prediction."""
    prediction, trace = result["prediction"], result["trace"]
    direct, agent = trace.get("direct") or {}, trace.get("nep_agent") or {}
    routing = direct.get("routing") or {}
    source = agent.get("final_source", "direct")
    entry = (agent.get("reasoning") or {}).get(source) or {}
    record = entry.get("record") or {}
    lines = _line("Direct prediction", (direct.get("direct_prediction") or {}).get("event"))
    confidence, tau = routing.get("generation_confidence"), routing.get("gate_tau")
    if confidence is not None and tau is not None:
        verdict = "below the threshold: tool and reasoning stages run" if routing.get("routed") else "at or above the threshold: Direct is kept"
        lines += [f"Direct confidence: {confidence:.2f} (threshold {tau:g}; {verdict})"]
    selection = entry.get("selection") or direct.get("tool_selection")
    observation = entry.get("observation") or direct.get("tool_observation")
    if selection and selection.get("tool"):
        added = ", ".join((observation or {}).get("new_evidence_ids") or []) or "nothing"
        where = " to ".join(selection.get("where") or [])
        lines += _line("Tool", f"{selection['tool']} on frames {where} ({(observation or {}).get('status', 'not run')}; added {added})")
        lines += _line("Looked for", selection.get("cue"))
        if (observation or {}).get("evidence_text"):
            lines += _line("Audio in that window", observation["evidence_text"])
    if record:
        lines += _line("Terminal state", record.get("terminal_state"))
        lines += _line("Ongoing action", record.get("ongoing_action"))
        lines += _line("Causal transition", record.get("causal_transition"))
        lines += _line("Supporting evidence", ", ".join(record.get("supporting_evidence") or []))
    lines += _line("Predicted next event", prediction["next_event"])
    lines += [f"Reported candidate: {source}"]
    probs = prediction.get("option_probs") or {}
    if probs:
        lines += ["Options (matcher probability):"]
        for item in prediction["top_k"]:
            lines += textwrap.wrap(f"  {item['option']} {probs[item['option']]:.2f}  {item['event']}", WRAP, subsequent_indent="          ")
    return lines


def draw_timeline(result: dict[str, Any], output_path: str | Path) -> None:
    """Save one PNG: a strip of the sampled frames, the frames the tool added or the reasoning cited, and the text."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not result["prediction"]:
        raise ValueError(f"{result['sample_id']} has no prediction to draw")
    agent = result["trace"].get("nep_agent") or {}
    entry = (agent.get("reasoning") or {}).get(agent.get("final_source", "direct")) or {}
    cited = set((entry.get("record") or {}).get("supporting_evidence") or [])
    added = set((entry.get("observation") or {}).get("new_evidence_ids") or [])
    sampled = [f for f in result["frames"] if f["source"] == "uniform"]
    strip = _even(sampled, STRIP_FRAMES)
    shown = {f["frame_id"] for f in strip}
    extra = [f for f in result["frames"] if f["frame_id"] in added or (f["frame_id"] in cited and f["frame_id"] not in shown)][:EXTRA_FRAMES]
    lines = describe(result)

    rows = [1.25] + ([1.9] if extra else []) + [0.19 * len(lines) + 0.3]
    figure = plt.figure(figsize=(11, sum(rows) + 0.5))
    grid = figure.add_gridspec(len(rows), 1, height_ratios=rows, hspace=0.25)
    top = grid[0].subgridspec(1, max(len(strip), 1), wspace=0.04)
    for slot, frame in enumerate(strip):
        is_cited = frame["frame_id"] in cited
        _show(figure.add_subplot(top[0, slot]), frame, frame["frame_id"] + (" cited" if is_cited else ""), CITED_COLOR if is_cited else None)
    if extra:
        middle = grid[1].subgridspec(1, EXTRA_FRAMES, wspace=0.06)
        for slot, frame in enumerate(extra):
            new = frame["frame_id"] in added
            label = frame["frame_id"] + (" added by the tool" if new else "") + (" cited" if frame["frame_id"] in cited else "")
            _show(figure.add_subplot(middle[0, slot]), frame, label, CITED_COLOR if frame["frame_id"] in cited else ADDED_COLOR)
    text_axis = figure.add_subplot(grid[-1])
    text_axis.axis("off")
    text_axis.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=8, linespacing=1.35, transform=text_axis.transAxes)
    figure.suptitle(f"{result['sample_id']}: sampled frames {sampled[0]['frame_id']} to {sampled[-1]['frame_id']} ({len(sampled)} frames)" if sampled else result["sample_id"],
                    fontsize=9, y=0.995)
    figure.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(figure)
