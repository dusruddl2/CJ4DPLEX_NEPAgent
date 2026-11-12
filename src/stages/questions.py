"""Option-free question text for the open-ended stages.

The benchmark questions are written for a multiple-choice setting. Two modes are supported:
  generic      - a fixed next-event question; the benchmark question is ignored.
  conditioned  - the benchmark question with option-dependent wording rewritten by rules (no model call, no labels).
Every rewrite is recorded so the limits of the transformation can be reported.
"""
from __future__ import annotations

import re
from typing import Any

GENERIC_QUESTION = "What happens next, right after the clip ends?"

# (name, pattern, replacement) applied in order; patterns are case-insensitive.
_RULES: tuple[tuple[str, str, str], ...] = (
    ("which_event_conclusion", r"which event is its most direct conclusion\??",
     "what event is its most direct conclusion? Describe that event in one sentence."),
    ("which_of_the_following", r"which (?:one )?of (?:the )?(?:following|these|the options|the choices)\b", "what"),
    ("which_option", r"which (?:option|choice|event|action|outcome)\b", "what event"),
    ("from_the_options", r"\s*(?:from|among) the (?:following |given )?(?:options|choices|candidates)(?: below)?", ""),
    ("select_the", r"\b(?:select|choose|pick) the (?:correct |best |most likely )?(?:option|answer|choice)\b", "describe the event"),
)
_SLOT = re.compile(r"\[\?\]")


def normalize_question(question: str, mode: str) -> dict[str, Any]:
    """Return {"text", "mode", "rules", "slots"}; `rules` lists the rewrites that fired."""
    if mode == "generic":
        return {"text": GENERIC_QUESTION, "observer_text": GENERIC_QUESTION, "mode": mode, "rules": [], "slots": 0}
    if mode != "conditioned":
        raise ValueError(f"unknown question_mode: {mode}")
    text = question.strip()
    fired: list[str] = []
    for name, pattern, replacement in _RULES:
        updated = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if updated != text:
            fired.append(name)
            text = updated
    slots = len(_SLOT.findall(text))
    if slots:
        fired.append("slot_instruction")
        text += " Describe the event for the first [?] slot in one sentence."
    if "option" in text.lower() or "following" in text.lower() and not slots:
        fired.append("residual_option_wording")  # reported as a limit; the text is still used as is
    return {"text": text, "observer_text": observer_question(text), "mode": mode, "rules": fired, "slots": slots}


_SEQUENCE = re.compile(r"\s*1\.\s*\[\?\].*$", re.DOTALL)


def observer_question(text: str) -> str:
    """The question without its "1. [?] 2. ..." slot sequence; the later slots describe events after the clip."""
    if "[?]" not in text:
        return text
    stripped = _SEQUENCE.sub("", text).strip()
    return stripped or GENERIC_QUESTION
