"""Value objects shared by the agent modules. Inputs carry only observed content."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STATUS_OK = "ok"
STATUS_PARSE_ERROR = "parse_error"
STATUS_VIDEO_ERROR = "video_error"


@dataclass(frozen=True)
class AgentInput:
    """Everything the agent may see for one sample."""

    sample_id: str
    prefix_path: str
    question: str
    options: tuple[tuple[str, str], ...]

    @property
    def letters(self) -> tuple[str, ...]:
        return tuple(letter for letter, _ in self.options)

    def options_text(self) -> str:
        return "\n".join(f"{letter}. {text}" for letter, text in self.options)


@dataclass(frozen=True)
class RunConfig:
    """One run configuration; see configs/*.yaml."""

    run_name: str
    method: str = "nep_agent"
    num_frames: int = 16
    use_vp: bool = True  # burn frame-number labels into the frames
    use_reflector: bool = True  # route an item to the tool and reasoning stages when the Direct sentence confidence is below gate_tau
    use_audio: bool = True  # add the audio transcript text to the prompts
    gate_tau: float = 0.5  # routing threshold on exp(mean token log-prob) of the Direct sentence
    uncertain_tau: float = 0.5  # the result is flagged uncertain when the matcher confidence is below this
    lora_path: str = ""  # adapter served as a merged model (<lora_path>_merged); empty = base model
    input_mode: str = "images"
    frame_order: str = "temporal"  # "shuffle": present the sampled frames in a seeded random order (temporal control)
    frame_select: str = "uniform"  # "last": take the last num_frames decoded frames instead of uniform sampling
    question_mode: str = "conditioned"  # "conditioned" (option-free rewrite of the benchmark question) | "generic" (fixed next-event question)
    reasoning_variants: str = "coe,causal,coe_causal,tool"  # reasoning candidates computed after Direct (comma list, may be empty)
    store_all_candidates: bool = False  # true: compute every listed candidate for every item (ablations); false: Direct first,
    #   more calls only for items whose Direct confidence is below gate_tau, one preferred candidate at a time
    tool_menu: str = ""  # comma list of the tools offered to the selection call; empty = every available tool
    zoom_mode: str = "region"  # "grounded": crop around the object Grounding DINO finds for the cue; "region": fixed half-size crop
    audio_tags: bool = True  # False keeps only the speech transcript in the audio text (no sound-tag lines)
    reason_audio: str = "full"  # "none": reasoning calls get "Audio: not available." as in the training prompts (Direct keeps the audio text)
    tool_note: bool = False  # name the frames a tool added in the reasoning prompt after the tool
    final_answer: str = "direct"  # reported candidate: "direct", one variant, or a preference list "a,b" (first available, else Direct)
    config_path: str = ""


@dataclass(frozen=True)
class SampleState:
    """Immutable per-sample progress; each stage returns an updated copy."""

    input: AgentInput
    frames: tuple[dict[str, Any], ...] = ()
    audio: dict[str, Any] | None = None
    events: tuple[dict[str, Any], ...] = ()
    prediction: dict[str, Any] | None = None
    verification_notes: tuple[dict[str, Any], ...] = ()
    extra_evidence: tuple[dict[str, Any], ...] = ()
    evidence_gaps: tuple[dict[str, Any], ...] = ()
    choice_history: tuple[str, ...] = ()
    unresolvable: bool = False
    checks: tuple[dict[str, Any], ...] = ()
    all_cited_unsupported: bool = False
    reflection_done: bool = False
    rounds: int = 0
    calls: int = 0
    extra_frames: int = 0
    invalid_citations: int = 0
    elapsed: float = 0.0
    status: str = STATUS_OK
    flags: tuple[str, ...] = ()

    @property
    def alive(self) -> bool:
        return self.status == STATUS_OK
