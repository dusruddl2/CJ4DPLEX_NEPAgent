"""Run one configuration on a batch: sample frames, dispatch the method, serialise the results."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from src.agent import observer
from src.agent.observer import FrameIO
from src.models.vlm import VLMBackend
from src.schemas import AgentInput, RunConfig, SampleState
from src.utils.cache import JsonCache

AUDIO_CACHE_KEY: dict[str, Any] = {"whisper": "small", "panns": "cnn14", "version": 1}


@dataclass
class Services:
    """Shared models and stores. `audio` is optional when audio results are already cached."""

    vlm: VLMBackend
    cache: JsonCache
    frames: FrameIO
    audio: Any | None = None


def build_services(vlm: VLMBackend, cache_root: str | Path, audio: Any | None = None) -> Services:
    return Services(vlm=vlm, cache=JsonCache(cache_root), frames=FrameIO(Path(cache_root)), audio=audio)


def attach_audio(state: SampleState, services: Services) -> SampleState:
    """Load cached audio cues, computing them only if a model was provided."""
    from src.utils.media import has_audio_stream

    if not state.alive:
        return state
    sample_id = state.input.sample_id
    cached = services.cache.get(sample_id, "audio", AUDIO_CACHE_KEY)
    if cached is None:
        if services.audio is None:
            raise RuntimeError(f"audio cache is missing for {sample_id}; run scripts/precompute_audio.py first")
        analysed = services.audio.analyze(state.input.prefix_path) if has_audio_stream(state.input.prefix_path) else None
        cached = {"audio": analysed}
        services.cache.put(sample_id, "audio", AUDIO_CACHE_KEY, cached)
    return replace(state, audio=cached["audio"])


def _merge(base: Sequence[SampleState], updates: Sequence[SampleState]) -> list[SampleState]:
    by_id = {state.input.sample_id: state for state in updates}
    return [by_id.get(state.input.sample_id, state) for state in base]


def _run_nep_agent(states: list[SampleState], cfg: RunConfig, services: Services) -> list[SampleState]:
    from src.nep_agent import run

    return run(states, cfg, services)


_METHODS: dict[str, Callable[[list[SampleState], RunConfig, Services], list[SampleState]]] = {"nep_agent": _run_nep_agent}


def run_batch(samples: Sequence[AgentInput], cfg: RunConfig, services: Services) -> list[dict[str, Any]]:
    """Run one configuration on a batch and return the JSON-ready results."""
    from src.nep_agent import lift, to_result

    if cfg.method not in _METHODS:
        raise ValueError(f"unknown method: {cfg.method}")
    initial = [observer.sample_frames(sample, cfg, services.frames) for sample in samples]
    live = [state for state in initial if state.alive]
    finished = _METHODS[cfg.method](live, cfg, services) if live else []
    return [to_result(lift(state), cfg) for state in _merge(initial, finished)]
