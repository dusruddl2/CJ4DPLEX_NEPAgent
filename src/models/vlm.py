"""Common VLM interface: greedy generation and option probabilities.

Source: none (infrastructure). Every method shares this class, temperature 0 and the same
image resolution, so baselines and ablations differ only in prompts and modules.
LoRA adapters are merged into a separate model directory first (src/train/merge_adapter.py).
"""
from __future__ import annotations

import json
import logging
import math
import re
import os
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from dotenv import load_dotenv
from PIL import Image

logger = logging.getLogger(__name__)

load_dotenv()
DEFAULT_MODEL = os.environ.get("VLM_PATH") or "Qwen/Qwen3-VL-8B-Instruct"
MAX_IMAGE_PIXELS = 448 * 448
IMAGE_RESIZE_FACTOR = 32
VIDEO_PATCH_SIZE = 16
MAX_PROCESSOR_PIXELS = 720 * 720
VIDEO_ENGINE_MAX_IMAGES = 16
MIN_VIDEO_PIXELS = 32 * 32 * 4
MISSING_OPTION_LOGPROB = -30.0
LOGPROB_TOP_K = 500
MERGED_SUFFIX = "_merged"


@dataclass(frozen=True)
class VLMRequest:
    """One conversation turn: image paths, user text, optional assistant prefix."""

    text: str
    images: tuple[str, ...] = ()
    assistant_prefix: str = ""
    max_pixels: int = MAX_IMAGE_PIXELS
    as_video: bool = False
    video_fps: float = 0.0


@dataclass(frozen=True)
class GenerationConfig:
    max_tokens: int = 1024
    temperature: float = 0.0


class VLMBackend(Protocol):
    def generate(self, requests: Sequence[VLMRequest], cfg: GenerationConfig = ...) -> list[str | None]: ...

    def generate_scored(
        self, requests: Sequence[VLMRequest], cfg: GenerationConfig = ...
    ) -> list[tuple[str, float] | None]: ...

    def option_probs(
        self, requests: Sequence[VLMRequest], letters: Sequence[Sequence[str]]
    ) -> list[dict[str, float] | None]: ...

    def sample_votes(
        self, requests: Sequence[VLMRequest], letters: Sequence[Sequence[str]], samples: int, temperature: float
    ) -> list[dict[str, float] | None]: ...


def softmax_option_logprobs(token_logprobs: dict[str, float], letters: Sequence[str]) -> dict[str, float]:
    """Convert first-token log-probabilities into a distribution over option letters.

    Token variants such as " A" and "A" are merged by summing their probability.
    An option absent from the returned tokens gets a very low constant log-probability.
    """
    merged: dict[str, float] = {}
    for token, logprob in token_logprobs.items():
        letter = token.strip()
        if letter in letters:
            merged[letter] = merged.get(letter, 0.0) + math.exp(logprob)
    masses = {letter: merged.get(letter, math.exp(MISSING_OPTION_LOGPROB)) for letter in letters}
    total = sum(masses.values())
    return {letter: mass / total for letter, mass in masses.items()}


def resolve_model_path(base_model: str, lora_path: str) -> str:
    """Return the base model, or the merged model directory for a trained adapter."""
    if not lora_path:
        return base_model
    merged = f"{lora_path.rstrip('/')}{MERGED_SUFFIX}"
    if not os.path.isdir(merged):
        raise FileNotFoundError(f"merged model not found: {merged}; run src/train/merge_adapter.py first")
    return merged


class MockVLM:
    """Scripted backend for tests: replies depend on the prompt type, without a GPU."""

    def __init__(
        self,
        top_prob: float = 0.9,
        frame_supported: bool = True,
        broken_chain: bool = False,
        sufficient: bool = True,
        gap_tool: str = "dense_resample",
        resolvable: bool = True,
        broken_gap: bool = False,
        chain_update: bool = False,
        bad_citation: bool = False,
    ) -> None:
        self.top_prob = top_prob
        self.frame_supported = frame_supported
        self.broken_chain = broken_chain
        self.sufficient = sufficient
        self.gap_tool = gap_tool
        self.resolvable = resolvable
        self.broken_gap = broken_gap
        self.chain_update = chain_update
        self.bad_citation = bad_citation
        self.requests: list[VLMRequest] = []

    def generate(self, requests: Sequence[VLMRequest], cfg: GenerationConfig = GenerationConfig()) -> list[str | None]:
        self.requests.extend(requests)
        return [self._reply(request) for request in requests]

    def generate_scored(
        self, requests: Sequence[VLMRequest], cfg: GenerationConfig = GenerationConfig()
    ) -> list[tuple[str, float] | None]:
        """Open-ended replies with a fixed mean token log-probability (tests override `_score`)."""
        self.requests.extend(requests)
        return [(reply, self._score(request)) if reply is not None else None
                for request, reply in zip(requests, [self._reply(r) for r in requests])]

    def _score(self, request: VLMRequest) -> float:
        return -0.5

    def _reply(self, request: VLMRequest) -> str:
        if "Describe what is visible" in request.text:
            ids = re.findall(r'"frame": "(#[^"]+)"', request.text)
            return json.dumps({"captions": [{"frame": i, "caption": f"scene {i}"} for i in ids]})
        if "Update the event chain" in request.text:
            new_ids = request.text.split("New frames: ")[1].split("\n")[0].split(", ")
            return json.dumps(
                {"events": [{"id": "E9", "frames": [new_ids[0]], "subject": "n", "action": "o", "object": "p",
                             "state_change": "q", "modality": ["visual"]}]}
            )
        if "chronological chain" in request.text:
            if self.broken_chain:
                return "no json here"
            return json.dumps(
                {"events": [{"id": "E1", "frames": ["#01", "#02"], "subject": "a", "action": "b", "object": "c",
                             "state_change": "d", "modality": ["visual"]}]}
            )
        if "observed cues" in request.text:
            return json.dumps(
                {"options": {"A": {"evidence_events": ["E1"], "evidence_frames": ["#02", "#99"] if self.bad_citation else ["#02"], "support": "seen"}},
                 "next_event": "x", "evidence_sufficient": self.sufficient}
            )
        if "Identify the evidence gap" in request.text:
            if self.broken_gap:
                return "no json here"
            return json.dumps(
                {"missing": "who acts", "where": ["#05", "#07"], "tool": self.gap_tool, "region": "center",
                 "resolvable": self.resolvable, "chain_update_required": self.chain_update}
            )
        if "state of the scene at the end" in request.text:
            return json.dumps({"ongoing_action": "walking", "held_objects": ["bag"], "object_states": ["door open"]})
        return "because"

    def option_probs(
        self, requests: Sequence[VLMRequest], letters: Sequence[Sequence[str]]
    ) -> list[dict[str, float] | None]:
        self.requests.extend(requests)
        output: list[dict[str, float] | None] = []
        for options in letters:
            if tuple(options) == ("Yes", "No"):
                output.append({"Yes": 0.9, "No": 0.1} if self.frame_supported else {"Yes": 0.1, "No": 0.9})
                continue
            rest = (1 - self.top_prob) / (len(options) - 1)
            output.append({o: (self.top_prob if o == options[0] else rest) for o in options})
        return output

    def sample_votes(
        self, requests: Sequence[VLMRequest], letters: Sequence[Sequence[str]], samples: int, temperature: float
    ) -> list[dict[str, float] | None]:
        self.requests.extend(requests)
        return [{o: (1.0 if o == options[0] else 0.0) for o in options} for options in letters]


class VLLMQwen:
    """Offline vLLM engine for Qwen3-VL with a multi-image prompt and optional LoRA adapter."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        max_images: int = 64,
        allow_video: bool = False,
        gpu_memory_utilization: float = float(os.environ.get("VLM_GPU_FRACTION", "0.6")),
        max_model_len: int = int(os.environ.get("VLM_MAX_MODEL_LEN", "32768")),  # lower it for 32B on one A100 (KV cache)
        seed: int = 0,
    ) -> None:
        from transformers import AutoProcessor
        from vllm import LLM

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.llm = LLM(
            model=model_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={
                "image": VIDEO_ENGINE_MAX_IMAGES if allow_video else max_images,
                "video": 1 if allow_video else 0,
            },
            max_logprobs=LOGPROB_TOP_K,
            mm_processor_cache_gb=0,
            mm_processor_kwargs={"size": {"shortest_edge": MIN_VIDEO_PIXELS, "longest_edge": MAX_PROCESSOR_PIXELS}},
            seed=seed,
        )
        self._token_ids: dict[tuple[str, ...], list[int]] = {}

    def _allowed_ids(self, letters: Sequence[str]) -> list[int]:
        """Single-token ids of each letter, with and without a leading space."""
        key = tuple(letters)
        if key not in self._token_ids:
            ids: set[int] = set()
            for letter in letters:
                for variant in (letter, f" {letter}"):
                    encoded = self.tokenizer.encode(variant, add_special_tokens=False)
                    if len(encoded) == 1:
                        ids.add(encoded[0])
            self._token_ids[key] = sorted(ids)
        return self._token_ids[key]

    @staticmethod
    def _load_image(path: str, max_pixels: int = MAX_IMAGE_PIXELS) -> Image.Image:
        from qwen_vl_utils import smart_resize

        with Image.open(path) as opened:
            image = opened.convert("RGB")
        height, width = smart_resize(
            image.height, image.width, factor=IMAGE_RESIZE_FACTOR, max_pixels=max_pixels
        )
        return image.resize((width, height), Image.Resampling.BICUBIC)

    def _build_video(self, request: VLMRequest) -> dict[str, Any]:
        """Native Qwen video path for the same frames, with their temporal metadata."""
        from qwen_vl_utils import process_vision_info

        video = {
            "type": "video",
            "video": [f"file://{path}" for path in request.images],
            "sample_fps": request.video_fps,
            "min_pixels": MIN_VIDEO_PIXELS,
            "max_pixels": request.max_pixels,
        }
        messages = [{"role": "user", "content": [video, {"type": "text", "text": request.text}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, videos, video_kwargs = process_vision_info(
            messages, image_patch_size=VIDEO_PATCH_SIZE, return_video_kwargs=True, return_video_metadata=True
        )
        return {
            "prompt": prompt + request.assistant_prefix,
            "multi_modal_data": {"video": videos},
            "mm_processor_kwargs": video_kwargs,
        }

    def _build(self, request: VLMRequest) -> dict[str, Any]:
        if request.as_video:
            return self._build_video(request)
        content: list[dict[str, Any]] = [{"type": "image"} for _ in request.images]
        content.append({"type": "text", "text": request.text})
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        item: dict[str, Any] = {"prompt": prompt + request.assistant_prefix}
        if request.images:
            item["multi_modal_data"] = {"image": [self._load_image(path, request.max_pixels) for path in request.images]}
        return item

    def _build_all(self, requests: Sequence[VLMRequest]) -> tuple[list[dict[str, Any]], list[int]]:
        inputs: list[dict[str, Any]] = []
        kept: list[int] = []
        for index, request in enumerate(requests):
            try:
                inputs.append(self._build(request))
                kept.append(index)
            except (OSError, ValueError) as exc:
                logger.error("failed to build VLM input #%d: %s", index, exc)
        return inputs, kept

    def generate(
        self, requests: Sequence[VLMRequest], cfg: GenerationConfig = GenerationConfig()
    ) -> list[str | None]:
        from vllm import SamplingParams

        outputs: list[str | None] = [None] * len(requests)
        inputs, kept = self._build_all(requests)
        if not inputs:
            return outputs
        params = SamplingParams(temperature=cfg.temperature, max_tokens=cfg.max_tokens)
        results = self.llm.generate(inputs, params, use_tqdm=False)
        for index, result in zip(kept, results):
            outputs[index] = result.outputs[0].text
        return outputs

    def generate_scored(
        self, requests: Sequence[VLMRequest], cfg: GenerationConfig = GenerationConfig()
    ) -> list[tuple[str, float] | None]:
        """Free-text generation plus the mean token log-probability of the reply (open-ended confidence)."""
        from vllm import SamplingParams

        outputs: list[tuple[str, float] | None] = [None] * len(requests)
        inputs, kept = self._build_all(requests)
        if not inputs:
            return outputs
        params = SamplingParams(temperature=cfg.temperature, max_tokens=cfg.max_tokens, logprobs=1)
        results = self.llm.generate(inputs, params, use_tqdm=False)
        for index, result in zip(kept, results):
            out = result.outputs[0]
            count = max(1, len(out.token_ids))
            total = out.cumulative_logprob if out.cumulative_logprob is not None else 0.0
            outputs[index] = (out.text, total / count)
        return outputs

    def option_probs(
        self, requests: Sequence[VLMRequest], letters: Sequence[Sequence[str]]
    ) -> list[dict[str, float] | None]:
        """Force the next token to be an option letter and normalise over the letters only.

        vLLM reports logprobs before the allowed-token mask, so a wide top-k is needed.
        """
        from vllm import SamplingParams

        outputs: list[dict[str, float] | None] = [None] * len(requests)
        inputs, kept = self._build_all(requests)
        if not inputs:
            return outputs
        params = [
            SamplingParams(
                temperature=0.0,
                max_tokens=1,
                logprobs=LOGPROB_TOP_K,
                allowed_token_ids=self._allowed_ids(letters[index]),
            )
            for index in kept
        ]
        results = self.llm.generate(inputs, params, use_tqdm=False)
        for index, result in zip(kept, results):
            step = result.outputs[0].logprobs[0] if result.outputs[0].logprobs else {}
            tokens = {item.decoded_token: item.logprob for item in step.values()}
            outputs[index] = softmax_option_logprobs(tokens, letters[index])
        return outputs

    def sample_votes(
        self, requests: Sequence[VLMRequest], letters: Sequence[Sequence[str]], samples: int, temperature: float
    ) -> list[dict[str, float] | None]:
        """Sample one option letter `samples` times per request and return vote shares."""
        from vllm import SamplingParams

        outputs: list[dict[str, float] | None] = [None] * len(requests)
        inputs, kept = self._build_all(requests)
        if not inputs:
            return outputs
        params = [
            SamplingParams(
                temperature=temperature,
                max_tokens=1,
                n=samples,
                allowed_token_ids=self._allowed_ids(letters[index]),
            )
            for index in kept
        ]
        for index, result in zip(kept, self.llm.generate(inputs, params, use_tqdm=False)):
            votes = [completion.text.strip() for completion in result.outputs]
            outputs[index] = {letter: votes.count(letter) / len(votes) for letter in letters[index]}
        return outputs
