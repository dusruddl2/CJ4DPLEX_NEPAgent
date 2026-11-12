"""Speech recognition (Whisper) and sound tagging (PANNs CNN14) for the observed audio.

Source: FutureOmni [4] motivates audio cues; AVP [3] motivates time-stamped evidence.
Changed from source: per-second tags are merged into runs so long sounds use one prompt line.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.media import ffmpeg_binary

logger = logging.getLogger(__name__)

WHISPER_RATE = 16000
PANNS_RATE = 32000
TAG_THRESHOLD = 0.3
TAG_BATCH = 64
FFMPEG_TIMEOUT_SEC = 300
SPEECH_LABELS = frozenset({"Speech", "Male speech, man speaking", "Female speech, woman speaking", "Conversation"})


def _decode_mono(video: str, rate: int) -> np.ndarray:
    """Decode the audio track to a mono float32 waveform."""
    with tempfile.NamedTemporaryFile(suffix=".f32le") as tmp:
        subprocess.run(
            [ffmpeg_binary(), "-y", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", tmp.name],
            capture_output=True, timeout=FFMPEG_TIMEOUT_SEC, check=True,
        )
        return np.fromfile(Path(tmp.name), dtype=np.float32)


def merge_tag_runs(per_second: list[list[tuple[str, float]]]) -> list[dict[str, Any]]:
    """Collapse consecutive seconds with the same label into one entry at the run start."""
    open_runs: dict[str, dict[str, Any]] = {}
    finished: list[dict[str, Any]] = []
    for second, labels in enumerate(per_second):
        current = dict(labels)
        for label in list(open_runs):
            if label not in current:
                finished.append(open_runs.pop(label))
        for label, score in current.items():
            if label in open_runs:
                open_runs[label] = {
                    **open_runs[label],
                    "t_end": float(second + 1),
                    "score": max(open_runs[label]["score"], score),
                }
            else:
                open_runs[label] = {"t_sec": float(second), "t_end": float(second + 1), "label": label, "score": score}
    return sorted([*finished, *open_runs.values()], key=lambda item: item["t_sec"])


class AudioAnalyzer:
    """Load both audio models once and analyse one observed clip at a time."""

    def __init__(self, whisper_model: str = "small", device: str = "cuda") -> None:
        import whisper
        from panns_inference import AudioTagging, labels

        self.whisper = whisper.load_model(whisper_model, device=device)
        self.tagger = AudioTagging(checkpoint_path=None, device=device)
        self.labels = labels

    def _transcribe(self, waveform: np.ndarray) -> list[dict[str, Any]]:
        result = self.whisper.transcribe(waveform, fp16=True, condition_on_previous_text=False)
        return [
            {"t_start": float(s["start"]), "t_end": float(s["end"]), "text": s["text"].strip()}
            for s in result["segments"]
            if s["text"].strip()
        ]

    def _tag(self, waveform: np.ndarray) -> list[dict[str, Any]]:
        seconds = len(waveform) // PANNS_RATE
        per_second: list[list[tuple[str, float]]] = []
        for start in range(0, seconds, TAG_BATCH):
            stop = min(start + TAG_BATCH, seconds)
            batch = np.stack([waveform[s * PANNS_RATE : (s + 1) * PANNS_RATE] for s in range(start, stop)])
            scores, _ = self.tagger.inference(batch)
            for row in scores:
                top = np.argsort(row)[::-1][:2]
                per_second.append(
                    [
                        (self.labels[i], float(row[i]))
                        for i in top
                        if row[i] >= TAG_THRESHOLD and self.labels[i] not in SPEECH_LABELS
                    ]
                )
        return merge_tag_runs(per_second)

    def analyze(self, video: str) -> dict[str, Any]:
        """Return time-stamped speech segments and sound tags for the clip."""
        return {
            "asr": self._transcribe(_decode_mono(video, WHISPER_RATE)),
            "tags": self._tag(_decode_mono(video, PANNS_RATE)),
        }
