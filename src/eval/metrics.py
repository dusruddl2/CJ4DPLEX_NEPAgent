"""Evaluation metrics: accuracy, bootstrap CI, exact McNemar test and ECE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import binomtest

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
ECE_BINS = 10
CI_LOW_PERCENT = 2.5
CI_HIGH_PERCENT = 97.5


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float


def accuracy(correct: Sequence[bool]) -> float:
    if len(correct) == 0:
        raise ValueError("accuracy needs at least one sample")
    return float(np.mean(np.asarray(correct, dtype=float)))


def _bootstrap_indices(size: int, resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, size, size=(resamples, size))


def bootstrap_accuracy(
    correct: Sequence[bool], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> Interval:
    """Percentile bootstrap CI over samples."""
    values = np.asarray(correct, dtype=float)
    means = values[_bootstrap_indices(len(values), resamples, seed)].mean(axis=1)
    return Interval(
        float(values.mean()),
        float(np.percentile(means, CI_LOW_PERCENT)),
        float(np.percentile(means, CI_HIGH_PERCENT)),
    )


def bootstrap_difference(
    method_a: Sequence[bool],
    method_b: Sequence[bool],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Interval:
    """Paired percentile bootstrap CI of accuracy(a) - accuracy(b)."""
    a_values = np.asarray(method_a, dtype=float)
    b_values = np.asarray(method_b, dtype=float)
    if a_values.shape != b_values.shape:
        raise ValueError("paired comparison needs equal-length results")
    diff = a_values - b_values
    means = diff[_bootstrap_indices(len(diff), resamples, seed)].mean(axis=1)
    return Interval(
        float(diff.mean()),
        float(np.percentile(means, CI_LOW_PERCENT)),
        float(np.percentile(means, CI_HIGH_PERCENT)),
    )


def bootstrap_ratio(
    numerators: Sequence[float],
    denominators: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Interval:
    """Percentile bootstrap CI of sum(numerators) / sum(denominators), resampling samples."""
    top = np.asarray(numerators, dtype=float)
    bottom = np.asarray(denominators, dtype=float)
    if top.shape != bottom.shape or top.size == 0:
        raise ValueError("ratio needs equal-length, non-empty inputs")
    if np.any(bottom <= 0):
        raise ValueError("ratio denominators must be positive")
    indices = _bootstrap_indices(len(top), resamples, seed)
    ratios = top[indices].sum(axis=1) / bottom[indices].sum(axis=1)
    return Interval(
        float(top.sum() / bottom.sum()),
        float(np.percentile(ratios, CI_LOW_PERCENT)),
        float(np.percentile(ratios, CI_HIGH_PERCENT)),
    )


def mcnemar_exact(method_a: Sequence[bool], method_b: Sequence[bool]) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs."""
    a_values = np.asarray(method_a, dtype=bool)
    b_values = np.asarray(method_b, dtype=bool)
    if a_values.shape != b_values.shape:
        raise ValueError("paired comparison needs equal-length results")
    only_a = int(np.sum(a_values & ~b_values))
    only_b = int(np.sum(~a_values & b_values))
    if only_a + only_b == 0:
        return 1.0
    return float(binomtest(only_a, only_a + only_b, 0.5).pvalue)


def reliability_bins(
    confidence: Sequence[float], correct: Sequence[bool], bins: int = ECE_BINS
) -> list[dict[str, float]]:
    """Return count, mean confidence and accuracy per equal-width confidence bin."""
    conf = np.asarray(confidence, dtype=float)
    hits = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    for index in range(bins):
        upper_closed = index == bins - 1
        mask = (conf >= edges[index]) & ((conf <= edges[index + 1]) if upper_closed else (conf < edges[index + 1]))
        count = int(mask.sum())
        rows.append(
            {
                "low": float(edges[index]),
                "high": float(edges[index + 1]),
                "count": count,
                "confidence": float(conf[mask].mean()) if count else 0.0,
                "accuracy": float(hits[mask].mean()) if count else 0.0,
            }
        )
    return rows


def expected_calibration_error(
    confidence: Sequence[float], correct: Sequence[bool], bins: int = ECE_BINS
) -> float:
    total = len(confidence)
    if total == 0:
        raise ValueError("ECE needs at least one sample")
    rows = reliability_bins(confidence, correct, bins)
    return float(sum(row["count"] / total * abs(row["accuracy"] - row["confidence"]) for row in rows))
