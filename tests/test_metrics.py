"""Unit tests with hand-computed expected values for the evaluation metrics."""
import pytest

from src.eval.metrics import (
    accuracy,
    bootstrap_accuracy,
    bootstrap_difference,
    expected_calibration_error,
    mcnemar_exact,
)
from src.models.vlm import softmax_option_logprobs
from src.utils.cache import JsonCache, config_hash


def test_accuracy_counts_true_fraction() -> None:
    assert accuracy([True, False, True, True]) == 0.75


def test_accuracy_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        accuracy([])


def test_bootstrap_interval_contains_point_estimate() -> None:
    result = bootstrap_accuracy([True] * 60 + [False] * 40)
    assert result.low <= result.point <= result.high
    assert result.point == pytest.approx(0.6)


def test_bootstrap_difference_is_zero_for_identical_methods() -> None:
    result = bootstrap_difference([True, False, True], [True, False, True])
    assert (result.point, result.low, result.high) == (0.0, 0.0, 0.0)


def test_mcnemar_uses_only_discordant_pairs() -> None:
    method_a = [True] * 8 + [False] * 2
    method_b = [False] * 8 + [True] * 2
    # Two-sided binomial test of 8 successes in 10 trials: 2 * 56 / 1024.
    assert mcnemar_exact(method_a, method_b) == pytest.approx(112 / 1024)


def test_mcnemar_without_discordant_pairs_is_one() -> None:
    assert mcnemar_exact([True, False], [True, False]) == 1.0


def test_ece_is_zero_when_confidence_matches_accuracy() -> None:
    assert expected_calibration_error([0.75] * 4, [True, True, True, False]) == pytest.approx(0.0)


def test_ece_measures_overconfidence() -> None:
    assert expected_calibration_error([0.95] * 2, [True, False]) == pytest.approx(0.45)


def test_ece_places_confidence_one_in_last_bin() -> None:
    assert expected_calibration_error([1.0], [True]) == pytest.approx(0.0)


def test_softmax_merges_token_variants_and_normalises() -> None:
    import math

    probs = softmax_option_logprobs(
        {"A": math.log(0.3), " A": math.log(0.3), "B": math.log(0.2), "the": math.log(0.2)},
        ["A", "B"],
    )
    assert probs["A"] == pytest.approx(0.75, rel=1e-6)
    assert probs["B"] == pytest.approx(0.25, rel=1e-6)


def test_softmax_gives_missing_option_negligible_mass() -> None:
    import math

    probs = softmax_option_logprobs({"A": math.log(0.9)}, ["A", "B"])
    assert probs["B"] < 1e-9


def test_config_hash_ignores_key_order() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_cache_round_trip_and_miss(tmp_path) -> None:
    cache = JsonCache(tmp_path)
    assert cache.get("s1", "chain", {"n": 16}) is None
    cache.put("s1", "chain", {"n": 16}, {"events": []})
    assert cache.get("s1", "chain", {"n": 16}) == {"events": []}
    assert cache.get("s1", "chain", {"n": 32}) is None
