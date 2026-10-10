import math
from collections import Counter
from itertools import product

import numpy as np
import pytest

from lsa import (
    log_a_lambda_hybrid,
    log_a_lambda_saddlepoint,
    partition_multiplicities,
)


def test_partition_multiplicities_counts_equal_parts() -> None:
    assert partition_multiplicities((4, 2, 2, 1, 1, 1)) == (
        (1, 3),
        (2, 2),
        (4, 1),
    )


def test_partition_multiplicities_rejects_nonpositive_parts() -> None:
    with pytest.raises(ValueError):
        partition_multiplicities((3, 0, 1))


def test_partition_multiplicities_rejects_noninteger_parts() -> None:
    with pytest.raises(ValueError):
        partition_multiplicities((3, 1.2, 1))


def test_saddlepoint_matches_uniform_singleton_profile() -> None:
    d = 1_000
    partition = (1,) * 50
    p = np.full(d, 1.0 / d)

    approximation = log_a_lambda_saddlepoint(p, partition)

    assert approximation.converged
    assert approximation.residual_norm < 1e-8
    assert approximation.log_weight == pytest.approx(
        _uniform_log_pattern_weight(d, partition),
        abs=0.01,
    )


def test_saddlepoint_matches_uniform_mixed_profile() -> None:
    d = 1_000
    partition = (2,) * 10 + (1,) * 30
    p = np.full(d, 1.0 / d)

    approximation = log_a_lambda_saddlepoint(p, partition)

    assert approximation.converged
    assert approximation.residual_norm < 1e-8
    assert approximation.log_weight == pytest.approx(
        _uniform_log_pattern_weight(d, partition),
        abs=0.05,
    )


def test_saddlepoint_returns_impossible_profile_for_small_support() -> None:
    p = np.array([0.5, 0.5, 0.0])

    approximation = log_a_lambda_saddlepoint(p, (1, 1, 1))

    assert approximation.log_weight == -math.inf
    assert not approximation.converged


def test_hybrid_with_no_top_atoms_matches_pure_saddlepoint() -> None:
    d = 1_000
    partition = (2,) * 10 + (1,) * 30
    p = np.full(d, 1.0 / d)

    pure = log_a_lambda_saddlepoint(p, partition)
    hybrid = log_a_lambda_hybrid(p, partition, top_count=0)

    assert hybrid.converged
    assert hybrid.top_count == 0
    assert hybrid.log_weight == pytest.approx(pure.log_weight)


def test_hybrid_with_full_support_is_exact() -> None:
    p = np.array([0.60, 0.30, 0.10])
    partition = (2, 1)

    hybrid = log_a_lambda_hybrid(p, partition, top_count=3)

    assert hybrid.converged
    assert hybrid.top_count == 3
    assert hybrid.tail_saddlepoints == 0
    assert hybrid.log_weight == pytest.approx(
        math.log(_brute_force_pattern_weight(p, partition))
    )


def test_hybrid_improves_concentrated_profile() -> None:
    p = np.array([0.70, 0.16, 0.07, 0.04, 0.02, 0.01])
    partition = (6, 1, 1)
    exact_log = math.log(_brute_force_pattern_weight(p, partition))

    pure = log_a_lambda_saddlepoint(p, partition)
    hybrid = log_a_lambda_hybrid(p, partition, top_count=2)

    assert hybrid.converged
    assert abs(hybrid.log_weight - exact_log) < abs(pure.log_weight - exact_log)


def _uniform_log_pattern_weight(d: int, partition: tuple[int, ...]) -> float:
    counts = Counter(partition)
    total_count = sum(partition)
    observed_symbols = len(partition)
    log_value = (
        math.lgamma(total_count + 1)
        - total_count * math.log(d)
        + math.lgamma(d + 1)
        - math.lgamma(d - observed_symbols + 1)
    )
    for part, multiplicity in counts.items():
        log_value -= multiplicity * math.lgamma(part + 1)
        log_value -= math.lgamma(multiplicity + 1)
    return log_value


def _brute_force_pattern_weight(p: np.ndarray, partition: tuple[int, ...]) -> float:
    total_count = sum(partition)
    target_profile = tuple(sorted(partition, reverse=True))
    probability = 0.0
    for counts in product(range(total_count + 1), repeat=p.size):
        if sum(counts) != total_count:
            continue
        profile = tuple(sorted((count for count in counts if count > 0), reverse=True))
        if profile != target_profile:
            continue
        log_probability = math.lgamma(total_count + 1)
        for count, prob in zip(counts, p, strict=True):
            log_probability -= math.lgamma(count + 1)
            if count > 0:
                log_probability += count * math.log(float(prob))
        probability += math.exp(log_probability)
    return probability
