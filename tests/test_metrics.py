import math

import numpy as np
import pytest

from lsa import entropy, kl_divergence


def test_entropy_matches_direct_formula() -> None:
    p = np.array([0.50, 0.25, 0.25])

    value = entropy(p)

    assert value == pytest.approx(
        -(0.50 * math.log2(0.50) + 2.0 * 0.25 * math.log2(0.25))
    )


def test_entropy_of_point_mass_is_zero() -> None:
    assert entropy([0.0, 1.0, 0.0]) == pytest.approx(0.0)


def test_entropy_of_uniform_distribution_is_log_dimension() -> None:
    assert entropy(np.full(8, 1.0 / 8.0)) == pytest.approx(math.log2(8.0))


def test_entropy_rejects_invalid_probability_vector() -> None:
    with pytest.raises(ValueError):
        entropy([0.5, 0.6])


def test_kl_divergence_matches_direct_formula() -> None:
    p = np.array([0.50, 0.25, 0.25])
    q = np.array([0.25, 0.25, 0.50])

    divergence = kl_divergence(p, q)

    assert divergence == pytest.approx(0.5 * math.log2(2.0) + 0.25 * math.log2(0.5))


def test_kl_divergence_is_zero_for_equal_distributions() -> None:
    p = np.array([0.20, 0.30, 0.50])

    assert kl_divergence(p, p) == pytest.approx(0.0)


def test_kl_divergence_ignores_zero_p_entries() -> None:
    p = np.array([0.0, 1.0])
    q = np.array([0.0, 1.0])

    assert kl_divergence(p, q) == pytest.approx(0.0)


def test_kl_divergence_is_infinite_when_q_zero_on_positive_p() -> None:
    p = np.array([0.50, 0.50])
    q = np.array([1.00, 0.00])

    assert kl_divergence(p, q) == math.inf


def test_kl_divergence_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        kl_divergence([1.0], [0.5, 0.5])
