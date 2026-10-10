"""Independent identities and domain boundaries for the ALT comparators."""

import math
from itertools import product

import numpy as np
import pytest

from lsa.alt.baselines import (
    ABSOLUTE_DISCOUNTING_RULE,
    ABSOLUTE_DISCOUNTING_VERSION,
    DIRICHLET_EXPONENTS,
    absolute_discounting,
    add_constant,
    dirichlet_concentration_mixture,
    dirichlet_log_evidence,
    evaluate_baseline,
    good_turing_hybrid,
    kl_bits,
    natural_oracle,
    ristad,
    validate_counts,
)


def test_dirichlet_evidence_sequence_enumeration_and_predictive_identity():
    # Counts do not carry a multinomial coefficient: summing sequences gives 1.
    for alpha in (2 ** -24, 0.5, 1.0, 16.0):
        total = 0.0
        for seq in product(range(3), repeat=3):
            counts = np.bincount(seq, minlength=3)
            total += math.exp(dirichlet_log_evidence(counts, alpha))
        assert total == pytest.approx(1, abs=2e-12)
        counts = np.array([3, 0, 1])
        ratios = [math.exp(dirichlet_log_evidence(counts + np.eye(3, dtype=int)[i], alpha)
                           - dirichlet_log_evidence(counts, alpha)) for i in range(3)]
        np.testing.assert_allclose(ratios, (counts + alpha) / (4 + 3 * alpha), rtol=2e-12)


def test_concentration_mixture_sequential_chain_rule():
    counts = np.zeros(5, dtype=int)
    log_probability = 0.0
    for symbol in [2, 2, 4, 2, 0, 0, 1]:
        result = dirichlet_concentration_mixture(counts, exponents=DIRICHLET_EXPONENTS)
        log_probability += math.log(result.probabilities[symbol])
        counts[symbol] += 1
    result = dirichlet_concentration_mixture(counts, exponents=DIRICHLET_EXPONENTS)
    assert log_probability == pytest.approx(result.diagnostics["mixture_log_evidence_nats"], abs=2e-12)
    np.testing.assert_allclose(result.probabilities.sum(), 1, atol=2e-13)
    assert len(result.diagnostics["posterior"]) == 29


def test_concentration_empty_sample_is_uniform_with_equal_prior():
    result = dirichlet_concentration_mixture([0] * 7, exponents=DIRICHLET_EXPONENTS)
    np.testing.assert_allclose(result.probabilities, np.full(7, 1 / 7))
    np.testing.assert_allclose(result.diagnostics["posterior"], np.full(29, 1 / 29))
    assert result.diagnostics["mixture_log_evidence_nats"] == pytest.approx(0)


def test_gt_empirical_switch_equality_and_zero_class():
    # For t=1, c_2=1, equality uses GT, not the empirical branch.
    np.testing.assert_allclose(good_turing_hybrid([1, 1, 2, 0, 0]), np.array([2, 2, 2, 1.5, 1.5]) / 9)
    np.testing.assert_allclose(good_turing_hybrid([0, 0, 0]), [1 / 3] * 3)
    np.testing.assert_allclose(good_turing_hybrid([3, 4]), [3 / 7, 4 / 7])


def test_ristad_boundaries_and_normalization():
    np.testing.assert_allclose(ristad([0, 0]), [0.5, 0.5])
    np.testing.assert_allclose(ristad([1, 4]), add_constant([1, 4], 1))
    for counts in product(range(4), repeat=3):
        q = ristad(counts)
        assert np.all(q > 0)
        assert q.sum() == pytest.approx(1, abs=2e-15)


@pytest.mark.parametrize("counts, discount, expected", [
    ([0, 3, 4], 1 / 2, [1 / 7, 5 / 14, 1 / 2]),  # No singletons or doubletons.
    ([0, 2, 4], 1 / 2, [1 / 6, 1 / 4, 7 / 12]),  # No singletons.
    ([0, 1, 4], 1 / 2, [1 / 5, 1 / 10, 7 / 10]),  # No doubletons.
    ([0, 1, 1], 0.9, [9 / 10, 1 / 20, 1 / 20]),  # All observations singleton.
    ([1, 0, 0], 0.9, [1 / 10, 9 / 20, 9 / 20]),  # One observation.
    ([3, 0, 0], 0.9, [7 / 10, 3 / 20, 3 / 20]),  # One observed type.
    ([1, 1, 2, 0, 0], 2 / 3, [1 / 12, 1 / 12, 1 / 3, 1 / 4, 1 / 4]),
])
def test_ad_hand_calculated_profiles(counts, discount, expected):
    result = absolute_discounting(counts)
    assert result.status == "defined" and result.reason is None
    assert result.diagnostics["discount"] == pytest.approx(discount)
    assert result.diagnostics["boundary_case"] is None
    np.testing.assert_allclose(result.probabilities, expected, rtol=0, atol=2e-16)
    uniform = np.full(len(counts), 1 / len(counts))
    assert math.isfinite(kl_bits(uniform, result.probabilities))


@pytest.mark.parametrize("counts, expected, boundary", [
    ([0], [1], "empty_sample_uniform"),
    ([0, 0, 0], [1 / 3] * 3, "empty_sample_uniform"),
    ([1], [1], "full_alphabet_empirical"),
    ([1, 2], [1 / 3, 2 / 3], "full_alphabet_empirical"),
    ([3, 4], [3 / 7, 4 / 7], "full_alphabet_empirical"),
    ([1, 1], [1 / 2, 1 / 2], "full_alphabet_empirical"),
])
def test_ad_explicit_empty_and_full_alphabet_extensions(counts, expected, boundary):
    result = absolute_discounting(counts)
    assert result.status == "defined" and result.reason is None
    assert result.diagnostics["boundary_case"] == boundary
    assert result.diagnostics["discount"] == 0
    np.testing.assert_allclose(result.probabilities, expected, rtol=0, atol=2e-16)


def test_ad_versioned_diagnostics_and_compatible_dispatch():
    assert ABSOLUTE_DISCOUNTING_RULE == "falahatgar2017_experimental"
    assert ABSOLUTE_DISCOUNTING_VERSION == "robust-ad-v1"
    for counts in ([0, 0, 0], [0, 2, 4], [1, 2, 4]):
        result = evaluate_baseline(
            "absolute_discounting", counts, target=[1 / 3] * 3,
            dirichlet_exponents=DIRICHLET_EXPONENTS,
        )
        assert result.diagnostics["estimator_rule"] == ABSOLUTE_DISCOUNTING_RULE
        assert result.diagnostics["estimator_version"] == ABSOLUTE_DISCOUNTING_VERSION
        assert result.diagnostics["n"] == sum(counts)
        assert result.diagnostics["distinct"] == sum(c > 0 for c in counts)
        assert result.diagnostics["c1"] == counts.count(1)
        assert result.diagnostics["c2"] == counts.count(2)
        np.testing.assert_array_equal(
            result.probabilities, absolute_discounting(counts).probabilities,
        )


def test_ad_exhaustive_small_profiles_are_positive_normalized_and_label_invariant():
    target = [0.05, 0.15, 0.3, 0.5]
    for counts in product(range(5), repeat=4):
        result = absolute_discounting(counts)
        q = result.probabilities
        assert result.status == "defined" and result.reason is None
        assert np.all(np.isfinite(q)) and np.all(q > 0)
        assert q.sum() == pytest.approx(1, abs=3e-16)
        assert math.isfinite(kl_bits(target, q))
        np.testing.assert_array_equal(
            absolute_discounting(counts[::-1]).probabilities, q[::-1],
        )
        # Equal count classes must receive equal probabilities.
        for count in set(counts):
            values = q[np.asarray(counts) == count]
            np.testing.assert_array_equal(values, np.full(values.size, values[0]))
        if 0 < result.diagnostics["distinct"] < len(counts):
            assert 0 < result.diagnostics["discount"] <= 0.9


def test_ad_finite_case_allocates_the_reserved_mass():
    result = absolute_discounting([3, 2, 1, 0, 0])
    delta = 1 / 3
    expected = np.array([(3 - delta) / 6, (2 - delta) / 6,
                         (1 - delta) / 6, delta * 3 / 12, delta * 3 / 12])
    np.testing.assert_allclose(result.probabilities, expected)
    assert expected.sum() == pytest.approx(1)


def test_oracle_is_class_average_and_lower_bound():
    counts = np.array([0, 0, 1, 1, 3])
    p = np.array([0.03, 0.17, 0.25, 0.15, 0.4])
    q = natural_oracle(counts, p)
    np.testing.assert_allclose(q, [0.1, 0.1, 0.2, 0.2, 0.4])
    for alternative in (add_constant(counts, 1), add_constant(counts, 0.5),
                        good_turing_hybrid(counts), ristad(counts)):
        assert kl_bits(p, q) <= kl_bits(p, alternative) + 1e-14


@pytest.mark.parametrize("counts", [[-1, 0], [0.5, 1], [], [[1]], [float("nan")]])
def test_invalid_counts_fail(counts):
    with pytest.raises(ValueError):
        validate_counts(counts)
    with pytest.raises(ValueError):
        absolute_discounting(counts)


def test_kl_rejects_missing_normalization_and_does_not_clip_zeros():
    with pytest.raises(ValueError):
        kl_bits([0.5, 0.5], [0.3, 0.3])
    assert math.isinf(kl_bits([0.5, 0.5], [1, 0]))
    assert kl_bits([0, 1], [0, 1]) == 0
