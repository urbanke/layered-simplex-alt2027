"""Tests for the Section-5 estimators.

The important identities: every assignment normalizes; Ristad's closed
form is exactly normalized and collapses to add-one on a saturated
alphabet; the Good--Turing hybrid follows its stated branch rule; the
natural oracle is the per-instance optimum among count-based rules; the
LSA predictive at L = 1 is exactly add-one (Proposition 1); predictives
sum to one at every depth (the Appendix-C normalization check); and the
sequential coders match their batch counterparts (chain rule).
"""

import math
import tempfile

import numpy as np
import pytest

from lsa.estimators import (
    add_constant_codelength_bits,
    add_half,
    add_one,
    braess_sauer,
    good_turing_hybrid,
    lsa_predictive_by_count,
    natural_oracle,
    ristad_natural_law,
    sequential_codelength_bits,
)

RNG = np.random.default_rng(20260819)


def random_counts(d: int, n: int) -> np.ndarray:
    p = RNG.dirichlet(np.full(d, 0.3))
    return RNG.multinomial(n, p)


@pytest.mark.parametrize("rule", [
    add_one, add_half, braess_sauer, good_turing_hybrid, ristad_natural_law,
])
def test_batch_rules_normalize(rule):
    for d, n in [(5, 0), (5, 3), (50, 200), (400, 37)]:
        counts = random_counts(d, n)
        q = rule(counts)
        assert q.shape == (d,)
        assert np.all(q > 0)
        assert abs(q.sum() - 1.0) < 1e-12


def test_ristad_matches_add_one_when_saturated():
    counts = np.array([3, 1, 2, 5])
    assert np.allclose(ristad_natural_law(counts), add_one(counts))


def test_good_turing_hybrid_hand_example():
    # counts (3, 1, 1, 0, 0): n = 5, c_0 = 2, c_1 = 2, c_3 = 1.
    # t = 3: c_4 = 0 < 3        -> empirical 3/5
    # t = 1: c_2 = 0 < 1        -> empirical 1/5
    # t = 0: GT (c_1 + 1)/(5*2) -> 3/10
    counts = np.array([3, 1, 1, 0, 0])
    w = np.array([3 / 5, 1 / 5, 1 / 5, 3 / 10, 3 / 10])
    assert np.allclose(good_turing_hybrid(counts), w / w.sum())


def test_natural_oracle_is_the_count_based_optimum():
    d, n = 60, 150
    p = RNG.dirichlet(np.full(d, 0.2))
    counts = RNG.multinomial(n, p)

    def kl(q):
        return float(np.sum(p * np.log2(p / q)))

    oracle = kl(natural_oracle(counts, p))
    for rule in (add_one, add_half, braess_sauer, good_turing_hybrid,
                 ristad_natural_law):
        assert oracle <= kl(rule(counts)) + 1e-12


def test_lsa_depth_one_is_add_one():
    d = 7
    counts = np.array([2, 1, 1, 0, 0, 0, 0])
    n = counts.sum()
    with tempfile.TemporaryDirectory() as cache:
        pred = lsa_predictive_by_count(counts, d=d, l_max=3, cache_dir=cache)
    for c in (0, 1, 2):
        assert abs(pred.by_count[1][c] - (c + 1) / (n + d)) < 1e-9


def test_lsa_predictives_normalize_at_every_depth():
    d = 12
    counts = np.array([5, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    with tempfile.TemporaryDirectory() as cache:
        pred = lsa_predictive_by_count(counts, d=d, l_max=6, cache_dir=cache)
    prevalence = {1: 1, 3: 2, 5: 1}   # count value -> how many symbols
    unseen = d - sum(prevalence.values())
    tables = [pred.by_count[L] for L in range(1, 7)] + [pred.avg_by_count]
    for table in tables:
        total = unseen * table[0] + sum(
            k * table[c] for c, k in prevalence.items()
        )
        assert abs(total - 1.0) < 5e-6
    q = pred.q_hat(counts)
    assert abs(q.sum() - 1.0) < 5e-6


def test_sequential_add_constant_matches_closed_form():
    d = 9
    ids = RNG.integers(0, d, size=200)
    counts = np.bincount(ids, minlength=d)
    for method, beta in [("add_one", 1.0), ("add_half", 0.5)]:
        seq = sequential_codelength_bits(ids, d, method)[len(ids)]
        batch = add_constant_codelength_bits(counts, beta, d=d)
        assert abs(seq - batch) < 1e-8 * max(1.0, batch)


def test_sequential_good_turing_matches_batch_recomputation():
    # the incrementally maintained normalizer must equal a from-scratch
    # batch evaluation of the hybrid at every step
    d = 15
    ids = RNG.integers(0, d, size=300)
    seq = sequential_codelength_bits(
        ids, d, "good_turing", checkpoints=range(1, len(ids) + 1)
    )
    counts = np.zeros(d, dtype=np.int64)
    bits = 0.0
    for step, x in enumerate(ids, start=1):
        q = 1.0 / d if step == 1 else good_turing_hybrid(counts)[x]
        bits -= math.log2(q)
        counts[x] += 1
        assert abs(seq[step] - bits) < 1e-9 * max(1.0, bits), step


def test_sequential_ristad_and_braess_sauer_match_batch_rules():
    d = 11
    ids = RNG.integers(0, d, size=250)
    for method, rule in [("ristad", ristad_natural_law),
                         ("braess_sauer", braess_sauer)]:
        seq = sequential_codelength_bits(
            ids, d, method, checkpoints=range(1, len(ids) + 1)
        )
        counts = np.zeros(d, dtype=np.int64)
        bits = 0.0
        for step, x in enumerate(ids, start=1):
            q = 1.0 / d if step == 1 else rule(counts)[x]
            bits -= math.log2(q)
            counts[x] += 1
            assert abs(seq[step] - bits) < 1e-9 * max(1.0, bits), (method, step)
