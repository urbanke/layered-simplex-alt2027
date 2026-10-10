#!/usr/bin/env python3
"""The validation checks of Appendix C, as one runnable gate.

Re-runs the implementation checks that the paper requires to pass before
any experiment (Appendix C, first table), plus the profile-weight
identities of Appendix A, and prints a pass/fail table:

1. layer recursion (Equation 16) at L = 2 against independent adaptive
   quadrature;
2. closed forms at L = 1 against Proposition 1 (relative);
3. exchangeability identities at the experiment scale: q_(1) = 1/d and
   d q_(2) + d(d-1) q_(1,1) = 1, at fixed depths;
4. predictive normalization sum_i q_hat(i) = 1 on a sampled profile;
5. sum over profiles A_lambda = 1 (exact enumeration at small d, N) and
   the uniform-target closed form of Appendix A.

    python scripts/validate_appendix_c.py            # experiment scale
    python scripts/validate_appendix_c.py --quick    # small-d variant

Exits nonzero if any check fails.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import tempfile
from collections import Counter

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln

os.environ.setdefault("LSA_NO_TRUNCATE", "1")

from lsa.codelength import depth_averaged_codelength  # noqa: E402
from lsa.estimators import lsa_predictive_by_count  # noqa: E402
from lsa.mellin import log_phi_closed_l1, log_phi_contour  # noqa: E402
from lsa.mixture_weights import log_q_lambda_closed_l1  # noqa: E402
from lsa.pattern_weights import log_a_lambda_hybrid  # noqa: E402

LOG2 = math.log(2.0)
CHECKS: list[tuple[str, str, float, float, bool]] = []


def record(name: str, reference: str, agreement: float,
           tolerance: float) -> None:
    CHECKS.append((name, reference, agreement, tolerance,
                   agreement < tolerance))


def check_layer_recursion() -> None:
    worst = 0.0
    for r in (0, 3, 11):
        for t in (0.5, 5.0, 200.0):
            def log_integrand(x: float, r=r, t=t) -> float:
                return (-x + r * math.log(x)
                        + log_phi_closed_l1(r, math.log(t * x)))
            # peak-normalize so the quadrature works in relative terms
            grid = np.exp(np.linspace(-30.0, 8.0, 4001))
            peak = float(max(log_integrand(x) for x in grid))
            reference, _ = quad(
                lambda x: math.exp(log_integrand(x) - peak) if x > 0 else 0.0,
                0.0, np.inf, limit=400,
            )
            value = log_phi_contour(r, 2, math.log(t), dispatch=False)
            worst = max(worst, abs(value - (math.log(reference) + peak)))
    record("layer recursion (16), L = 2", "adaptive quadrature", worst, 2e-6)


def check_l1_closed_form() -> None:
    worst = 0.0
    for d, partition in [(7, (3, 2, 1)), (1000, (10, 5, 5, 1)),
                         (10_000, (40, 3, 1, 1, 1))]:
        n = sum(partition)
        value = log_q_lambda_closed_l1(d=d, partition=partition).log2_q
        reference = (
            gammaln(d) - gammaln(d + n)
            + sum(gammaln(m + 1) for m in partition)
        ) / LOG2
        worst = max(worst, abs(value - reference) / abs(reference))
    record("closed forms at L = 1", "Proposition 1", worst, 2e-9)


def check_exchangeability(d: int, depths: tuple[int, ...],
                          tolerance: float) -> None:
    l_max = max(depths)
    with tempfile.TemporaryDirectory() as cache:
        one = depth_averaged_codelength((1,), d=d, l_max=l_max,
                                        cache_dir=cache)
        two = depth_averaged_codelength((2,), d=d, l_max=l_max,
                                        cache_dir=cache)
        pair = depth_averaged_codelength((1, 1), d=d, l_max=l_max,
                                         cache_dir=cache)
    worst = 0.0
    for L in depths:
        q1 = 2.0 ** one.log2_q_by_depth[L - 1]
        worst = max(worst, abs(math.log(q1 * d)))
        q2 = 2.0 ** two.log2_q_by_depth[L - 1]
        q11 = 2.0 ** pair.log2_q_by_depth[L - 1]
        worst = max(worst, abs(math.log(d * q2 + d * (d - 1) * q11)))
    record(f"q_(1) d = 1/d; d q_(2) + d(d-1) q_(1,1) = 1 (d = {d:g})",
           "exchangeability", worst, tolerance)


def check_predictive_normalization(d: int) -> None:
    rng = np.random.default_rng(11)
    w = np.arange(1, d + 1, dtype=float) ** -1.5
    counts = rng.multinomial(1000, w / w.sum())
    prevalence = Counter(int(c) for c in counts[counts > 0])
    unseen = d - sum(prevalence.values())
    with tempfile.TemporaryDirectory() as cache:
        pred = lsa_predictive_by_count(counts, d=d, l_max=22,
                                       cache_dir=cache)
    for L, tolerance in ((5, 1e-5), (22, 4e-4)):
        table = pred.by_count[L]
        total = unseen * table[0] + sum(
            k * table[c] for c, k in prevalence.items()
        )
        record(f"predictive normalization at L = {L} (d = {d:g})",
               "sampled profile", abs(total - 1.0), tolerance)


def check_a_lambda() -> None:
    d, n = 5, 6
    rng_weights = np.arange(1, d + 1, dtype=float) ** -1.2
    p = rng_weights / rng_weights.sum()
    log_p = np.log(p)
    a_of_profile: dict[tuple[int, ...], float] = {}
    for split in itertools.combinations(range(n + d - 1), d - 1):
        cuts = (-1, *split, n + d - 1)
        m = np.array([cuts[i + 1] - cuts[i] - 1 for i in range(d)])
        profile = tuple(sorted((int(x) for x in m[m > 0]), reverse=True))
        log_pmf = (
            gammaln(n + 1) - sum(gammaln(int(x) + 1) for x in m)
            + float((m * log_p).sum())
        )
        a_of_profile[profile] = a_of_profile.get(profile, 0.0) + math.exp(
            log_pmf
        )
    record(f"sum over profiles A_lambda = 1 (d = {d}, N = {n})",
           "exact enumeration", abs(sum(a_of_profile.values()) - 1.0), 1e-12)

    a_uniform: dict[tuple[int, ...], float] = {}
    for profile, _ in a_of_profile.items():
        parts = list(profile)
        s = len(parts)
        multiplicity = Counter(parts)
        log_a = (
            gammaln(n + 1) - n * math.log(d)
            + gammaln(d + 1) - gammaln(d - s + 1)
            - sum(gammaln(r + 1) * c for r, c in multiplicity.items())
            - sum(gammaln(c + 1) for c in multiplicity.values())
        )
        a_uniform[profile] = math.exp(log_a)
    enumerated_uniform: dict[tuple[int, ...], float] = {}
    for split in itertools.combinations(range(n + d - 1), d - 1):
        cuts = (-1, *split, n + d - 1)
        m = np.array([cuts[i + 1] - cuts[i] - 1 for i in range(d)])
        profile = tuple(sorted((int(x) for x in m[m > 0]), reverse=True))
        log_pmf = gammaln(n + 1) - sum(
            gammaln(int(x) + 1) for x in m
        ) - n * math.log(d)
        enumerated_uniform[profile] = (
            enumerated_uniform.get(profile, 0.0) + math.exp(log_pmf)
        )
    worst = max(
        abs(a_uniform[prof] - enumerated_uniform[prof])
        for prof in enumerated_uniform
    )
    record("uniform-target closed form for A_lambda", "Appendix A",
           worst, 1e-12)

    worst = 0.0
    for profile in [(3, 2, 1), (6,), (2, 2, 1, 1)]:
        value = log_a_lambda_hybrid(p, profile, top_count=d).log_weight
        reference = math.log(a_of_profile[profile])
        worst = max(worst, abs(float(value) - reference))
    record("hybrid A_lambda evaluation (all atoms exact)",
           "exact enumeration", worst, 1e-8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="run the exchangeability and normalization "
                             "checks at d = 100 instead of d = 10^4")
    args = parser.parse_args()

    check_layer_recursion()
    check_l1_closed_form()
    if args.quick:
        check_exchangeability(100, (5, 12), 2e-5)
        check_predictive_normalization(100)
    else:
        check_exchangeability(10_000, (5, 22), 2e-5)
        check_predictive_normalization(10_000)
    check_a_lambda()

    print(f"{'check':<52} {'compared against':<20} "
          f"{'agreement':>11} {'tolerance':>10}  result")
    ok = True
    for name, reference, agreement, tolerance, passed in CHECKS:
        ok = ok and passed
        print(f"{name:<52} {reference:<20} {agreement:>11.2e} "
              f"{tolerance:>10.0e}  {'PASS' if passed else 'FAIL'}")
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
