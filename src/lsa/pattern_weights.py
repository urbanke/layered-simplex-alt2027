from __future__ import annotations

import math
import operator
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq, least_squares


FloatArray = NDArray[np.float64]


@dataclass
class SaddlepointApproximation:
    """Gaussian saddlepoint approximation for one profile weight A_lambda."""

    log_weight: float
    eta: FloatArray
    covariance: FloatArray
    parts: tuple[int, ...]
    multiplicities: tuple[int, ...]
    residual_norm: float
    converged: bool
    message: str

    @property
    def log2_weight(self) -> float:
        return self.log_weight / math.log(2.0)


@dataclass
class HybridSaddlepointApproximation:
    """Top-atoms-exact plus tail-saddlepoint approximation for A_lambda."""

    log_weight: float
    top_count: int
    top_mass: float
    parts: tuple[int, ...]
    multiplicities: tuple[int, ...]
    terms_considered: int
    finite_terms: int
    tail_saddlepoints: int
    failed_tail_saddlepoints: int
    max_tail_residual_norm: float
    dominant_consumed_multiplicities: tuple[int, ...]
    converged: bool
    message: str

    @property
    def log2_weight(self) -> float:
        return self.log_weight / math.log(2.0)


@dataclass
class _CoefficientSaddlepointApproximation:
    log_coefficient: float
    eta: FloatArray
    covariance: FloatArray
    parts: tuple[int, ...]
    multiplicities: tuple[int, ...]
    residual_norm: float
    converged: bool
    message: str


def partition_multiplicities(partition: Iterable[int]) -> tuple[tuple[int, int], ...]:
    """Return ``(r, c_r)`` pairs for a positive integer partition."""

    try:
        parts = tuple(operator.index(part) for part in partition)
    except TypeError as exc:
        raise ValueError("partition entries must be integers") from exc
    if any(part <= 0 for part in parts):
        raise ValueError("partition entries must be positive integers")
    return tuple(sorted(Counter(parts).items()))


def log_a_lambda_saddlepoint(
    p: ArrayLike,
    partition: Iterable[int],
    *,
    chunk_size: int = 65_536,
    tolerance: float = 1e-10,
    max_nfev: int = 200,
) -> SaddlepointApproximation:
    """Approximate ``log A_lambda(p)`` by a multivariate saddlepoint formula.

    The returned logarithm is a natural logarithm.  If
    ``lambda = 1^{c_1} 2^{c_2} ...``, then ``A_lambda(p)`` is the probability
    that a multinomial count vector with probabilities ``p`` has this unordered
    nonzero-count profile.
    """

    p_array = np.asarray(p, dtype=np.float64)
    _validate_probability_vector(p_array)

    multiplicity_pairs = partition_multiplicities(partition)
    coefficient = _log_profile_coefficient_saddlepoint(
        weights=p_array,
        multiplicity_pairs=multiplicity_pairs,
        chunk_size=chunk_size,
        tolerance=tolerance,
        max_nfev=max_nfev,
    )
    total_count = sum(part * count for part, count in multiplicity_pairs)
    if math.isfinite(coefficient.log_coefficient):
        log_weight = math.lgamma(total_count + 1) + coefficient.log_coefficient
    else:
        log_weight = -math.inf

    return SaddlepointApproximation(
        log_weight=float(log_weight),
        eta=coefficient.eta,
        covariance=coefficient.covariance,
        parts=coefficient.parts,
        multiplicities=coefficient.multiplicities,
        residual_norm=coefficient.residual_norm,
        converged=coefficient.converged,
        message=coefficient.message,
    )


def log_a_lambda_hybrid(
    p: ArrayLike,
    partition: Iterable[int],
    *,
    top_count: int | None = None,
    min_expected_count: float = 5.0,
    chunk_size: int = 65_536,
    tolerance: float = 1e-10,
    max_nfev: int = 200,
    max_top_states: int = 200_000,
) -> HybridSaddlepointApproximation:
    """Approximate ``log A_lambda(p)`` with top atoms exact and tail by saddlepoint.

    If ``top_count`` is omitted, the routine treats every atom with
    ``N * p_i >= min_expected_count`` exactly, where ``N`` is the total count of
    the partition.  The remaining labels are represented by the same Gaussian
    saddlepoint coefficient approximation used by
    :func:`log_a_lambda_saddlepoint`.
    """

    p_array = np.asarray(p, dtype=np.float64)
    _validate_probability_vector(p_array)
    multiplicity_pairs = partition_multiplicities(partition)
    if not multiplicity_pairs:
        return HybridSaddlepointApproximation(
            log_weight=0.0,
            top_count=0,
            top_mass=0.0,
            parts=(),
            multiplicities=(),
            terms_considered=1,
            finite_terms=1,
            tail_saddlepoints=0,
            failed_tail_saddlepoints=0,
            max_tail_residual_norm=0.0,
            dominant_consumed_multiplicities=(),
            converged=True,
            message="empty partition",
        )

    parts_tuple = tuple(part for part, _ in multiplicity_pairs)
    counts_tuple = tuple(count for _, count in multiplicity_pairs)
    total_count = sum(part * count for part, count in multiplicity_pairs)
    if top_count is None:
        if min_expected_count < 0.0:
            raise ValueError("min_expected_count must be non-negative")
        top_count = int(np.count_nonzero(total_count * p_array >= min_expected_count))
    if top_count < 0:
        raise ValueError("top_count must be non-negative")

    top_weights, tail_weights = _split_top_tail_weights(p_array, top_count)
    if sum(counts_tuple) > top_weights.size + tail_weights.size:
        return HybridSaddlepointApproximation(
            log_weight=-math.inf,
            parts=parts_tuple,
            multiplicities=counts_tuple,
            top_count=int(top_weights.size),
            top_mass=float(np.sum(top_weights, dtype=np.float64)),
            terms_considered=0,
            finite_terms=0,
            tail_saddlepoints=0,
            failed_tail_saddlepoints=0,
            max_tail_residual_norm=math.inf,
            dominant_consumed_multiplicities=(),
            converged=False,
            message="profile uses more symbols than the support of p",
        )

    top_coefficients = _exact_top_profile_coefficients(
        top_weights=top_weights,
        parts=parts_tuple,
        multiplicities=counts_tuple,
        max_states=max_top_states,
    )
    tail_cache: dict[tuple[int, ...], _CoefficientSaddlepointApproximation] = {}
    log_terms = []
    dominant_state = ()
    dominant_log_term = -math.inf
    finite_terms = 0
    failed_tail = 0
    max_tail_residual = 0.0

    for consumed, log_top_coefficient in top_coefficients.items():
        remaining = tuple(
            count - used for count, used in zip(counts_tuple, consumed, strict=True)
        )
        tail_approximation = tail_cache.get(remaining)
        if tail_approximation is None:
            remaining_pairs = tuple(
                (part, count)
                for part, count in zip(parts_tuple, remaining, strict=True)
                if count > 0
            )
            tail_approximation = _log_profile_coefficient_saddlepoint(
                weights=tail_weights,
                multiplicity_pairs=remaining_pairs,
                chunk_size=chunk_size,
                tolerance=tolerance,
                max_nfev=max_nfev,
            )
            tail_cache[remaining] = tail_approximation
            if remaining_pairs and math.isfinite(tail_approximation.log_coefficient):
                max_tail_residual = max(
                    max_tail_residual,
                    tail_approximation.residual_norm,
                )
                if not tail_approximation.converged:
                    failed_tail += 1

        if not math.isfinite(tail_approximation.log_coefficient):
            continue
        log_term = log_top_coefficient + tail_approximation.log_coefficient
        log_terms.append(log_term)
        finite_terms += 1
        if log_term > dominant_log_term:
            dominant_log_term = log_term
            dominant_state = consumed

    log_coefficient = _logsumexp_array(log_terms)
    log_weight = (
        math.lgamma(total_count + 1) + log_coefficient
        if math.isfinite(log_coefficient)
        else -math.inf
    )

    message = (
        f"top_count={top_weights.size}, top_mass={np.sum(top_weights):.6g}, "
        f"tail_profiles={len(tail_cache)}"
    )
    return HybridSaddlepointApproximation(
        log_weight=float(log_weight),
        top_count=int(top_weights.size),
        top_mass=float(np.sum(top_weights, dtype=np.float64)),
        parts=parts_tuple,
        multiplicities=counts_tuple,
        terms_considered=len(top_coefficients),
        finite_terms=finite_terms,
        tail_saddlepoints=sum(
            1
            for key, approximation in tail_cache.items()
            if any(count > 0 for count in key)
            and math.isfinite(approximation.log_coefficient)
        ),
        failed_tail_saddlepoints=failed_tail,
        max_tail_residual_norm=max_tail_residual,
        dominant_consumed_multiplicities=dominant_state,
        converged=finite_terms > 0 and failed_tail == 0,
        message=message,
    )


def _log_profile_coefficient_saddlepoint(
    *,
    weights: FloatArray,
    multiplicity_pairs: tuple[tuple[int, int], ...],
    chunk_size: int,
    tolerance: float,
    max_nfev: int,
) -> _CoefficientSaddlepointApproximation:
    if not multiplicity_pairs:
        return _CoefficientSaddlepointApproximation(
            log_coefficient=0.0,
            eta=np.empty(0, dtype=np.float64),
            covariance=np.empty((0, 0), dtype=np.float64),
            parts=(),
            multiplicities=(),
            residual_norm=0.0,
            converged=True,
            message="empty profile coefficient",
        )

    parts = np.array([part for part, _ in multiplicity_pairs], dtype=np.float64)
    counts = np.array([count for _, count in multiplicity_pairs], dtype=np.float64)
    parts_tuple = tuple(int(part) for part in parts)
    counts_tuple = tuple(int(count) for count in counts)
    observed_symbols = int(np.sum(counts))

    positive_weights = weights[weights > 0.0]
    if observed_symbols > positive_weights.size:
        k = len(parts_tuple)
        return _CoefficientSaddlepointApproximation(
            log_coefficient=-math.inf,
            eta=np.full(k, math.nan, dtype=np.float64),
            covariance=np.full((k, k), math.nan, dtype=np.float64),
            parts=parts_tuple,
            multiplicities=counts_tuple,
            residual_norm=math.inf,
            converged=False,
            message="profile uses more symbols than the support of weights",
        )

    log_p = np.log(positive_weights)
    log_factorials = np.array(
        [math.lgamma(int(part) + 1) for part in parts_tuple],
        dtype=np.float64,
    )
    eta0 = _initial_eta(log_p, parts, counts, log_factorials)
    scale = np.sqrt(np.maximum(counts, 1.0))
    if len(parts_tuple) == 1:
        eta = np.array(
            [
                _solve_single_saddlepoint(
                    eta0=float(eta0[0]),
                    log_p=log_p,
                    part=float(parts[0]),
                    log_factorial=float(log_factorials[0]),
                    count=float(counts[0]),
                )
            ],
            dtype=np.float64,
        )
        k_value, means, covariance = _evaluate_profile_cumulants(
            eta=eta,
            log_p=log_p,
            parts=parts,
            log_factorials=log_factorials,
            chunk_size=chunk_size,
            compute_covariance=True,
        )
        final_residual = (means - counts) / scale
        residual_norm = float(np.linalg.norm(final_residual, ord=np.inf))
        logdet = _positive_logdet(covariance)
        log_weight = (
            k_value
            - float(np.dot(counts, eta))
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * logdet
        )
        return _CoefficientSaddlepointApproximation(
            log_coefficient=float(log_weight),
            eta=eta,
            covariance=covariance,
            parts=parts_tuple,
            multiplicities=counts_tuple,
            residual_norm=residual_norm,
            converged=residual_norm <= math.sqrt(tolerance),
            message="brentq one-dimensional saddlepoint solve",
        )

    def residual(eta: FloatArray) -> FloatArray:
        _, means, _ = _evaluate_profile_cumulants(
            eta=eta,
            log_p=log_p,
            parts=parts,
            log_factorials=log_factorials,
            chunk_size=chunk_size,
            compute_covariance=False,
        )
        return (means - counts) / scale

    def jacobian(eta: FloatArray) -> FloatArray:
        _, _, covariance = _evaluate_profile_cumulants(
            eta=eta,
            log_p=log_p,
            parts=parts,
            log_factorials=log_factorials,
            chunk_size=chunk_size,
            compute_covariance=True,
        )
        return covariance / scale[:, None]

    optimization = least_squares(
        residual,
        eta0,
        jac=jacobian,
        xtol=tolerance,
        ftol=tolerance,
        gtol=tolerance,
        max_nfev=max_nfev,
    )

    eta = np.asarray(optimization.x, dtype=np.float64)
    k_value, means, covariance = _evaluate_profile_cumulants(
        eta=eta,
        log_p=log_p,
        parts=parts,
        log_factorials=log_factorials,
        chunk_size=chunk_size,
        compute_covariance=True,
    )
    final_residual = (means - counts) / scale
    residual_norm = float(np.linalg.norm(final_residual, ord=np.inf))
    logdet = _positive_logdet(covariance)
    log_weight = (
        k_value
        - float(np.dot(counts, eta))
        - 0.5 * len(parts_tuple) * math.log(2.0 * math.pi)
        - 0.5 * logdet
    )
    converged = bool(optimization.success and residual_norm <= math.sqrt(tolerance))

    return _CoefficientSaddlepointApproximation(
        log_coefficient=float(log_weight),
        eta=eta,
        covariance=covariance,
        parts=parts_tuple,
        multiplicities=counts_tuple,
        residual_norm=residual_norm,
        converged=converged,
        message=str(optimization.message),
    )


def _split_top_tail_weights(
    p: FloatArray,
    top_count: int,
) -> tuple[FloatArray, FloatArray]:
    positive = p[p > 0.0]
    if top_count <= 0:
        return np.empty(0, dtype=np.float64), positive
    if top_count >= positive.size:
        return positive.copy(), np.empty(0, dtype=np.float64)

    partition_index = positive.size - top_count
    split = np.argpartition(positive, partition_index)
    tail = positive[split[:partition_index]]
    top = positive[split[partition_index:]]
    return top, tail


def _exact_top_profile_coefficients(
    *,
    top_weights: FloatArray,
    parts: tuple[int, ...],
    multiplicities: tuple[int, ...],
    max_states: int,
) -> dict[tuple[int, ...], float]:
    zero_state = tuple(0 for _ in parts)
    states: dict[tuple[int, ...], float] = {zero_state: 0.0}
    if top_weights.size == 0:
        return states

    log_factorials = tuple(math.lgamma(part + 1) for part in parts)
    for weight in top_weights:
        log_weight = math.log(float(weight))
        log_part_terms = tuple(
            part * log_weight - log_factorial
            for part, log_factorial in zip(parts, log_factorials, strict=True)
        )
        next_states = dict(states)
        for state, log_coefficient in states.items():
            for index, log_part_term in enumerate(log_part_terms):
                if state[index] >= multiplicities[index]:
                    continue
                next_state_list = list(state)
                next_state_list[index] += 1
                next_state = tuple(next_state_list)
                next_states[next_state] = _logaddexp(
                    next_states.get(next_state, -math.inf),
                    log_coefficient + log_part_term,
                )
        if len(next_states) > max_states:
            raise RuntimeError(
                "top-atom exact dynamic program exceeded max_top_states="
                f"{max_states}"
            )
        states = next_states
    return states


def _validate_probability_vector(p: FloatArray, atol: float = 1e-10) -> None:
    if p.ndim != 1:
        raise ValueError("p must be one-dimensional")
    if p.size == 0:
        raise ValueError("p must not be empty")
    if not np.all(np.isfinite(p)):
        raise ValueError("p must be finite")
    if np.any(p < 0.0):
        raise ValueError("p must be non-negative")
    if not np.isclose(p.sum(dtype=np.float64), 1.0, atol=atol):
        raise ValueError("p must sum to 1")


def _initial_eta(
    log_p: FloatArray,
    parts: FloatArray,
    counts: FloatArray,
    log_factorials: FloatArray,
) -> FloatArray:
    eta = np.empty_like(parts)
    for index, (part, count, log_factorial) in enumerate(
        zip(parts, counts, log_factorials, strict=True)
    ):
        log_sum_a = _logsumexp(part * log_p - log_factorial)
        if not math.isfinite(log_sum_a):
            raise ValueError("partition contains a part impossible under p")
        eta[index] = math.log(float(count)) - log_sum_a
    return eta


def _evaluate_profile_cumulants(
    *,
    eta: FloatArray,
    log_p: FloatArray,
    parts: FloatArray,
    log_factorials: FloatArray,
    chunk_size: int,
    compute_covariance: bool,
) -> tuple[float, FloatArray, FloatArray]:
    k = eta.size
    cumulant = 0.0
    means = np.zeros(k, dtype=np.float64)
    second_moments = np.zeros((k, k), dtype=np.float64)

    for start in range(0, log_p.size, chunk_size):
        stop = min(start + chunk_size, log_p.size)
        log_terms = (
            log_p[start:stop, None] * parts[None, :]
            - log_factorials[None, :]
            + eta[None, :]
        )
        max_terms = np.maximum(0.0, np.max(log_terms, axis=1))
        shifted_sum = np.exp(-max_terms) + np.sum(
            np.exp(log_terms - max_terms[:, None]),
            axis=1,
            dtype=np.float64,
        )
        log_denominator = max_terms + np.log(shifted_sum)
        weights = np.exp(log_terms - log_denominator[:, None])

        cumulant += float(np.sum(log_denominator, dtype=np.float64))
        means += np.sum(weights, axis=0, dtype=np.float64)
        if compute_covariance:
            second_moments += weights.T @ weights

    if compute_covariance:
        covariance = np.diag(means) - second_moments
        covariance = 0.5 * (covariance + covariance.T)
    else:
        covariance = np.empty((k, k), dtype=np.float64)
    return cumulant, means, covariance


def _solve_single_saddlepoint(
    *,
    eta0: float,
    log_p: FloatArray,
    part: float,
    log_factorial: float,
    count: float,
) -> float:
    if count >= log_p.size:
        raise RuntimeError("single-variable saddlepoint is on the boundary")

    log_a = part * log_p - log_factorial

    def objective(eta: float) -> float:
        log_terms = eta + log_a
        max_terms = np.maximum(0.0, log_terms)
        log_denominator = max_terms + np.log(
            np.exp(-max_terms) + np.exp(log_terms - max_terms)
        )
        return float(np.sum(np.exp(log_terms - log_denominator))) - count

    lower = eta0
    upper = eta0
    lower_value = objective(lower)
    upper_value = lower_value
    for _ in range(200):
        if lower_value < 0.0:
            break
        lower -= 1.0
        lower_value = objective(lower)
    for _ in range(200):
        if upper_value > 0.0:
            break
        upper += 1.0
        upper_value = objective(upper)
    if lower_value >= 0.0 or upper_value <= 0.0:
        raise RuntimeError("could not bracket one-dimensional saddlepoint")
    return float(brentq(objective, lower, upper, xtol=1e-12, rtol=1e-12))


def _positive_logdet(matrix: FloatArray) -> float:
    eigenvalues = np.linalg.eigvalsh(matrix)
    largest = max(1.0, float(np.max(np.abs(eigenvalues))))
    smallest = float(np.min(eigenvalues))
    if smallest <= 0.0:
        if smallest < -1e-10 * largest:
            raise RuntimeError("saddlepoint covariance is not positive definite")
        eigenvalues = np.clip(eigenvalues, np.finfo(np.float64).tiny, None)
    return float(np.sum(np.log(eigenvalues), dtype=np.float64))


def _logsumexp(values: FloatArray) -> float:
    max_value = float(np.max(values))
    if not math.isfinite(max_value):
        return -math.inf
    return max_value + math.log(float(np.sum(np.exp(values - max_value))))


def _logsumexp_array(values: list[float]) -> float:
    if not values:
        return -math.inf
    return _logsumexp(np.array(values, dtype=np.float64))


def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))
