from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
LOG_2 = np.log(2.0)


def entropy(
    p: ArrayLike,
    *,
    validate: bool = True,
) -> float:
    """Return the Shannon entropy of ``p`` in bits."""

    p_array = np.asarray(p, dtype=np.float64)
    _validate_probability_vector_shape(p_array, "p")
    if validate:
        _validate_probability_vector(p_array, "p")

    positive_p = p_array > 0.0
    return float(
        -np.sum(
            p_array[positive_p] * np.log(p_array[positive_p]),
            dtype=np.float64,
        )
        / LOG_2
    )


def kl_divergence(
    p: ArrayLike,
    q: ArrayLike,
    *,
    validate: bool = True,
) -> float:
    """Return the Kullback-Leibler divergence ``D(p || q)``.

    The value is measured in bits. The conventions are:

    - ``0 * log2(0 / q_i) = 0``
    - if ``p_i > 0`` and ``q_i = 0``, then ``D(p || q) = inf``
    """

    p_array = np.asarray(p, dtype=np.float64)
    q_array = np.asarray(q, dtype=np.float64)
    _validate_same_shape(p_array, q_array)

    if validate:
        _validate_probability_vector(p_array, "p")
        _validate_probability_vector(q_array, "q")

    positive_p = p_array > 0.0
    if np.any(q_array[positive_p] == 0.0):
        return float("inf")

    return float(
        np.sum(
            p_array[positive_p] * np.log(p_array[positive_p] / q_array[positive_p]),
            dtype=np.float64,
        )
        / LOG_2
    )


def _validate_same_shape(p: FloatArray, q: FloatArray) -> None:
    _validate_probability_vector_shape(p, "p")
    _validate_probability_vector_shape(q, "q")
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")


def _validate_probability_vector_shape(vector: FloatArray, name: str) -> None:
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")


def _validate_probability_vector(vector: FloatArray, name: str, atol: float = 1e-10) -> None:
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if not np.isclose(vector.sum(dtype=np.float64), 1.0, atol=atol):
        raise ValueError(f"{name} must sum to 1")
