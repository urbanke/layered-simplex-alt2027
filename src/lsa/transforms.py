from __future__ import annotations

import operator

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def largest_entries_first(
    probabilities: ArrayLike,
    k: int,
    *,
    validate: bool = True,
) -> FloatArray:
    """Return a copy with the largest ``k`` entries moved to the front.

    The first ``k`` entries of the output are the largest entries of
    ``probabilities``, sorted largest to smallest. The remaining entries are all
    non-selected entries in their original relative order.

    Ties are broken by original index: earlier entries are selected first and
    appear first among equal values.
    """

    vector = np.asarray(probabilities, dtype=np.float64)
    _validate_vector_shape(vector)
    if validate:
        _validate_probability_vector(vector)

    k = _validate_k(k, vector.size)
    if k == 0:
        return vector.copy()

    if k == vector.size:
        return vector[np.argsort(-vector, kind="stable")]

    threshold = np.partition(vector, vector.size - k)[vector.size - k]

    greater_indices = np.flatnonzero(vector > threshold)
    selected_count = greater_indices.size
    equal_needed = k - selected_count

    if equal_needed > 0:
        equal_indices = np.flatnonzero(vector == threshold)[:equal_needed]
        selected_indices = np.concatenate((greater_indices, equal_indices))
    else:
        selected_indices = greater_indices

    top_order = np.argsort(-vector[selected_indices], kind="stable")
    top_indices = selected_indices[top_order]

    selected_mask = np.zeros(vector.size, dtype=np.bool_)
    selected_mask[selected_indices] = True

    out = np.empty_like(vector)
    out[:k] = vector[top_indices]
    out[k:] = vector[~selected_mask]
    return out


def _validate_vector_shape(vector: FloatArray) -> None:
    if vector.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if vector.size == 0:
        raise ValueError("probabilities must not be empty")


def _validate_probability_vector(vector: FloatArray, atol: float = 1e-10) -> None:
    if not np.all(np.isfinite(vector)):
        raise ValueError("probabilities must be finite")
    if np.any(vector < 0.0):
        raise ValueError("probabilities must be non-negative")
    if not np.isclose(vector.sum(dtype=np.float64), 1.0, atol=atol):
        raise ValueError("probabilities must sum to 1")


def _validate_k(k: int, d: int) -> int:
    try:
        k = operator.index(k)
    except TypeError as exc:
        raise TypeError("k must be an integer") from exc

    if k < 0:
        raise ValueError("k must be non-negative")
    if k > d:
        raise ValueError("k must be at most the vector length")
    return k

