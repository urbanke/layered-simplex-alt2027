import numpy as np
import pytest

from lsa import largest_entries_first


def test_largest_entries_first_moves_top_k_and_preserves_remainder_order() -> None:
    probabilities = np.array([0.10, 0.40, 0.05, 0.30, 0.15])

    transformed = largest_entries_first(probabilities, k=2)

    assert transformed == pytest.approx([0.40, 0.30, 0.10, 0.05, 0.15])
    assert transformed.sum() == pytest.approx(1.0)


def test_largest_entries_first_with_k_zero_returns_copy_in_same_order() -> None:
    probabilities = np.array([0.20, 0.10, 0.70])

    transformed = largest_entries_first(probabilities, k=0)

    assert transformed == pytest.approx(probabilities)
    assert transformed is not probabilities


def test_largest_entries_first_with_k_equal_dimension_sorts_all_entries() -> None:
    probabilities = np.array([0.20, 0.10, 0.70])

    transformed = largest_entries_first(probabilities, k=3)

    assert transformed == pytest.approx([0.70, 0.20, 0.10])


def test_largest_entries_first_breaks_ties_by_original_index() -> None:
    probabilities = np.array([0.25, 0.10, 0.25, 0.15, 0.25])

    transformed = largest_entries_first(probabilities, k=2)

    assert transformed == pytest.approx([0.25, 0.25, 0.10, 0.15, 0.25])


def test_largest_entries_first_rejects_invalid_k() -> None:
    probabilities = np.array([0.20, 0.30, 0.50])

    with pytest.raises(ValueError):
        largest_entries_first(probabilities, k=4)

