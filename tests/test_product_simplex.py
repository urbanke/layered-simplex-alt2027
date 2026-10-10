import numpy as np
import pytest

from lsa import (
    ProductSimplexSampler,
    fill_product_simplex_sample,
    sample_product_simplex_distribution,
)


def test_sample_is_probability_vector() -> None:
    rng = np.random.default_rng(123)

    sample = sample_product_simplex_distribution(d=10_000, L=7, rng=rng)

    assert sample.shape == (10_000,)
    assert np.all(sample >= 0.0)
    assert sample.sum() == pytest.approx(1.0)


def test_reusable_sampler_can_fill_preallocated_output() -> None:
    sampler = ProductSimplexSampler(d=128, L=3, rng=np.random.default_rng(123))
    out = np.empty(128, dtype=np.float64)

    sample = sampler.sample(out=out)

    assert sample is out
    assert out.sum() == pytest.approx(1.0)


def test_l_equals_one_matches_normalized_exponentials_for_same_rng_seed() -> None:
    seed = 123
    product_sample = sample_product_simplex_distribution(
        d=256,
        L=1,
        rng=np.random.default_rng(seed),
    )

    uniforms = np.random.default_rng(seed).random(256)
    exponentials = -np.log(uniforms)
    direct_sample = exponentials / exponentials.sum()

    assert product_sample == pytest.approx(direct_sample)


def test_fill_rejects_shared_work_array() -> None:
    out = np.empty(8, dtype=np.float64)

    with pytest.raises(ValueError):
        fill_product_simplex_sample(out=out, L=2, work=out)

