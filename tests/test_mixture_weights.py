import math

import numpy as np
import pytest

from lsa import (
    build_product_moment_tables,
    build_selected_product_moment_tables,
    compute_log_q_by_partition,
    log_q_lambda_closed_l1,
    log_q_lambda_grid,
    log_q_lambda_laplace,
)


def test_product_moment_table_matches_l1_closed_form() -> None:
    u_grid = np.linspace(-5.0, 5.0, 501)
    tables = build_product_moment_tables(
        max_L=1,
        max_r=4,
        u_grid=u_grid,
        laguerre_order=32,
    )

    for r in range(5):
        for u in (-3.5, -0.24, 2.74):
            expected = math.lgamma(r + 1) - (r + 1) * np.logaddexp(0.0, u)
            assert tables.log_phi_value(L=1, r=r, u=u) == pytest.approx(
                expected,
                abs=1e-12,
            )


def test_log_q_lambda_closed_l1_matches_dirichlet_moment() -> None:
    result = log_q_lambda_closed_l1(d=7, partition=(2, 1))

    expected = math.log(2.0) - math.log(7.0 * 8.0 * 9.0)
    assert result.log_q == pytest.approx(expected)
    assert result.method == "closed_l1"


def test_compute_log_q_n_equals_one_uses_exchangeability() -> None:
    assert compute_log_q_by_partition(d=123, L=17, N=1) == {
        (1,): pytest.approx(-math.log(123.0))
    }


def test_laplace_q_lambda_tracks_grid_integral_for_finite_l() -> None:
    u_grid = np.linspace(-40.0, 25.0, 4001)
    tables = build_product_moment_tables(
        max_L=3,
        max_r=4,
        u_grid=u_grid,
        laguerre_order=64,
        chunk_size=256,
    )

    grid = log_q_lambda_grid(d=20, L=3, partition=(2, 1), tables=tables)
    laplace = log_q_lambda_laplace(d=20, L=3, partition=(2, 1), tables=tables)

    assert grid.converged
    assert laplace.converged
    assert laplace.curvature is not None
    assert laplace.curvature < 0.0
    assert laplace.log_q == pytest.approx(grid.log_q, abs=0.05)


def test_compute_log_q_by_partition_laplace_returns_all_partitions() -> None:
    u_grid = np.linspace(-40.0, 25.0, 4001)
    tables = build_product_moment_tables(
        max_L=2,
        max_r=5,
        u_grid=u_grid,
        laguerre_order=48,
        chunk_size=256,
    )

    log_q = compute_log_q_by_partition(
        d=30,
        L=2,
        N=3,
        method="laplace",
        tables=tables,
    )

    assert set(log_q) == {(3,), (2, 1), (1, 1, 1)}
    assert all(math.isfinite(value) for value in log_q.values())


def test_selected_product_moment_table_matches_full_table() -> None:
    u_grid = np.linspace(-20.0, 10.0, 1001)
    full = build_product_moment_tables(
        max_L=3,
        max_r=8,
        u_grid=u_grid,
        laguerre_order=48,
        chunk_size=256,
    )
    selected = build_selected_product_moment_tables(
        max_L=3,
        r_values=[0, 2, 5, 8],
        u_grid=u_grid,
        laguerre_order=48,
        chunk_size=256,
    )

    assert selected.r_values == (0, 2, 5, 8)
    for r in selected.r_values:
        assert selected.log_phi_value(L=3, r=r, u=-1.7) == pytest.approx(
            full.log_phi_value(L=3, r=r, u=-1.7)
        )


def test_laplace_q_lambda_works_with_sparse_moment_table() -> None:
    u_grid = np.linspace(-40.0, 25.0, 4001)
    tables = build_selected_product_moment_tables(
        max_L=3,
        r_values=[0, 1, 2, 4, 5, 6],
        u_grid=u_grid,
        laguerre_order=64,
        chunk_size=256,
    )

    result = log_q_lambda_laplace(d=50, L=3, partition=(4,), tables=tables)

    assert result.converged
    assert math.isfinite(result.log_q)
