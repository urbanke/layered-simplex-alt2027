"""Render the paper from saved numerical summaries; never evaluate models."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from paper_results import require
from scipy.optimize import brentq
from scipy.special import digamma


def validate_probabilities(weights, *, size):
    p = np.asarray(weights, dtype=float)
    require(
        p.shape == (size,) and np.all(np.isfinite(p)) and np.all(p >= 0),
        "invalid posterior",
    )
    require(abs(float(p.sum()) - 1) <= 1e-10, "posterior does not sum to one")
    return p


METHOD_LABELS = {
    "add_one": "add-1",
    "kt": "KT",
    "ristad": "Ristad",
    "good_turing_hybrid": "GT",
    "dirichlet_concentration_mixture": "Dir-$\\tau$",
    "absolute_discounting": "AD",
    "lsa_fixed": "LSA fixed",
    "lsa_depth_mixture": "LSA depth avg",
    "power_mixture": "power avg",
    "oracle": "natural oracle",
}


TARGET_LABELS = {
    "uniform": "uniform",
    "step": "step",
    "geometric": "geometric",
    "dirichlet_1": "Dirichlet-1",
    "dirichlet_half": r"Dirichlet-$\frac{1}{2}$",
    **{f"zipf_{a}": f"Zipf $\\alpha={a}$" for a in ("1", "1.5", "2", "3", "4", "5")},
}


def posterior_summary(
    indices: Sequence[int],
    weights: Sequence[float],
    *,
    mass: float,
    broad_min_depths: int,
    individual_min_weight: float,
) -> dict:
    """Shortest contiguous interval; ties prefer greater enclosed mass.

    Contiguous means adjacent integer model indices, not adjacent locations in
    a sparse plotting grid. The actual mass is retained without rounding.
    """
    indices = list(indices)
    p = validate_probabilities(weights, size=len(indices))
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("posterior summary requires a contiguous integer grid")
    if (
        not (0 < mass <= 1)
        or broad_min_depths < 1
        or not (0 <= individual_min_weight <= 1)
    ):
        raise ValueError("invalid posterior reporting rule")
    cumulative = np.r_[0.0, np.cumsum(p)]
    candidates = []
    for left in range(len(p)):
        for right in range(left, len(p)):
            enclosed = float(cumulative[right + 1] - cumulative[left])
            if enclosed >= mass - 1e-14:
                candidates.append((right - left + 1, -enclosed, left, right))
                break
    if not candidates:
        raise ArithmeticError("posterior has insufficient total mass")
    width, neg_mass, left, right = min(candidates)
    result = {
        "interval": [indices[left], indices[right]],
        "enclosed_mass": -neg_mass,
        "number_of_depths": width,
        "requested_mass": mass,
    }
    if width >= broad_min_depths:
        result.update(
            kind="interval",
            text=f"broad: {-neg_mass:.1%} on L in [{indices[left]},{indices[right]}]",
        )
    else:
        kept = [(i, float(w)) for i, w in zip(indices, p) if w >= individual_min_weight]
        result.update(
            kind="individual",
            weights=kept,
            text=", ".join(f"L={i}: {w:.2f}" for i, w in kept),
        )
    return result


def _ensure_cells(
    summary: Mapping[str, Any],
    targets: Sequence[str],
    n_values: Sequence[int],
    methods: Sequence[str],
) -> None:
    for target in targets:
        if target not in summary["targets"]:
            raise ValueError(f"missing report target {target}")
        for n in n_values:
            if str(n) not in summary["targets"][target]:
                raise ValueError(f"missing report cell {target}, n={n}")
            cell = summary["targets"][target][str(n)]
            missing = set(methods) - cell["methods"].keys()
            if missing:
                raise ValueError(f"missing enabled report methods {sorted(missing)}")


def _display(cell: Mapping[str, Any], *, tex: bool = False) -> str:
    if cell["status"] == "undefined":
        return r"n.a.$^{*}$" if tex else "undefined"
    if cell["status"] == "infinite":
        return r"$\infty^{*}$" if tex else "+infinity"
    return f"{cell['mean_bits']:.4f}"


def _write_tsv(path: Path, rows: Sequence[Sequence[Any]]) -> None:
    with path.open("x", encoding="utf8", newline="") as stream:
        csv.writer(stream, delimiter="\t").writerows(rows)


def write_loss_table(
    summary: Mapping[str, Any],
    output_stem: str | Path,
    *,
    n: int,
    targets: Sequence[str],
    methods: Sequence[str],
    label: str,
) -> dict:
    """Full precision TSV plus manuscript precision TeX from identical cells."""
    _ensure_cells(summary, targets, [n], methods)
    stem = Path(output_stem)
    rows = [
        [
            "target",
            "n",
            "method",
            "status",
            "mean_bits",
            "se_bits",
            "trials",
            "finite_trials",
            "infinite_trials",
            "undefined_trials",
        ]
    ]
    tex = [
        f"% label: {label}; sample_set={summary['config']['sample_set_id']}; n={n}",
        r"\begin{tabular}{l" + "r" * len(methods) + "}",
        r"\toprule",
        "target & " + " & ".join(METHOD_LABELS[m] for m in methods) + r" \\",
        r"\midrule",
    ]
    finite_ses = []
    for target in targets:
        cells = summary["targets"][target][str(n)]["methods"]
        candidates = [
            round(cells[m]["mean_bits"], 4)
            for m in methods
            if m != "oracle" and cells[m]["status"] == "finite"
        ]
        best = min(candidates) if candidates else None
        values = []
        for method in methods:
            cell = cells[method]
            rows.append(
                [
                    target,
                    n,
                    method,
                    cell["status"],
                    cell["mean_bits"],
                    cell["se_bits"],
                    cell["trials"],
                    *[cell["counts"][s] for s in ("finite", "infinite", "undefined")],
                ]
            )
            if cell["se_bits"] is not None:
                finite_ses.append(cell["se_bits"])
            value = _display(cell, tex=True)
            if (
                method != "oracle"
                and cell["status"] == "finite"
                and round(cell["mean_bits"], 4) == best
            ):
                value = r"\textbf{" + value + "}"
            values.append(value)
        tex.append(TARGET_LABELS[target] + " & " + " & ".join(values) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    _write_tsv(stem.with_suffix(".tsv"), rows)
    with stem.with_suffix(".tex").open("x", encoding="utf8") as stream:
        stream.write("\n".join(tex) + "\n")
    return {
        "label": label,
        "n": n,
        "methods": list(methods),
        "max_finite_standard_error_bits": max(finite_ses) if finite_ses else None,
        "rounding": "four decimal places; bold ties at that precision, oracle excluded",
        "source": summary.get("source"),
    }


def write_posterior_table(
    summary: Mapping[str, Any],
    output_stem: str | Path,
    *,
    targets: Sequence[str],
    n_values: Sequence[int],
    mass: float,
    broad_min_depths: int,
    individual_min_weight: float,
) -> dict:
    _ensure_cells(summary, targets, n_values, ["lsa_depth_mixture"])
    stem = Path(output_stem)
    rows = [["target", *[f"n={n}" for n in n_values]]]
    details = {}
    tex = [
        r"\begin{tabular}{l" + "l" * len(n_values) + "}",
        r"\toprule",
        "target & " + " & ".join(f"$n={n}$" for n in n_values) + r" \\",
        r"\midrule",
    ]
    for target in targets:
        cells = []
        details[target] = {}
        for n in n_values:
            post = summary["targets"][target][str(n)]["mean_posteriors"]["depth"]
            detail = posterior_summary(
                post["indices"],
                post["weights"],
                mass=mass,
                broad_min_depths=broad_min_depths,
                individual_min_weight=individual_min_weight,
            )
            cells.append(detail["text"])
            details[target][str(n)] = detail
        rows.append([target, *cells])
        tex.append(
            TARGET_LABELS[target]
            + " & "
            + " & ".join(s.replace("%", r"\%") for s in cells)
            + r" \\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    _write_tsv(stem.with_suffix(".tsv"), rows)
    with stem.with_suffix(".tex").open("x", encoding="utf8") as stream:
        stream.write("\n".join(tex) + "\n")
    return {
        "label": "tab:posteriors",
        "source": summary.get("source"),
        "cells": details,
    }


def plot_learning_curves(
    summary: Mapping[str, Any],
    output_path: str | Path,
    *,
    targets: Sequence[str],
    methods: Sequence[str],
    uniform_floor: float,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter

    n_values = summary["config"]["n_values"]
    _ensure_cells(summary, targets, n_values, methods)
    if uniform_floor <= 0 or not np.isfinite(uniform_floor):
        raise ValueError("uniform plot floor must be positive")
    if Path(output_path).exists():
        raise FileExistsError(output_path)
    styles = {
        "add_one": ("0.6", "+", "-"),
        "kt": ("0.4", "x", "-"),
        "ristad": ("tab:purple", "^", "-"),
        "good_turing_hybrid": ("tab:green", "s", "-"),
        "dirichlet_concentration_mixture": ("tab:cyan", "P", "-"),
        "absolute_discounting": ("tab:orange", "v", "-"),
        "lsa_fixed": ("tab:blue", "o", "-"),
        "lsa_depth_mixture": ("tab:red", "D", "-"),
        "oracle": ("black", "", ":"),
    }
    if set(methods) - styles.keys():
        raise ValueError(
            "learning curve requested a method absent from the current figure"
        )
    columns = 4
    rows = (len(targets) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(2.25 * columns, 2.25 * rows), squeeze=False
    )
    details = {
        "nonfinite_points": [],
        "uniform_floor_points": [],
        "zero_points_outside_uniform": [],
    }
    handles = []
    for method in methods:
        color, marker, line = styles[method]
        label = METHOD_LABELS[method]
        if method == "lsa_fixed":
            label = f"LSA L={summary['config']['fixed_depth']}"
        elif method == "lsa_depth_mixture":
            depths = summary["config"]["depths"]
            label = f"LSA avg, L={depths[0]}–{depths[-1]}"
        handles.append(
            Line2D([], [], color=color, marker=marker, linestyle=line, label=label)
        )
    for panel, target in enumerate(targets):
        ax = axes.flat[panel]
        for method in methods:
            color, marker, line = styles[method]
            x, means, ses = [], [], []
            for n in n_values:
                cell = summary["targets"][target][str(n)]["methods"][method]
                if cell["status"] != "finite":
                    details["nonfinite_points"].append(
                        {
                            "target": target,
                            "n": n,
                            "method": method,
                            "status": cell["status"],
                            "counts": cell["counts"],
                        }
                    )
                    # NaN breaks a line; do not bridge omitted AD points.
                    x.append(n)
                    means.append(np.nan)
                    ses.append(np.nan)
                    continue
                mean, se = cell["mean_bits"], cell["se_bits"]
                if se is None:
                    raise ValueError("SE figure requires at least two trials per cell")
                if target == "uniform" and mean <= uniform_floor:
                    ax.scatter(
                        [n], [uniform_floor], marker="v", color=color, s=24, zorder=5
                    )
                    details["uniform_floor_points"].append(
                        {"method": method, "n": n, "mean_bits": mean}
                    )
                    x.append(n)
                    means.append(np.nan)
                    ses.append(np.nan)
                elif mean <= 0:
                    details["zero_points_outside_uniform"].append(
                        {"target": target, "method": method, "n": n}
                    )
                    x.append(n)
                    means.append(np.nan)
                    ses.append(np.nan)
                else:
                    x.append(n)
                    means.append(mean)
                    ses.append(se)
            ax.errorbar(
                x,
                means,
                yerr=ses,
                color=color,
                marker=marker,
                linestyle=line,
                markersize=3,
                linewidth=0.9,
                capsize=1.5,
            )
        omitted = [
            item for item in details["nonfinite_points"] if item["target"] == target
        ]
        if omitted:
            names = list(
                dict.fromkeys(METHOD_LABELS[item["method"]] for item in omitted)
            )
            ax.text(
                0.035,
                0.025,
                ", ".join(names) + ": ∞ / n.a.",
                transform=ax.transAxes,
                fontsize=7.8,
            )
        ax.set(xscale="log", yscale="log", title=TARGET_LABELS[target])
        if target == "uniform":
            ax.set_yticks([1.0, 1e-4, 1e-8, 1e-12, uniform_floor])
        tick_indices = np.unique(
            np.linspace(0, len(n_values) - 1, min(4, len(n_values))).round().astype(int)
        )
        tick_values = [n_values[i] for i in tick_indices]
        ax.set_xticks(tick_values, labels=[f"{n / 1000:g}k" for n in tick_values])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=8.5)
        if panel // columns == rows - 1:
            ax.set_xlabel("sample size n", fontsize=9)
        if panel % columns == 0:
            ax.set_ylabel("mean next-symbol KL [bits]", fontsize=9)
    for ax in axes.flat[len(targets) :]:
        ax.axis("off")
    # The twelfth panel holds the same nine-method legend as the manuscript.
    if len(targets) < rows * columns:
        axes.flat[-1].legend(handles=handles, loc="center", fontsize=9.5, frameon=False)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    fig.text(
        0.5,
        0.006,
        f"Bars: one standard error; triangles: uniform loss ≤ {uniform_floor:g} bits.\n"
        "Sample-size ticks: k = 1,000. Nonfinite AD cases marked in panels.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "label": "fig:orlitsky",
        "methods": list(methods),
        "source": summary.get("source"),
        "standard_errors": "sample SD / sqrt(trials); not confidence intervals",
        "uniform_floor_bits": uniform_floor,
        "nonfinite_policy": "nonfinite means not plotted; line broken and locations recorded",
        **details,
    }


def spectrum(summary, output_path):
    import matplotlib.pyplot as plt

    config, rows = summary["config"], summary["cells"]
    assert config["kind"] == "spectrum" and config["trials"] == 1000
    assert len(rows) == 33
    omissions, plotted = [], []
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.6))
    common_handles = None
    for ax, panel in zip(axes, config["panels"], strict=True):
        group = sorted(
            (r for r in rows if r["d"] == panel["d"] and r["n"] == panel["n"]),
            key=lambda r: r["alpha"],
        )
        assert len(group) == 11 and all(r["trials"] == 1000 for r in group)
        assert [r["alpha"] for r in group] == config["alphas"]
        xs = np.asarray([r["alpha"] for r in group])
        positive_depths = [d for d in panel["show_depths"] if d != 0]
        colors = plt.get_cmap("viridis")(np.linspace(0, 0.94, len(positive_depths)))
        colors = dict(zip(positive_depths, colors, strict=True))
        handles, labels = [], []
        for depth in panel["show_depths"]:
            ix = group[0]["depths"].index(depth)
            assert all(r["depths"] == group[0]["depths"] for r in group)
            y = np.asarray([r["regret_mean_bits"][ix] for r in group])
            se = np.asarray([r["regret_se_bits"][ix] for r in group])
            keep = np.ones(len(group), dtype=bool)
            if depth == 0:
                keep = xs != 0
                zero = group[0]
                assert zero["alpha"] == 0 and abs(y[0]) < 1e-10 and se[0] == 0
                omissions.append(
                    {
                        "d": panel["d"],
                        "n": panel["n"],
                        "L": 0,
                        "alpha": 0,
                        "saved_mean_bits": float(y[0]),
                        "saved_se_bits": float(se[0]),
                        "reason": "Exact uniform predictor/target regret is zero; omitted on log axis, no display floor.",
                    }
                )
            assert np.all(np.isfinite(y)) and np.all(np.isfinite(se))
            assert np.all(y[keep] > 0) and np.all(se >= 0)
            color = ".38" if depth == 0 else colors[depth]
            handle = ax.errorbar(
                xs[keep],
                y[keep],
                yerr=se[keep],
                color=color,
                linestyle=":" if depth == 0 else "-",
                marker=None if depth == 0 else "o",
                markersize=3,
                linewidth=1.2,
                capsize=2,
                capthick=0.8,
                elinewidth=0.8,
            )
            if depth == 0:
                zero_handle = handle
            else:
                handles.append(handle)
                labels.append(rf"$L={depth}$")
            plotted.extend(
                {
                    "d": panel["d"],
                    "n": panel["n"],
                    "alpha": float(x),
                    "L": depth,
                    "mean_bits": float(v),
                    "se_bits": float(s),
                }
                for x, v, s in zip(xs[keep], y[keep], se[keep], strict=True)
            )
        my = np.asarray([r["mixture_mean_bits"] for r in group])
        ms = np.asarray([r["mixture_se_bits"] for r in group])
        assert np.all(np.isfinite(my)) and np.all(my > 0) and np.all(ms >= 0)
        mixture = ax.errorbar(
            xs,
            my,
            yerr=ms,
            color="red",
            linestyle="--",
            linewidth=1.9,
            capsize=2,
            capthick=0.8,
            elinewidth=0.8,
        )
        common_handles = [zero_handle, mixture]
        plotted.extend(
            {
                "d": panel["d"],
                "n": panel["n"],
                "alpha": float(x),
                "L": "mixture",
                "mean_bits": float(v),
                "se_bits": float(s),
            }
            for x, v, s in zip(xs, my, ms, strict=True)
        )
        exponent = {1000: 3, 10000: 4, 1000000: 6}[panel["d"]]
        ax.set(
            yscale="log",
            xlabel=r"Zipf exponent $\alpha$ of the target",
            title=rf"$d=10^{exponent},\ N={panel['n']}$",
            xlim=(-0.15, 3.15),
        )
        ax.set_xticks(np.arange(0, 3.1, 0.5))
        ax.grid(True, which="both", color=".85", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.legend(
            handles,
            labels,
            ncol=2,
            fontsize=7,
            loc="lower right",
            frameon=False,
            columnspacing=0.65,
            handletextpad=0.4,
            labelspacing=0.25,
            handlelength=1.4,
            borderpad=0.2,
        )
    axes[0].set_ylabel(r"online regret $R_N$ [bits/symbol]")
    fig.legend(
        common_handles,
        [r"$L=0$ (uniform model)", r"mixture, $L=0,\ldots,L_{\max}$"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.052),
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=2.1,
    )
    fig.text(
        0.5,
        0.007,
        r"$L=0,\ \alpha=0$: exact zero omitted. Bars: one standard error over 1,000 profiles.",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.135, 1, 1), pad=0.55)
    temp_spectrum = Path(output_path)
    fig.savefig(
        temp_spectrum,
        bbox_inches="tight",
        metadata={
            "Title": "Layered simplex spectrum",
            "Creator": "reproduce_paper.py; saved numerical summaries",
        },
    )
    plt.close(fig)
    return {"omitted_exact_zeros": omissions, "plotted_points": plotted}


from scipy.special import gammaln

LOG2 = math.log(2)
C_STAR = 1 / (1 - float(np.euler_gamma))


def rho(c):
    if c >= C_STAR:
        return 1.0
    y, upper = 1 / c, 10.0
    while digamma(1 + upper) < y:
        upper *= 2
    s = brentq(lambda t: digamma(1 + t) - y, 1e-9, upper, xtol=1e-12)
    return float(c * (s * y - gammaln(1 + s)))


def depth_scaling(summary, output_path):
    import matplotlib.pyplot as plt

    config, rows = summary["config"], summary["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    claims = []
    for r in rows:
        cs = np.asarray(config["c_values"])
        ix = [r["depths"].index(max(1, round(c * math.log(r["d"])))) for c in cs]
        measured = (
            np.asarray(r["regret_mean_bits"])[ix] * r["n"] / r["expected_discovered"]
        )
        x = r["naming_mean_bits"] / r["expected_discovered"]
        predicted = np.asarray([rho(c) * x + r["alpha"] / LOG2 - 2 for c in cs])
        color = {2: "tab:blue", 3: "tab:orange", 4: "tab:green"}.get(r["alpha"])
        for ax, c_min in zip(axes, (0, 0.75)):
            keep = cs >= c_min
            ax.plot(
                cs[keep],
                measured[keep],
                "o",
                ms=3.5,
                color=color,
                label=f"measured, α = {r['alpha']:g}",
            )
            ax.plot(
                cs[keep],
                predicted[keep],
                "--",
                lw=1.1,
                color=color,
                label=f"prediction, α = {r['alpha']:g}",
            )
        best = int(np.argmin(measured))
        at_star = int(np.argmin(np.abs(cs - C_STAR)))
        claims.append(
            {
                "alpha": r["alpha"],
                "best_tested_c": float(cs[best]),
                "cost_at_nearest_c_star": float(measured[at_star]),
                "minimum_cost": float(measured[best]),
                "nearest_c_star_grid_point": float(cs[at_star]),
                "offset_bits": r["alpha"] / LOG2 - 2,
            }
        )
    for ax, title in zip(axes, ("Full depth range", "Logarithmic-depth range")):
        ax.axvline(C_STAR, color=".4", ls=":", lw=1)
        ax.set(xlabel="depth coefficient c", title=title)
    axes[0].set_ylabel("cost per discovered symbol [bits]")
    axes[1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return claims


def construction(draw, template, output_path):
    layers, theta = np.asarray(draw["layers"]), np.asarray(draw["theta"])
    require(
        layers.ndim == 2 and layers.shape[1] == 24,
        "construction must have 24 coordinates",
    )
    require(
        np.allclose(layers.sum(axis=1), 1, rtol=0, atol=1e-14), "invalid simplex layers"
    )
    weights = np.prod(layers, axis=0)
    require(
        np.allclose(theta, weights / weights.sum(), rtol=1e-12, atol=1e-15),
        "construction product differs from saved layers",
    )
    indices = [i - 1 for i in draw["displayed_layers_one_based"]]
    shown = np.vstack([layers[indices], theta]) * draw["height_scale_mm"]
    source = Path(template).read_text()
    for i, row in enumerate(shown, 1):
        source = source.replace(
            f"@BARS_{i}@", ",".join(f"{j + 1}/{h:.2f}" for j, h in enumerate(row))
        )
    guide = int(np.argmax(theta)) + 1.31
    source = source.replace("@WINNER_X@", f"{guide:.2f}")
    require("@" not in source, "unresolved construction template token")
    Path(output_path).write_text(source)


def bible_table(summary, output_path):
    methods = summary["config"]["methods"]
    labels = ["Add-one", "KT", "Ristad", "GT", r"Dir-$\tau$", "LSA"]
    lines = [
        r"\begin{tabular}{" + "r" * 10 + "}",
        r"$n$ & distinct & $\hat H_n$ & "
        + " & ".join(labels)
        + r" & mode $L$ \\ \hline",
    ]
    for row in summary["rows"]:
        if row["n"] not in summary["config"]["prefixes"]:
            continue
        minimum = min(round(row["redundancy_bits"][m], 3) for m in methods)
        values = []
        for m in methods:
            value = f"{row['redundancy_bits'][m]:.3f}"
            values.append(
                r"\textbf{" + value + "}"
                if round(row["redundancy_bits"][m], 3) == minimum
                else value
            )
        lines.append(
            f"{row['n']} & {row['distinct']} & {row['empirical_entropy_bits']:.2f} & "
            + " & ".join(values)
            + " & "
            + str(row["depths"][int(np.argmax(row["posterior"]))])
            + r" \\"
        )
    Path(output_path).write_text("\n".join([*lines, r"\end{tabular}"]) + "\n")


def factorial_tables(fits, output_path):
    alphabet = {(r["alpha"], r["n"]): r for r in fits["alphabet"]}
    data = {(r["alpha"], r["d"]): r for r in fits["data"]}
    alphas = sorted({r["alpha"] for r in fits["alphabet"]})
    lines = [
        r"\begin{tabular}{c ccc ccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{free-fit slope} & \multicolumn{3}{c}{slope-one offset $B$} \\",
        r"$\alpha$ & $N=10^2$ & $10^3$ & $10^4$ & $N=10^2$ & $10^3$ & $10^4$ \\",
        r"\midrule",
    ]
    for a in alphas:
        rows = [alphabet[a, n] for n in (100, 1000, 10000)]
        values = [
            f"{r[k]:.2f}" for k in ("slope", "slope_one_offset_bits") for r in rows
        ]
        lines.append(f"{a:g} & " + " & ".join(values) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\hfill",
        r"\begin{tabular}{c c cc}",
        r"\toprule",
        r"& asymptotic & \multicolumn{2}{c}{measured exponent} \\",
        r"$\alpha$ & $1-1/\alpha$ & $d=10^3$ & $d=10^5$ \\",
        r"\midrule",
    ]
    for a in alphas:
        values = [f"{data[a, d]['exponent']:.3f}" for d in (1000, 100000)]
        lines.append(f"{a:g} & {1 - 1 / a:.3f} & " + " & ".join(values) + r" \\")
    Path(output_path).write_text(
        "\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n"
    )


def verify_fits(summary, fits):
    """Check saved regression results against saved means, without sampling."""
    for row in fits["alphabet"]:
        group = sorted(
            (
                r
                for r in summary["cells"]
                if (r["alpha"], r["n"]) == (row["alpha"], row["n"])
            ),
            key=lambda r: r["d"],
        )
        x = np.array([r["naming_mean_bits"] / r["expected_discovered"] for r in group])
        y = np.array(
            [
                r["n"] * r["regret_mean_bits"][0] / r["expected_discovered"]
                for r in group
            ]
        )
        slope, intercept = np.polyfit(x, y, 1)
        for key, value in [
            ("slope", slope),
            ("offset_bits", intercept),
            ("slope_one_offset_bits", np.mean(y - x)),
        ]:
            require(
                math.isclose(row[key], float(value), rel_tol=1e-12, abs_tol=1e-12),
                "saved alphabet fit disagrees with saved means",
            )
    for row in fits["data"]:
        group = sorted(
            (
                r
                for r in summary["cells"]
                if (r["alpha"], r["d"]) == (row["alpha"], row["d"])
            ),
            key=lambda r: r["n"],
        )
        slope, _ = np.polyfit(
            np.log([r["n"] for r in group]),
            np.log([r["regret_mean_bits"][0] for r in group]),
            1,
        )
        require(
            math.isclose(row["exponent"], -float(slope), rel_tol=1e-12, abs_tol=1e-12),
            "saved data exponent disagrees with saved means",
        )


def upward_tex(value, digits=3):
    from decimal import ROUND_CEILING, Decimal

    x = Decimal(str(value))
    if not x:
        return "0"
    power = x.adjusted()
    mantissa = x.scaleb(-power).quantize(
        Decimal(1).scaleb(1 - digits), rounding=ROUND_CEILING
    )
    if mantissa == 10:
        mantissa /= 10
        power += 1
    return rf"{mantissa}\times10^{{{power}}}"


def validation_table(validation, output_path):
    m = validation["metrics"]
    rows = [
        (
            "Independent kernels: 263 cases",
            r"Log-kernel error, all / $r\le3$ (nats)",
            ["kernel.sealed_max_error", "kernel.sealed_low_count_max_error"],
        ),
        (
            r"Meijer-$G$ versus contour: $L=1,2,3$, $r=0,\ldots,3$, $u=-2,0,2$",
            "Log-kernel discrepancy (nats)",
            ["kernel.meijer_error_nats"],
        ),
        (
            "Simplex versus reference: $d=2$, $L=2,3$",
            "Sequence-probability discrepancy",
            ["prior.direct_simplex"],
        ),
        (
            "Table versus direct: 29 profiles",
            "Component / mixture codelength change (bits/token)",
            [
                "sealed_profile.maximum_codelength_difference_bits_per_token",
                "sealed_profile.mixture_codelength_difference_bits_per_token",
            ],
        ),
        (
            "",
            "Augmented codelength change (bits/token)",
            ["sealed_profile.maximum_augmented_codelength_difference_bits_per_token"],
        ),
        (
            "",
            "Component / mixture predictive KL change (bits)",
            [
                "sealed_profile.maximum_predictive_kl_change_bits",
                "sealed_profile.mixture_predictive_kl_change_bits",
            ],
        ),
        ("", "Raw probability-mass error", ["sealed_profile.maximum_raw_mass_error"]),
        (
            "Outer-grid halving: 27 profiles",
            "Component codelength change (bits/token)",
            ["depth.maximum_codelength_difference_bits_per_token"],
        ),
        (
            "",
            "Component predictive KL change (bits)",
            ["depth.maximum_predictive_kl_change_bits"],
        ),
        (
            r"Power refinement: 11 profiles, $d=10^4$, $N=10^3$, $w=0,\ldots,80$",
            "Component log-evidence change (bits)",
            ["power.max_log_evidence_change_bits"],
        ),
        (
            "",
            "Component / mixture predictive KL change (bits)",
            [
                "power.max_predictive_kl_change_bits",
                "power.mixture_predictive_kl_change_bits",
            ],
        ),
        (
            r"Bible chain: first 2,000 transitions, $d=10^5$, depths $0,\ldots,54$",
            "Cumulative chain residual (total bits)",
            ["chain.max_chain_error_bits"],
        ),
        ("", "Next-symbol probability residual", ["chain.max_probability_error"]),
    ]
    lines = [
        r"\begin{longtable}{@{}p{0.29\linewidth}p{0.32\linewidth}p{0.32\linewidth}@{}}",
        r"\caption{Maximum absolute discrepancies on the finite test domains. Values are rounded upwards; slash-separated quantities follow the listed order.}\label{tab:validation}\\",
        r"\toprule Check & Quantity and unit & Maximum\\\midrule",
        r"\endfirsthead",
        r"\toprule Check & Quantity and unit & Maximum\\\midrule\endhead",
        r"\bottomrule\endfoot",
    ]
    for check, quantity, keys in rows:
        values = " / ".join("$" + upward_tex(m[k]["value"]) + "$" for k in keys)
        lines.append(f"{check} & {quantity} & {values}" + r"\\")
    Path(output_path).write_text("\n".join([*lines, r"\end{longtable}"]) + "\n")
    return [
        {
            "check": c,
            "quantity": q,
            "metric_ids": keys,
            "values": [m[k]["value"] for k in keys],
        }
        for c, q, keys in rows
    ]
