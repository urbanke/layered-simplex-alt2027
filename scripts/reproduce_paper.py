#!/usr/bin/env python3
"""Regenerate current-paper figures and tables from the compact saved results.

This is deterministic reporting, not a fresh experiment. Rendering may vary
slightly with library versions; numerical plotted/table data are preserved.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from export_comparisons import export_rows
from paper_render import (
    bible_table,
    construction,
    depth_scaling,
    factorial_tables,
    plot_learning_curves,
    spectrum,
    validation_table,
    verify_fits,
    write_loss_table,
    write_posterior_table,
)
from paper_results import (
    ROOT,
    digest,
    load_benchmarks,
    new_output,
    output_manifest,
    read,
    require,
    verify_bundle,
    write,
)


def reproduce(results, out, *, runs=None, validation_root=None):
    results = Path(results)
    manifest = verify_bundle(results)
    manifest_hash = digest(results / "manifest.json")
    if runs is not None:
        from paper_fresh import load_fresh

        families, fresh, inputs = load_fresh(
            runs, validation_root, read(results / "data/validation.json")
        )
        data = fresh.__getitem__
        inputs["presentation_bundle_sha256"] = manifest_hash
    else:
        require(validation_root is None, "--validation-root requires --runs")
        families = load_benchmarks(results)
        data = lambda name: read(results / "data" / name)
        inputs = {"results_manifest_sha256": manifest_hash}
    fits = data("factorial-fits.json")
    verify_fits(data("factorial.json.gz"), fits)
    root = new_output(
        out,
        [
            results,
            *([runs] if runs else []),
            *([validation_root] if validation_root else []),
        ],
    )
    figures = root / "figures"
    tables = root / "tables"
    comparisons = root / "comparisons"
    for p in (figures, tables, comparisons):
        p.mkdir()
    primary, power = [families[x][0] for x in ("benchmark_primary", "benchmark_powers")]
    settings = data("report-settings.json")
    construction(
        data("construction.json"),
        results / "templates/construction.tex",
        figures / "construction_tikz.tex",
    )
    spectrum_details = spectrum(
        data("spectrum.json.gz"), figures / "complexity_spectrum_L0.pdf"
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "pdf.fonttype": 42,
        }
    )
    learning_details = plot_learning_curves(
        primary,
        figures / "regret_vs_n_figures.pdf",
        targets=primary["config"]["targets"],
        methods=primary["config"]["methods"],
        uniform_floor=settings["uniform_floor"],
    )
    # The depth plot retains the paper's original Matplotlib typography.
    plt.rcdefaults()
    depth_details = depth_scaling(
        data("depth_scaling.json.gz"), figures / "depth_scaling.pdf"
    )
    table_details = {}
    for name, summary, label in [
        ("primary", primary, "tab:bench"),
        ("powers", power, "tab:powers-bench"),
    ]:
        table_details[label] = write_loss_table(
            summary,
            tables / name,
            n=settings["table_n"],
            targets=summary["config"]["targets"],
            methods=summary["config"]["methods"],
            label=label,
        )
    table_details["tab:posteriors"] = write_posterior_table(
        primary,
        tables / "posteriors",
        targets=primary["config"]["targets"],
        n_values=settings["posterior_n_values"],
        mass=settings["posterior_mass"],
        broad_min_depths=settings["posterior_broad_min_depths"],
        individual_min_weight=settings["posterior_individual_min_weight"],
    )
    bible_table(data("bible.json"), tables / "bible.tex")
    factorial_tables(fits, tables / "factorial.tex")
    validation = data("validation.json")
    validation_rows = (
        validation_table(validation, tables / "validation.tex")
        if validation
        else {
            "status": "not_supplied",
            "reason": "Fresh validation Runs were not supplied; reference measurements are never substituted.",
        }
    )
    counts = export_rows(families, comparisons)
    write(
        root / "plot-data.json",
        {
            "spectrum": spectrum_details,
            "learning_curves": learning_details,
            "depth_scaling": depth_details,
        },
    )
    write(
        root / "table-data.json",
        {
            "benchmark_tables": table_details,
            "bible": data("bible.json"),
            "factorial": fits,
            "validation": validation_rows,
        },
    )
    write(
        root / "in-text-controls.json",
        {
            "bible_short_prefixes": [
                r for r in data("bible.json")["rows"] if r["n"] in (100, 500)
            ],
            "tokenization": data("tokenization-controls.json"),
            "validation": data("validation.json"),
            "runtime": data("runtime.json"),
            "depth_scaling": depth_details,
        },
    )
    paper_map = deepcopy(manifest["paper_map"])
    if runs is not None:
        paper_map["input_mode"] = "fresh_runs"
        for asset in [*paper_map["figures"], *paper_map["tables"]]:
            asset.pop("data", None)
            asset["input_binding"] = (
                "Fresh Run manifests are pinned in the output manifest."
            )
        paper_map["in_text_controls"] = ["in-text-controls.json"]
        paper_map["runtime"] = "Reference timing pilots are not reused."
        if validation is None:
            paper_map["tables"] = [
                t for t in paper_map["tables"] if t["labels"] != ["tab:validation"]
            ]
            paper_map["omitted_tables"] = [
                "tab:validation: no fresh validation Runs supplied"
            ]
        from paper_fresh import recheck_fresh

        recheck_fresh(runs, validation_root, inputs)
    write(root / "paper-map.json", paper_map)
    verify_bundle(results)
    assert digest(results / "manifest.json") == manifest_hash
    output_manifest(
        root,
        inputs=inputs,
        scope="Current-paper reporting from saved numerical results; no model evaluation or sampling.",
    )
    return {
        "figures": 4,
        "numbered_tables": 6 if validation is not None else 5,
        "table_labels": 7 if validation is not None else 6,
        "comparisons": counts,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--results", type=Path, default=ROOT / "results")
    group.add_argument(
        "--runs",
        type=Path,
        help="Verified full campaign or merged Run root; uses fresh saved measurements",
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        help="Eight fresh numerical Run directories for the validation table",
    )
    parser.add_argument(
        "--out", type=Path, help="New output directory (required unless --check)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify compact data and comparison coverage without rendering",
    )
    args = parser.parse_args(argv)
    if args.validation_root and not args.runs:
        parser.error("--validation-root requires --runs")
    if args.check and args.runs:
        parser.error(
            "--check applies to compact results; fresh Runs are verified during rendering"
        )
    if args.check:
        load_benchmarks(args.results)
        verify_fits(
            read(args.results / "data/factorial.json.gz"),
            read(args.results / "data/factorial-fits.json"),
        )
        print(
            "Verified compact results, coverage, ten-block partitions and factorial fits."
        )
    else:
        if args.out is None:
            parser.error("--out is required")
        print(
            json.dumps(
                reproduce(
                    args.results,
                    args.out,
                    runs=args.runs,
                    validation_root=args.validation_root,
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
