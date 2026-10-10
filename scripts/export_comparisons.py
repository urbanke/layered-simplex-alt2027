#!/usr/bin/env python3
"""Export all saved method and paired statistics, including nonfinite outcomes.

Use the compact results bundle by default. --runs reads locally recomputed
benchmark Runs after verifying their complete saved-file inventories.
No estimators, new uncertainty estimates, or hypothesis tests are computed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from paper_results import (
    EXPERIMENTS,
    ROOT,
    STATUSES,
    check_targets,
    digest,
    load_benchmarks,
    new_output,
    output_manifest,
    pairs,
    read,
    read_outcomes,
    require,
    validate_blocks,
    write,
)


def verify_runs(root):
    sys.path.insert(0, str(ROOT / "src"))
    from lsa.alt.artifacts import verify_run

    hashes = {}
    for family in EXPERIMENTS:
        source = Path(root) / family
        require(
            source.is_dir() and not source.is_symlink(),
            "missing or unsafe benchmark Run",
        )
        require(
            not any(p.is_symlink() for p in source.rglob("*")),
            "symlink in benchmark Run",
        )
        record = verify_run(source, require_complete=True)
        require(record["experiment"] == family, "wrong benchmark experiment")
        hashes[family] = digest(source / "manifest.json")
    return hashes


def load_runs(root):
    hashes = verify_runs(root)
    result = {}
    for family in EXPERIMENTS:
        source = Path(root) / family
        summary = read(source / "data/summary.json")
        config = summary["config"]
        protocol = read(source / "protocol.json")
        require(
            protocol["experiments"][family] == config, "summary and Run protocol differ"
        )
        require(
            digest(source / "data/trials.jsonl") == summary["trial_records_sha256"],
            "saved trial hash mismatch",
        )
        blocks = read(source / "data/block-diagnostics.json")
        check_targets(summary["targets"], config, config["trials"])
        validate_blocks(blocks, config, summary, 10)
        reasons = read_outcomes(source, config, summary)
        nested = {
            t: {str(n): reasons[t, n] for n in config["n_values"]}
            for t in config["targets"]
        }
        clean = {
            k: summary[k]
            for k in (
                "schema_version",
                "config",
                "nonfinite_policy",
                "uncertainty",
                "units",
                "targets",
            )
        }
        result[family] = (clean, blocks, nested)
    a, b = [result[f][0]["config"] for f in EXPERIMENTS]
    require(
        a["seed"] != b["seed"] and a["sample_set_id"] != b["sample_set_id"],
        "sample sets must differ",
    )
    return result, hashes


def export_rows(families, out):
    common = ["experiment", "sample_set_id", "target", "n"]
    measures = [
        "status",
        "mean_bits",
        "se_bits",
        "trials",
        *[s + "_trials" for s in STATUSES],
    ]
    counts = {}

    def fields(item):
        return [item[k] for k in ("status", "mean_bits", "se_bits", "trials")] + [
            item["counts"][s] for s in STATUSES
        ]

    with (
        (out / "methods-all.tsv").open("x", newline="") as mf,
        (out / "paired-differences-all.tsv").open("x", newline="") as pf,
    ):
        mw, pw = csv.writer(mf, delimiter="\t"), csv.writer(pf, delimiter="\t")
        mw.writerow(common + ["method"] + measures)
        pw.writerow(
            common
            + ["comparison", "left", "right"]
            + measures
            + [f"{side}_{s}_trials" for side in ("left", "right") for s in STATUSES]
            + ["recorded_reasons_json"]
        )
        for family in EXPERIMENTS:
            summary, blocks, reasons = families[family]
            config = summary["config"]
            count = {
                "cells": 0,
                "method_rows": 0,
                "paired_rows": 0,
                "blocks": len(blocks["blocks"]),
            }
            for target in config["targets"]:
                for n in config["n_values"]:
                    count["cells"] += 1
                    prefix = [family, config["sample_set_id"], target, n]
                    cell = summary["targets"][target][str(n)]
                    for method, item in cell["methods"].items():
                        mw.writerow(prefix + [method] + fields(item))
                        count["method_rows"] += 1
                    for pair, (left, right) in sorted(pairs(config).items()):
                        sides = {**cell["methods"], **cell.get("derived", {})}
                        pw.writerow(
                            prefix
                            + [pair, left, right]
                            + fields(cell["paired_differences"][pair])
                            + [
                                sides[m]["counts"][s]
                                for m in (left, right)
                                for s in STATUSES
                            ]
                            + [
                                json.dumps(
                                    dict(reasons[target][str(n)][pair]), sort_keys=True
                                )
                            ]
                        )
                        count["paired_rows"] += 1
            write(out / f"{family}-summary.json", summary)
            write(out / f"{family}-blocks.json", blocks)
            counts[family] = count
    write(out / "coverage.json", counts)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--results", type=Path, default=ROOT / "results")
    group.add_argument(
        "--runs",
        type=Path,
        help="Root containing complete benchmark_primary and benchmark_powers Runs",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.runs or args.results
    if args.runs:
        families, before = load_runs(source)
    else:
        families = load_benchmarks(source)
        before = {"results_manifest_sha256": digest(source / "manifest.json")}
    out = new_output(args.out, [source])
    counts = export_rows(families, out)
    if args.runs:
        require(verify_runs(source) == before, "input Runs changed during export")
    else:
        load_benchmarks(source)
        require(
            digest(source / "manifest.json") == before["results_manifest_sha256"],
            "input bundle changed during export",
        )
    output_manifest(
        out,
        inputs=before,
        scope="Saved paired and marginal statistics; blocks are descriptive diagnostics, not confidence intervals.",
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
