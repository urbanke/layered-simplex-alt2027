#!/usr/bin/env python3
"""Freeze a recomputation specification from existing accepted local metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recompute_saved_ad import EXPERIMENTS, digest_file, require, write_json


def make(local_package, remote_package, local_repo, remote_repo, output):
    local_package, local_repo = Path(local_package), Path(local_repo)
    metadata, trials = {}, {}
    for experiment in EXPERIMENTS:
        root = local_package / "merged" / experiment
        manifest = json.loads((root / "manifest.json").read_text())
        for relative in ("manifest.json", "data/benchmark-config.json", "data/samples/manifest.json",
                         "data/summary.json", "data/block-diagnostics.json"):
            path = root / relative
            pin = {"bytes": path.stat().st_size, "sha256": digest_file(path)}
            if relative != "manifest.json":
                require(pin == manifest["outputs"][relative], f"local metadata pin mismatch: {path}")
            metadata[f"merged/{experiment}/{relative}"] = pin
        trials[experiment] = manifest["outputs"]["data/trials.jsonl"]
    # Bind the successful accepted package independently of the original Runs'
    # historical validation-purpose labels. No raw artifact is relabelled.
    extra = ("acceptance/manifest.json", "acceptance/result.json", "merged/merge.json")
    for relative in extra:
        path = local_package / relative
        metadata[relative] = {"bytes": path.stat().st_size, "sha256": digest_file(path)}
    spec = {
        "schema_version": 1,
        "source_package": str(remote_package),
        "corrected_repo": str(remote_repo),
        "original_source_commit": "1c67b7af1928f0fdec7e2a9cb5abae631a9e00cb",
        "original_source_tree_sha256": "a27bfdcbcc29b5326c2ea088188097d85061d39ddac761c5f40cde956ccf23fc",
        "baseline_rule": "falahatgar2017_experimental",
        "baseline_version": "robust-ad-v1",
        "baseline_module_sha256": digest_file(local_repo / "src/lsa/alt/baselines.py"),
        "required_runtime": {"python": "3.13.15", "numpy": "2.5.1", "scipy": "1.18.0"},
        "require_campaign_scope": True,
        "metadata_pins": metadata,
        "trial_pins": trials,
        "scope": {"primary_cells": 88, "power_cells": 11, "trials_per_cell": 1000,
                  "full_trial_rows": 99000, "method_rows": 902, "paired_rows": 3773,
                  "recomputed_ad_paired_rows": 803, "blocks_per_cell": 10},
        "input_policy": "Read exact saved target/count arrays and original losses; never regenerate samples or reevaluate LSA.",
        "output_policy": "Fresh derived outputs only; original accepted artifacts remain unchanged.",
    }
    write_json(output, spec)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-package", required=True)
    parser.add_argument("--remote-package", required=True)
    parser.add_argument("--local-repo", required=True)
    parser.add_argument("--remote-repo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    make(args.local_package, args.remote_package, args.local_repo, args.remote_repo, args.output)


if __name__ == "__main__":
    main()
