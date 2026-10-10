"""Saved scientific result checks shared by the public reporting commands."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath

EXPERIMENTS = ("benchmark_primary", "benchmark_powers")
STATUSES = ("finite", "infinite", "undefined")
ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(config):
    methods = config["methods"]
    result = {
        f"{a}_minus_{b}": (a, b)
        for i, a in enumerate(methods)
        for b in methods[i + 1 :]
    }
    if {"power_mixture", "lsa_depth_mixture"} <= set(methods):
        result["power_mixture_minus_lsa_depth_mixture"] = (
            "power_mixture",
            "lsa_depth_mixture",
        )
    if (
        "lsa_depth_mixture" in methods
        and 0 in config["depths"]
        and len(config["depths"]) > 1
    ):
        result["lsa_depth_mixture_minus_positive_depth_mixture"] = (
            "lsa_depth_mixture",
            "positive_depth_mixture",
        )
    return result


def check_cell(cell, trials, config, pair_names):
    require(
        set(cell["methods"]) == set(config["methods"]), "incomplete method coverage"
    )
    require(set(cell["paired_differences"]) == pair_names, "incomplete paired coverage")
    for item in (*cell["methods"].values(), *cell["paired_differences"].values()):
        require(item["trials"] == trials, "incorrect summary trial count")
        require(set(item["counts"]) == set(STATUSES), "missing outcome counts")
        require(
            all(isinstance(v, int) and v >= 0 for v in item["counts"].values()),
            "invalid outcome counts",
        )
        require(sum(item["counts"].values()) == trials, "incomplete outcome counts")
        expected = (
            "undefined"
            if item["counts"]["undefined"]
            else "infinite"
            if item["counts"]["infinite"]
            else "finite"
        )
        require(item["status"] == expected, "inconsistent summary outcome status")
        if expected != "finite":
            require(
                item["mean_bits"] is None and item["se_bits"] is None,
                "nonfinite outcome has a finite summary",
            )
        else:
            require(
                isinstance(item["mean_bits"], (int, float))
                and math.isfinite(item["mean_bits"]),
                "finite summary has invalid mean",
            )
            require(
                (trials == 1 and item["se_bits"] is None)
                or (
                    trials > 1
                    and isinstance(item["se_bits"], (int, float))
                    and math.isfinite(item["se_bits"])
                    and item["se_bits"] >= 0
                ),
                "finite summary has invalid standard error",
            )


def check_targets(targets, config, trials):
    require(set(targets) == set(config["targets"]), "incomplete target coverage")
    pair_names = set(pairs(config))
    for target in config["targets"]:
        require(
            set(targets[target]) == {str(n) for n in config["n_values"]},
            "incomplete sample-size coverage",
        )
        for n in config["n_values"]:
            check_cell(targets[target][str(n)], trials, config, pair_names)


def read_outcomes(root, config, summary):
    """Count recorded statuses/reasons, retaining no predictions or diagnostics."""
    expected_pairs = pairs(config)
    counts, reasons, seen = {}, {}, set()
    start = config.get("trial_start", 0)
    for target in config["targets"]:
        for n in config["n_values"]:
            counts[target, n] = {name: Counter() for name in expected_pairs}
            reasons[target, n] = {name: Counter() for name in expected_pairs}
    with (root / "data/trials.jsonl").open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["target_id"], row["n"])
            identity = (*key, row["trial"])
            require(
                key in counts and start <= row["trial"] < start + config["trials"],
                "unexpected raw trial identity",
            )
            require(identity not in seen, "duplicate raw trial identity")
            seen.add(identity)
            require(
                row["sample_set_id"] == config["sample_set_id"], "wrong raw sample set"
            )
            require(set(row["losses"]) == set(config["methods"]), "missing raw method")
            require(
                set(row["paired_differences"]) == set(expected_pairs),
                "missing raw pair",
            )
            for name, item in row["paired_differences"].items():
                require(item["status"] in STATUSES, "invalid raw pair status")
                counts[key][name][item["status"]] += 1
                if item.get("reason") is not None:
                    reasons[key][name][item["reason"]] += 1
    require(
        len(seen) == len(counts) * config["trials"], "incomplete raw trial coverage"
    )
    for (target, n), items in counts.items():
        for name, statuses in items.items():
            stored = summary["targets"][target][str(n)]["paired_differences"][name][
                "counts"
            ]
            require(
                {s: statuses[s] for s in STATUSES} == stored,
                "saved pair counts disagree with raw records",
            )
    return reasons


def validate_blocks(blocks, config, summary, requested):
    require(
        blocks["config"] == config and blocks["diagnostic_only"] is True,
        "block diagnostics configuration mismatch",
    )
    effective = min(requested, config["trials"])
    require(
        blocks["requested_blocks"] == requested and blocks["block_count"] == effective,
        "block count differs from protocol",
    )
    require(len(blocks["blocks"]) == effective, "missing diagnostic blocks")
    pooled = {
        target: {
            n: {key: cell[key] for key in ("methods", "paired_differences")}
            for n, cell in ns.items()
        }
        for target, ns in summary["targets"].items()
    }
    require(
        blocks["pooled_targets"] == pooled, "block pooled results differ from summary"
    )
    start = config.get("trial_start", 0)
    size, extra = divmod(config["trials"], effective)
    for i, block in enumerate(blocks["blocks"]):
        length = size + (i < extra)
        require(
            (
                block["block_id"],
                block["trial_start"],
                block["trial_stop_exclusive"],
                block["trials"],
            )
            == (i, start, start + length, length),
            "incorrect fixed block partition",
        )
        check_targets(block["targets"], config, length)
        start += length


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_bundle(root):
    root = Path(root)
    require(not root.is_symlink(), "results root cannot be a symlink")
    manifest = read(root / "manifest.json")
    require(manifest["format"] == "lsa-paper-results-v1", "unknown results format")
    files = manifest["files"]
    for name, entry in files.items():
        parts = PurePosixPath(name).parts
        require(
            parts
            and not name.startswith("/")
            and ".." not in parts
            and "\\" not in name,
            "unsafe results path",
        )
        p = root / name
        require(
            not any(q.is_symlink() for q in [p, *p.parents] if q != root.parent),
            "symlink in results input",
        )
        require(
            p.is_file()
            and p.stat().st_size == entry["bytes"]
            and digest(p) == entry["sha256"],
            f"results checksum mismatch: {name}",
        )
    actual = {
        p.relative_to(root).as_posix()
        for d in ("data", "templates")
        for p in (root / d).rglob("*")
        if p.is_file()
    }
    require(actual == set(files), "results inventory differs from manifest")
    return manifest


def load_benchmarks(root):
    root = Path(root)
    verify_bundle(root)
    reasons = read(root / "data/paired-reasons.json.gz")
    result = {}
    for family in EXPERIMENTS:
        summary = read(root / f"data/{family}.json.gz")
        blocks = read(root / f"data/{family}-blocks.json.gz")
        config = summary["config"]
        check_targets(summary["targets"], config, config["trials"])
        validate_blocks(blocks, config, summary, 10)
        require(
            set(reasons[family]) == set(config["targets"]),
            "missing target reason counts",
        )
        for target in config["targets"]:
            require(
                set(reasons[family][target]) == {str(n) for n in config["n_values"]},
                "missing cell reasons",
            )
            for n in config["n_values"]:
                require(
                    set(reasons[family][target][str(n)]) == set(pairs(config)),
                    "missing pair reasons",
                )
                for name, counts in reasons[family][target][str(n)].items():
                    item = summary["targets"][target][str(n)]["paired_differences"][
                        name
                    ]
                    require(
                        all(isinstance(v, int) and v > 0 for v in counts.values()),
                        "invalid reason count",
                    )
                    require(
                        sum(counts.values()) == item["counts"]["undefined"],
                        "nonfinite reason count mismatch",
                    )
        result[family] = (summary, blocks, reasons[family])
    a, b = [result[f][0]["config"] for f in EXPERIMENTS]
    require(
        a["seed"] != b["seed"] and a["sample_set_id"] != b["sample_set_id"],
        "sample sets must differ",
    )
    return result


def new_output(path, inputs):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), "output must be new")
    resolved = path.resolve()
    for item in inputs:
        require(
            not resolved.is_relative_to(Path(item).resolve()),
            "output cannot be inside its input bundle",
        )
    path.mkdir(parents=True, exist_ok=False)
    return path


def output_manifest(root, *, inputs, scope):
    root = Path(root)
    data = {
        "format": "lsa-paper-report-v1",
        "scope": scope,
        "inputs": inputs,
        "files": {
            p.relative_to(root).as_posix(): {
                "sha256": digest(p),
                "bytes": p.stat().st_size,
            }
            for p in sorted(root.rglob("*"))
            if p.is_file()
        },
    }
    write(root / "manifest.json", data)
    return data
