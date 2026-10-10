"""Read freshly recomputed immutable Runs for the current-paper renderer.

Only descriptive transformations of saved observations happen here. Numerical
admission remains the responsibility of alt_admission.py; this is no certificate.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from export_comparisons import load_runs
from paper_results import ROOT, digest, read, require

FAMILIES = (
    "architecture",
    "benchmark_primary",
    "benchmark_powers",
    "spectrum",
    "factorial",
    "depth_scaling",
    "bible",
    "bible_secondary",
)
SUITES = {
    "kernel": "kernel-validation.json",
    "prior": "prior-validation.json",
    "depth": "depth-validation.json",
    "power": "power-validation.json",
    "chain": "bible-chain-validation.json",
    "sealed_store": "sealed-store-validation.json",
    "sealed_profile": "sealed-profile-validation.json",
    "regressions": None,
}


def verify_run_at(path, experiment):
    sys.path.insert(0, str(ROOT / "src"))
    from lsa.alt.artifacts import verify_run

    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), "missing or unsafe input Run")
    require(not any(p.is_symlink() for p in path.rglob("*")), "symlink in input Run")
    record = verify_run(path, require_complete=True)
    require(record["experiment"] == experiment, f"wrong experiment in {path.name}")
    return record


def fit_factorial(summary):
    config, rows = summary["config"], summary["cells"]
    alphabet, data = [], []
    for alpha in config["alphas"]:
        for n in config["table_ns"]:
            group = sorted(
                (r for r in rows if r["alpha"] == alpha and r["n"] == n),
                key=lambda r: r["d"],
            )
            x = np.array(
                [r["naming_mean_bits"] / r["expected_discovered"] for r in group]
            )
            y = np.array(
                [
                    r["n"] * r["regret_mean_bits"][0] / r["expected_discovered"]
                    for r in group
                ]
            )
            slope, offset = np.polyfit(x, y, 1)
            alphabet.append(
                {
                    "alpha": alpha,
                    "n": n,
                    "slope": float(slope),
                    "offset_bits": float(offset),
                    "slope_one_offset_bits": float(np.mean(y - x)),
                }
            )
        for d in config["table_ds"]:
            group = sorted(
                (r for r in rows if r["alpha"] == alpha and r["d"] == d),
                key=lambda r: r["n"],
            )
            y = np.array([r["regret_mean_bits"][0] for r in group])
            require(np.all(y > 0), "data exponent needs positive saved mean regrets")
            slope, _ = np.polyfit(np.log([r["n"] for r in group]), np.log(y), 1)
            data.append(
                {"alpha": alpha, "d": d, "exponent": float(-slope), "status": "defined"}
            )
    return {"alphabet": alphabet, "data": data}


def clean_bible(summary):
    return {
        "config": {
            k: v
            for k, v in summary["config"].items()
            if k not in ("corpus", "manifest")
        },
        "rows": [
            {k: v for k, v in row.items() if k != "diagnostics"}
            for row in summary["rows"]
        ],
    }


def tokenization_controls(primary, secondary):
    rows = []
    a = [r for r in primary["rows"] if r["n"] in primary["config"]["prefixes"]]
    b = [r for r in secondary["rows"] if r["n"] in secondary["config"]["prefixes"]]
    require(len(a) == len(b), "tokenization prefix coverage differs")
    for left, right in zip(a, b, strict=True):
        for method in ("add_one", "kt", "ristad", "good_turing_hybrid"):
            rows.append(
                {
                    "primary_n": left["n"],
                    "secondary_n": right["n"],
                    "method": method,
                    "difference_bits": left["redundancy_bits"][method]
                    - right["redundancy_bits"][method],
                }
            )
    return rows


def extract_validation_values(results, kernel_rows, special, prior_checks):
    """Map actual saved measurement fields to report quantities (no fallback)."""
    values = {}
    store_keys = (
        "cases",
        "nominal_pass_cases",
        "precision_qualified_cases",
        "maximum_low_count_error_nats",
        "maximum_absolute_error_nats",
        "maximum_qualified_arithmetic_error_ulps",
        "maximum_independent_rounding_error_ulps",
        "maximum_scalar_matrix_delta_nats",
        "independent_reference_complete_cases",
    )
    for key in store_keys:
        values["sealed_store." + key] = results["sealed_store"][key]

    def maximum(items):
        items = [abs(float(v)) for v in items if v is not None]
        require(
            items and all(math.isfinite(v) for v in items),
            "missing/nonfinite validation measurement",
        )
        return max(items)

    require(
        len(kernel_rows) == 263, "current table requires the declared263 kernel cases"
    )
    values["kernel.sealed_max_error"] = maximum(
        r["stores"]["sealed_candidate"]["error_nats"] for r in kernel_rows
    )
    values["kernel.sealed_low_count_max_error"] = maximum(
        r["stores"]["sealed_candidate"]["error_nats"]
        for r in kernel_rows
        if r["r"] <= 3
    )
    for key in ("meijer_error_nats", "recursion_error_nats"):
        values["kernel." + key] = maximum(r.get(key) for r in special)
    for prefix in ("direct_simplex", "prior_monte_carlo"):
        values["prior." + prefix] = maximum(
            r["residual"] for r in prior_checks if r["name"].startswith(prefix + ".")
        )
    require(
        len(results["sealed_profile"]["cases"]) == 29
        and len(results["depth"]["assessment"]["cases"]) == 27,
        "current table requires full29/27 profile domains",
    )
    for key in (
        "maximum_codelength_difference_bits_per_token",
        "mixture_codelength_difference_bits_per_token",
        "maximum_augmented_codelength_difference_bits_per_token",
        "maximum_predictive_kl_change_bits",
        "mixture_predictive_kl_change_bits",
        "maximum_raw_mass_error",
    ):
        values["sealed_profile." + key] = maximum(
            c["measurements"].get(key) for c in results["sealed_profile"]["cases"]
        )
    for key in (
        "maximum_codelength_difference_bits_per_token",
        "mixture_codelength_difference_bits_per_token",
        "maximum_predictive_kl_change_bits",
        "mixture_predictive_kl_change_bits",
        "maximum_raw_mass_error",
    ):
        values["depth." + key] = maximum(
            c.get(field, {}).get(key)
            for c in results["depth"]["assessment"]["cases"]
            for field in ("measurements", "supplemental_measurements")
        )
    require(
        len(results["power"]["profiles"]) == 11,
        "current table requires11 power profiles",
    )
    for key in (
        "max_log_evidence_change_bits",
        "max_predictive_kl_change_bits",
        "mixture_predictive_kl_change_bits",
    ):
        values["power." + key] = maximum(
            c["comparison"].get(key) for c in results["power"]["profiles"]
        )
    for key in (
        "final_chain_error_bits",
        "max_chain_error_bits",
        "max_probability_error",
    ):
        values["chain." + key] = abs(results["chain"][key])
    require(
        all(
            isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
            for v in values.values()
        ),
        "invalid validation metric",
    )
    return values


def load_validation(root, definitions, source_tree, expected_engine):
    root = Path(root)
    results = {}
    records = {}
    hashes = {}
    for suite, specfile in SUITES.items():
        path = root / suite
        experiment = (
            "production_regressions"
            if suite == "regressions"
            else "calibration_" + suite
        )
        record = verify_run_at(path, experiment)
        require(
            record["source"]["tree_sha256"] == source_tree,
            "validation and experiment sources differ",
        )
        if specfile:
            require(
                read(path / "protocol.json")["config"]
                == read(ROOT / "experiments/alt2027" / specfile),
                "validation configuration differs from current declared paper domain",
            )
        result = read(path / "result.json")
        require(result["status"] == "passed", f"{suite} validation did not pass")
        records[suite] = record
        results[suite] = result
        hashes["validation/" + suite] = digest(path / "manifest.json")
    require(
        read(root / "chain/data/engine.json") == expected_engine,
        "chain validation and campaign numerical engines differ",
    )
    depth_engines = sorted((root / "depth/data").rglob("default-engine.json"))
    # This suite deliberately tightens only the probability-mass guard.
    depth_expected = dict(expected_engine)
    depth_expected["prediction_tolerance"] = read(root / "depth/protocol.json")[
        "config"
    ]["raw_mass_tolerance"]
    require(
        depth_engines and all(read(path) == depth_expected for path in depth_engines),
        "depth validation and campaign numerical engines differ",
    )
    require(
        records["sealed_store"]["engine"] == expected_engine,
        "store validation and campaign numerical engines differ",
    )
    require(
        records["sealed_profile"]["engine"]["candidate"] == expected_engine,
        "profile validation and campaign numerical engines differ",
    )
    for suite in ("kernel", "prior", "depth", "power", "chain"):
        options = records[suite]["engine"]
        require(
            options["store"]["files_sha256"]
            == expected_engine["store"]["files_sha256"],
            "validation uses a different numerical store",
        )
        require(
            options["store"].get("native_library_sha256")
            == expected_engine["store"].get("native_library_sha256"),
            "validation uses a different native binary",
        )
    kernel_rows = [
        json.loads(line)
        for line in (root / "kernel/data/rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    values = extract_validation_values(
        results,
        kernel_rows,
        read(root / "kernel/data/special-functions.json"),
        read(root / "prior/data/checks.json"),
    )
    require(
        set(values) == set(definitions["metrics"]), "validation metric coverage differs"
    )
    metrics = {
        key: {
            **{k: v for k, v in definitions["metrics"][key].items() if k != "value"},
            "value": value,
        }
        for key, value in values.items()
    }
    for key, metric in metrics.items():
        if key.startswith("sealed_store."):
            metric["domain"] = (
                f"{values['sealed_store.cases']} declared scalar/matrix reader queries; "
                "low-count strata at all L=2..138 and general-count strata through r=1045889; "
                f"direct/refined screening plus{values['sealed_store.independent_reference_complete_cases']} "
                "selected independent45/60-digit references"
            )
    validation = {
        "metrics": metrics,
        "qualification": definitions["qualification"],
        "scope": "Measured finite-domain validation from supplied fresh Runs; reporting is not a new admission certificate.",
    }
    return validation, hashes


def load_fresh(root, validation_root, definitions):
    root = Path(root)
    records = {}
    hashes = {}
    for family in FAMILIES:
        records[family] = verify_run_at(root / family, family)
        hashes[family] = digest(root / family / "manifest.json")
    require(
        len({r["protocol_sha256"] for r in records.values()}) == 1,
        "campaign protocol identities differ",
    )
    require(
        len({r["source"]["tree_sha256"] for r in records.values()}) == 1,
        "campaign sources differ",
    )
    expected_engine = records["benchmark_primary"]["engine"]["depth"]
    for family, record in records.items():
        if family not in ("architecture", "bible_secondary"):
            require(
                record["engine"]["depth"] == expected_engine,
                "campaign numerical engines differ",
            )
    protocol = read(root / "benchmark_primary/protocol.json")
    current = read(ROOT / "experiments/alt2027/protocol.json")
    require(
        protocol == current,
        "current-paper renderer requires the full declared paper protocol",
    )
    families, benchmark_hashes = load_runs(root)
    require(
        all(hashes[k] == v for k, v in benchmark_hashes.items()),
        "benchmark inputs changed",
    )
    data = {}
    for family in ("spectrum", "depth_scaling", "factorial"):
        summary = read(root / family / "data/summary.json")
        require(
            summary["config"] == protocol["experiments"][family],
            f"{family} config differs",
        )
        data[family + ".json.gz"] = summary
    for family in ("bible", "bible_secondary"):
        summary = read(root / family / "data/summary.json")
        require(
            summary["config"] == protocol["experiments"][family],
            f"{family} config differs",
        )
        data["bible.json" if family == "bible" else "bible-secondary.json"] = (
            clean_bible(summary)
        )
    data["tokenization-controls.json"] = tokenization_controls(
        data["bible.json"], data["bible-secondary.json"]
    )
    data["construction.json"] = read(root / "architecture/data/draw.json")
    data["report-settings.json"] = protocol["reports"]
    data["factorial-fits.json"] = fit_factorial(data["factorial.json.gz"])
    data["runtime.json"] = {
        "status": "not_measured",
        "scope": "The five reference timing pilots were not rerun by this campaign and are not reused. Fresh Run intervals have different timing boundaries.",
    }
    if validation_root:
        data["validation.json"], more = load_validation(
            validation_root,
            definitions,
            records["benchmark_primary"]["source"]["tree_sha256"],
            expected_engine,
        )
        hashes.update(more)
    else:
        data["validation.json"] = None
    return families, data, hashes


def recheck_fresh(root, validation_root, hashes):
    for family in FAMILIES:
        verify_run_at(Path(root) / family, family)
        require(
            digest(Path(root) / family / "manifest.json") == hashes[family],
            "fresh campaign changed during report",
        )
    if validation_root:
        for suite in SUITES:
            experiment = (
                "production_regressions"
                if suite == "regressions"
                else "calibration_" + suite
            )
            verify_run_at(Path(validation_root) / suite, experiment)
            require(
                digest(Path(validation_root) / suite / "manifest.json")
                == hashes["validation/" + suite],
                "fresh validation changed during report",
            )
