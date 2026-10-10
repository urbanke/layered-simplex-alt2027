"""Fresh reporting never substitutes included reference measurements."""

import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_fresh as fresh
import paper_results as data
import reproduce_paper as reporting


def fixture_validation():
    definitions = data.read(ROOT / "results/data/validation.json")
    result = {suite: {"status": "passed"} for suite in fresh.SUITES}
    values = {k: v["value"] for k, v in definitions["metrics"].items()}
    result["sealed_store"].update(
        {
            k.split(".", 1)[1]: v
            for k, v in values.items()
            if k.startswith("sealed_store.")
        }
    )
    result["sealed_profile"]["cases"] = [
        {
            "measurements": {
                k.split(".", 1)[1]: v
                for k, v in values.items()
                if k.startswith("sealed_profile.")
            }
        }
        for _ in range(29)
    ]
    result["depth"]["assessment"] = {
        "cases": [
            {
                "measurements": {
                    k.split(".", 1)[1]: v
                    for k, v in values.items()
                    if k.startswith("depth.")
                },
                "supplemental_measurements": {},
            }
            for _ in range(27)
        ]
    }
    result["power"]["profiles"] = [
        {
            "comparison": {
                k.split(".", 1)[1]: v
                for k, v in values.items()
                if k.startswith("power.")
            }
        }
        for _ in range(11)
    ]
    result["chain"].update(
        {k.split(".", 1)[1]: v for k, v in values.items() if k.startswith("chain.")}
    )
    rows = [
        {
            "r": 0 if i == 0 else 4,
            "stores": {
                "sealed_candidate": {
                    "error_nats": -values["kernel.sealed_low_count_max_error"]
                    if i == 0
                    else values["kernel.sealed_max_error"]
                }
            },
        }
        for i in range(263)
    ]
    special = [
        {
            "meijer_error_nats": values["kernel.meijer_error_nats"],
            "recursion_error_nats": values["kernel.recursion_error_nats"],
        }
    ]
    prior = [
        {"name": name + ".example", "residual": values["prior." + name]}
        for name in ("direct_simplex", "prior_monte_carlo")
    ]
    return definitions, result, rows, special, prior


def test_validation_mapping_uses_fresh_changed_measurements():
    definitions, result, rows, special, prior = fixture_validation()
    result["sealed_profile"]["cases"][0]["measurements"][
        "maximum_codelength_difference_bits_per_token"
    ] = 8.99e-8
    result["depth"]["assessment"]["cases"][0]["supplemental_measurements"][
        "maximum_predictive_kl_change_bits"
    ] = 9e-12
    result["chain"]["max_chain_error_bits"] = -8e-11
    values = fresh.extract_validation_values(result, rows, special, prior)
    assert set(values) == set(definitions["metrics"])
    assert (
        values["sealed_profile.maximum_codelength_difference_bits_per_token"] == 8.99e-8
    )
    assert values["depth.maximum_predictive_kl_change_bits"] == 9e-12
    assert values["chain.max_chain_error_bits"] == 8e-11
    assert (
        values["kernel.sealed_low_count_max_error"]
        == definitions["metrics"]["kernel.sealed_low_count_max_error"]["value"]
    )
    # Missing measurement remains an error; reference values never fill the gap.
    with pytest.raises(ValueError, match="missing/nonfinite"):
        fresh.extract_validation_values(result, rows, [{}], prior)
    with pytest.raises(ValueError, match="263"):
        fresh.extract_validation_values(result, rows[:-1], special, prior)


def test_fresh_descriptive_fits_change_with_saved_observations():
    summary = data.read(ROOT / "results/data/factorial.json.gz")
    fits = fresh.fit_factorial(summary)
    expected = data.read(ROOT / "results/data/factorial-fits.json")
    for group in ("alphabet", "data"):
        for actual_row, expected_row in zip(fits[group], expected[group], strict=True):
            for key, value in expected_row.items():
                assert actual_row[key] == (
                    pytest.approx(value, rel=1e-12, abs=1e-12)
                    if isinstance(value, (int, float))
                    else value
                )
    changed = deepcopy(summary)
    for cell in changed["cells"]:
        cell["regret_mean_bits"][0] *= 2
    actual = fresh.fit_factorial(changed)
    assert actual["alphabet"][0]["slope"] == pytest.approx(
        2 * fits["alphabet"][0]["slope"]
    )
    assert (
        actual["alphabet"][0]["slope_one_offset_bits"]
        != fits["alphabet"][0]["slope_one_offset_bits"]
    )
    assert actual["data"][0]["exponent"] == pytest.approx(fits["data"][0]["exponent"])


def test_real_validation_runs_reject_failed_status_and_mixed_source(tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    from lsa.alt.artifacts import Run

    definitions, result, rows, special, prior = fixture_validation()
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "pyproject.toml").write_text('[project]\nname="fixture"\n')
    root = tmp_path / "checks"
    source_tree = None
    engine = {
        "mode": "store",
        "prediction_tolerance": 0.001,
        "store": {
            "files_sha256": {"level.bin": "test-data"},
            "native_library_sha256": "test-native",
        },
    }
    for suite, spec in fresh.SUITES.items():
        experiment = (
            "production_regressions"
            if suite == "regressions"
            else "calibration_" + suite
        )
        protocol = (
            {"config": data.read(ROOT / "experiments/alt2027" / spec)}
            if spec
            else {"checks": []}
        )
        with Run(
            root / suite,
            repo=repository,
            protocol=protocol,
            experiment=experiment,
            purpose="validation",
            engine_identity={"candidate": engine}
            if suite == "sealed_profile"
            else engine,
        ) as run:
            source_tree = run.record["source"]["tree_sha256"]
            data.write(run.path / "result.json", result[suite])
            (run.path / "data").mkdir()
            if suite == "kernel":
                import json

                (run.path / "data/rows.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows)
                )
                data.write(run.path / "data/special-functions.json", special)
            if suite == "prior":
                data.write(run.path / "data/checks.json", prior)
            if suite == "chain":
                data.write(run.path / "data/engine.json", engine)
            if suite == "depth":
                (run.path / "data/shard-0").mkdir()
                depth_engine = dict(engine)
                depth_engine["prediction_tolerance"] = protocol["config"][
                    "raw_mass_tolerance"
                ]
                data.write(run.path / "data/shard-0/default-engine.json", depth_engine)
    measured, hashes = fresh.load_validation(root, definitions, source_tree, engine)
    assert measured["metrics"] == definitions["metrics"] and len(hashes) == 8
    wrong_engine = deepcopy(engine)
    wrong_engine["store"]["native_library_sha256"] = "different-native"
    with pytest.raises(ValueError, match="engines differ"):
        fresh.load_validation(root, definitions, source_tree, wrong_engine)
    with pytest.raises(ValueError, match="sources differ"):
        fresh.load_validation(root, definitions, "wrong", engine)
    # A complete immutable Run may validly record a failed numerical assessment;
    # reporting must reject that status even though every file checksum matches.
    failed = tmp_path / "failed"
    import shutil

    shutil.copytree(root, failed)
    shutil.rmtree(failed / "kernel")
    with Run(
        failed / "kernel",
        repo=repository,
        protocol={
            "config": data.read(ROOT / "experiments/alt2027/kernel-validation.json")
        },
        experiment="calibration_kernel",
        purpose="validation",
    ) as run:
        data.write(run.path / "result.json", {"status": "failed"})
    with pytest.raises(ValueError, match="did not pass"):
        fresh.load_validation(failed, definitions, source_tree, engine)


def test_fresh_render_routes_all_measurements_and_never_reuses_runtime(
    tmp_path, monkeypatch
):
    families = data.load_benchmarks(ROOT / "results")
    families = deepcopy(families)
    families["benchmark_primary"][0]["targets"]["zipf_5"]["1000"]["methods"][
        "lsa_depth_mixture"
    ]["mean_bits"] = 0.123456
    names = [
        "construction.json",
        "spectrum.json.gz",
        "depth_scaling.json.gz",
        "factorial.json.gz",
        "factorial-fits.json",
        "bible.json",
        "bible-secondary.json",
        "tokenization-controls.json",
        "report-settings.json",
        "validation.json",
    ]
    values = {n: data.read(ROOT / "results/data" / n) for n in names}
    values["validation.json"] = None
    values["runtime.json"] = {"status": "not_measured"}
    # Integrity and schema loading have separate actual-Run tests above; here
    # exercise the full renderer with distinguishable fresh numerical values.
    monkeypatch.setattr(
        fresh,
        "load_fresh",
        lambda *args: (families, values, {"benchmark_primary": "fresh-input"}),
    )
    monkeypatch.setattr(fresh, "recheck_fresh", lambda *args: None)
    out = tmp_path / "render"
    result = reporting.reproduce(ROOT / "results", out, runs=tmp_path / "new-runs")
    assert "0.1235" in (out / "tables/primary.tex").read_text()
    assert not (out / "tables/validation.tex").exists()
    assert result["numbered_tables"] == 5 and result["table_labels"] == 6
    controls = data.read(out / "in-text-controls.json")
    assert controls["validation"] is None and controls["runtime"] == {
        "status": "not_measured"
    }
    assert (
        data.read(out / "manifest.json")["inputs"]["benchmark_primary"] == "fresh-input"
    )
    assert data.read(out / "paper-map.json")["input_mode"] == "fresh_runs"


def test_cli_rejects_ambiguous_reference_and_fresh_inputs():
    with pytest.raises(SystemExit):
        reporting.main(["--check", "--runs", "fresh"])
    with pytest.raises(SystemExit):
        reporting.main(["--validation-root", "checks", "--out", "unused"])
