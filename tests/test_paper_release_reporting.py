"""Current-paper reporting: saved-value fidelity and independent failure checks."""

import csv
import gzip
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_comparisons as export
import paper_render as render
import paper_results as data


@pytest.fixture(scope="module")
def families():
    return data.load_benchmarks(ROOT / "results")


def test_release_coverage_and_finite_robust_ad_accounting(families):
    counts = []
    for summary, blocks, reasons in families.values():
        cells = [cell for ns in summary["targets"].values() for cell in ns.values()]
        counts.append(
            (
                len(cells),
                sum(len(c["methods"]) for c in cells),
                sum(len(c["paired_differences"]) for c in cells),
            )
        )
        assert len(blocks["blocks"]) == 10
        assert all(b["trials"] == 100 for b in blocks["blocks"])
        for cell in cells:
            ad = cell["methods"]["absolute_discounting"]
            assert ad["status"] == "finite" and ad["counts"] == {
                "finite": 1000, "infinite": 0, "undefined": 0,
            }
            assert math.isfinite(ad["mean_bits"]) and ad["mean_bits"] >= 0
            assert math.isfinite(ad["se_bits"]) and ad["se_bits"] >= 0
            for name, pair in cell["paired_differences"].items():
                if "absolute_discounting" in name:
                    assert pair["status"] == "finite" and pair["counts"] == {
                        "finite": 1000, "infinite": 0, "undefined": 0,
                    }
        for block in blocks["blocks"]:
            for ns in block["targets"].values():
                for cell in ns.values():
                    ad = cell["methods"]["absolute_discounting"]
                    assert ad["status"] == "finite" and ad["counts"] == {
                        "finite": 100, "infinite": 0, "undefined": 0,
                    }
    assert counts == [(88, 792, 3256), (11, 110, 517)]


def test_reference_correction_records_rule_and_reuse_provenance(families):
    for summary, blocks, _ in families.values():
        correction = summary["correction"]
        assert blocks["correction"] == correction
        assert correction["method"] == "absolute_discounting"
        assert correction["rule"] == "falahatgar2017_experimental"
        assert correction["version"] == "robust-ad-v1"
        assert correction["lsa_reevaluated"] is False
        assert correction["samples_regenerated"] is False
        source_hash = correction["baseline_module_sha256"]
        assert len(source_hash) == 64
        assert all(c in "0123456789abcdef" for c in source_hash)


def test_precise_large_n_paired_and_block_evidence(families):
    summary, blocks, _ = families["benchmark_primary"]
    key = "dirichlet_concentration_mixture_minus_lsa_depth_mixture"
    cell = summary["targets"]["step"]["20000"]["paired_differences"][key]
    assert cell["mean_bits"] == pytest.approx(-0.06166118383322384, rel=0, abs=1e-15)
    assert cell["se_bits"] == pytest.approx(0.00003888521123182404, rel=0, abs=1e-16)
    assert all(
        b["targets"]["step"]["20000"]["paired_differences"][key]["mean_bits"] < 0
        for b in blocks["blocks"]
    )
    # A paired SE uses covariance and is not the independent marginal formula.
    zipf = summary["targets"]["zipf_1"]["20000"]
    marg = zipf["methods"]
    naive = math.hypot(
        marg["good_turing_hybrid"]["se_bits"], marg["lsa_depth_mixture"]["se_bits"]
    )
    paired = zipf["paired_differences"]["good_turing_hybrid_minus_lsa_depth_mixture"][
        "se_bits"
    ]
    assert not math.isclose(paired, naive, rel_tol=0.01)


def test_export_keeps_all_saved_values_and_reasons(tmp_path, families):
    export.export_rows(families, tmp_path)
    with (tmp_path / "methods-all.tsv").open() as f:
        methods = list(csv.DictReader(f, delimiter="\t"))
    with (tmp_path / "paired-differences-all.tsv").open() as f:
        pairs = list(csv.DictReader(f, delimiter="\t"))
    assert len(methods) == 902 and len(pairs) == 3773
    for row in [*methods, *pairs]:
        saved = families[row["experiment"]][0]["targets"][row["target"]][row["n"]]
        expected = (
            saved["methods"][row["method"]]
            if "method" in row
            else saved["paired_differences"][row["comparison"]]
        )
        assert row["status"] == expected["status"]
        for k in ("mean_bits", "se_bits"):
            assert (float(row[k]) if row[k] else None) == expected[k]
        assert sum(int(row[s + "_trials"]) for s in data.STATUSES) == 1000
    for row in pairs:
        assert sum(json.loads(row["recorded_reasons_json"]).values()) == int(
            row["undefined_trials"]
        )


def test_table_is_generated_from_data_and_preserves_printed_ties(tmp_path, families):
    summary = deepcopy(families["benchmark_primary"][0])
    config = summary["config"]
    render.write_loss_table(
        summary,
        tmp_path / "original",
        n=1000,
        targets=config["targets"],
        methods=config["methods"],
        label="tab:bench",
    )
    original = (tmp_path / "original.tex").read_text()
    assert "0.0139" in original  # Unchanged LSA entry.
    assert r"$\infty^{*}$" not in original and r"n.a.$^{*}$" not in original
    summary["targets"]["zipf_5"]["1000"]["methods"]["lsa_depth_mixture"][
        "mean_bits"
    ] = 0.123456
    render.write_loss_table(
        summary,
        tmp_path / "changed",
        n=1000,
        targets=config["targets"],
        methods=config["methods"],
        label="tab:bench",
    )
    changed = (tmp_path / "changed.tex").read_text()
    assert "0.1235" in changed and original != changed
    assert all(
        a == b
        for a, b in zip(original.splitlines(), changed.splitlines())
        if not a.startswith("Zipf $\\alpha=5$")
    )


def test_factorial_uses_slope_one_offset_not_free_intercept(tmp_path):
    fits = data.read(ROOT / "results/data/factorial-fits.json")
    render.verify_fits(data.read(ROOT / "results/data/factorial.json.gz"), fits)
    render.factorial_tables(fits, tmp_path / "factorial.tex")
    first = (
        (tmp_path / "factorial.tex").read_text().split("1.5 & ", 1)[1].splitlines()[0]
    )
    assert first.startswith("0.87 & 0.76 & 0.58 & -2.73 & -2.17 & -1.34")
    changed = deepcopy(fits)
    changed["alphabet"][2]["slope_one_offset_bits"] = changed["alphabet"][2][
        "offset_bits"
    ]
    with pytest.raises(ValueError, match="fit disagrees"):
        render.verify_fits(data.read(ROOT / "results/data/factorial.json.gz"), changed)


def test_construction_uses_all_saved_layers(tmp_path):
    draw = data.read(ROOT / "results/data/construction.json")
    render.construction(
        draw, ROOT / "results/templates/construction.tex", tmp_path / "figure.tex"
    )
    assert "8/9.18" in (tmp_path / "figure.tex").read_text()
    changed = deepcopy(draw)
    changed["layers"][2][0] *= 2
    with pytest.raises(ValueError):
        render.construction(
            changed, ROOT / "results/templates/construction.tex", tmp_path / "bad.tex"
        )


def test_validation_upward_rounding_and_units(tmp_path):
    validation = data.read(ROOT / "results/data/validation.json")
    rows = render.validation_table(validation, tmp_path / "validation.tex")
    assert len(rows) == 13 and sum(len(r["values"]) for r in rows) == 17
    assert render.upward_tex(3.330181392212002e-9) == r"3.34\times10^{-9}"
    assert render.upward_tex(1.4193360736313249e-13) == r"1.42\times10^{-13}"
    assert "bits/token" in (tmp_path / "validation.tex").read_text()
    assert (
        validation["metrics"]["sealed_store.precision_qualified_cases"]["value"] == 355
    )
    assert validation["qualification"]["computational_budget_ulps"] == 4


def test_public_data_has_no_private_execution_metadata():
    manifest = data.verify_bundle(ROOT / "results")
    assert (
        len(manifest["paper_map"]["figures"]) == 4
        and len(manifest["paper_map"]["tables"]) == 6
    )
    assert sum(len(t["labels"]) for t in manifest["paper_map"]["tables"]) == 7
    for name in manifest["files"]:
        p = ROOT / "results" / name
        text = (
            gzip.decompress(p.read_bytes()).decode()
            if p.suffix == ".gz"
            else p.read_text()
        )
        for forbidden in [
            "/Users/",
            "/scratch/",
            "source_commit",
            "run_directory",
            "old_paper",
        ]:
            assert forbidden not in text, (name, forbidden)


def test_checksum_and_missing_cell_fail_closed(tmp_path, families):
    import shutil

    copy = tmp_path / "results"
    shutil.copytree(ROOT / "results", copy)
    p = copy / "data/construction.json"
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        data.verify_bundle(copy)
    summary = deepcopy(families["benchmark_primary"][0])
    summary["targets"]["step"].pop("20000")
    with pytest.raises(ValueError, match="sample-size"):
        data.check_targets(summary["targets"], summary["config"], 1000)


@pytest.mark.parametrize("status", ["infinite", "undefined"])
def test_nonfinite_cannot_acquire_finite_mean(families, status):
    summary = deepcopy(families["benchmark_primary"][0])
    cell = summary["targets"]["zipf_2"]["1000"]["methods"]["absolute_discounting"]
    counts = {"finite": 999, "infinite": 0, "undefined": 0}
    counts[status] = 1
    cell.update(status=status, counts=counts, mean_bits=0.1, se_bits=None)
    with pytest.raises(ValueError, match="nonfinite"):
        data.check_targets(summary["targets"], summary["config"], 1000)


def test_new_output_rejects_existing_or_symlink(tmp_path):
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(ValueError):
        data.new_output(out, [])
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError):
        data.new_output(link, [])


def test_recomputed_run_export_verifies_real_manifests_and_trial_hashes(tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    from lsa.alt.artifacts import Run

    root = tmp_path / "scientific"
    root.mkdir()
    repository = tmp_path / "source"
    repository.mkdir()
    (repository / "pyproject.toml").write_text(
        '[project]\nname="example"\nversion="1"\n'
    )
    for number, family in enumerate(data.EXPERIMENTS):
        run_root = root / family
        path = run_root / "data"
        config = {
            "methods": ["add_one", "kt"],
            "depths": [1],
            "targets": ["uniform"],
            "n_values": [3],
            "trials": 2,
            "seed": number,
            "sample_set_id": family,
        }

        def item(mean, trials):
            return {
                "status": "finite",
                "mean_bits": mean,
                "se_bits": 0.0 if trials > 1 else None,
                "trials": trials,
                "counts": {"finite": trials, "infinite": 0, "undefined": 0},
            }

        def targets(trials):
            return {
                "uniform": {
                    "3": {
                        "methods": {
                            "add_one": item(0.2, trials),
                            "kt": item(0.3, trials),
                        },
                        "paired_differences": {"add_one_minus_kt": item(-0.1, trials)},
                    }
                }
            }

        summary = {
            "schema_version": 1,
            "config": config,
            "targets": targets(2),
            "nonfinite_policy": "retain",
            "uncertainty": "sample SE",
            "units": "bits",
        }
        blocks = {
            "config": config,
            "diagnostic_only": True,
            "requested_blocks": 10,
            "block_count": 2,
            "pooled_targets": targets(2),
            "blocks": [
                {
                    "block_id": i,
                    "trial_start": i,
                    "trial_stop_exclusive": i + 1,
                    "trials": 1,
                    "targets": targets(1),
                }
                for i in range(2)
            ],
        }
        with Run(
            run_root,
            repo=repository,
            protocol={"experiments": {family: config}},
            experiment=family,
            purpose="validation",
        ):
            path.mkdir()
            trials = [
                {
                    "target_id": "uniform",
                    "n": 3,
                    "trial": i,
                    "sample_set_id": family,
                    "losses": {
                        "add_one": {"status": "finite"},
                        "kt": {"status": "finite"},
                    },
                    "paired_differences": {"add_one_minus_kt": {"status": "finite"}},
                }
                for i in range(2)
            ]
            (path / "trials.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in trials)
            )
            summary["trial_records_sha256"] = data.digest(path / "trials.jsonl")
            data.write(path / "summary.json", summary)
            data.write(path / "block-diagnostics.json", blocks)
    families, hashes = export.load_runs(root)
    assert len(families) == 2 and len(hashes) == 2
    assert (
        families["benchmark_primary"][0]["targets"]["uniform"]["3"]["methods"]["kt"][
            "mean_bits"
        ]
        == 0.3
    )
    (root / "benchmark_primary/data/trials.jsonl").write_text("{}\n")
    with pytest.raises(ValueError):
        export.load_runs(root)


def test_negative_se_and_nan_cannot_be_rendered(families):
    summary = deepcopy(families["benchmark_primary"][0])
    cell = summary["targets"]["step"]["1000"]["methods"]["lsa_depth_mixture"]
    cell["se_bits"] = -1
    with pytest.raises(ValueError, match="standard error"):
        data.check_targets(summary["targets"], summary["config"], 1000)
    cell["se_bits"] = 0
    cell["mean_bits"] = float("nan")
    with pytest.raises(ValueError, match="invalid mean"):
        data.check_targets(summary["targets"], summary["config"], 1000)


def test_figures_preserve_zero_nonfinite_and_uncertainty_conventions(
    tmp_path, families
):
    import matplotlib

    matplotlib.use("Agg")
    summary = data.read(ROOT / "results/data/spectrum.json.gz")
    spectrum = render.spectrum(summary, tmp_path / "spectrum.pdf")
    assert len(spectrum["omitted_exact_zeros"]) == 3
    assert len(spectrum["plotted_points"]) == 360
    assert all(p["mean_bits"] > 0 for p in spectrum["plotted_points"])
    assert {p["d"] for p in spectrum["omitted_exact_zeros"]} == {1000, 10000, 1000000}
    # The reported near-zero floating residuals are retained as data, not plotted.
    assert (
        spectrum["omitted_exact_zeros"][2]["saved_mean_bits"] == 5.901057420487632e-12
    )
    primary = families["benchmark_primary"][0]
    details = render.plot_learning_curves(
        primary,
        tmp_path / "learning.pdf",
        targets=primary["config"]["targets"],
        methods=primary["config"]["methods"],
        uniform_floor=1e-14,
    )
    assert details["nonfinite_points"] == []
    assert len(details["uniform_floor_points"]) == 15
    assert details["zero_points_outside_uniform"] == []
    assert "not confidence intervals" in details["standard_errors"]


def synthetic_cell(status="finite", mean=0.2):
    counts = {"finite": 1000, "infinite": 0, "undefined": 0}
    if status != "finite":
        counts["finite"], counts[status] = 999, 1
    return {
        "status": status, "trials": 1000, "counts": counts,
        "mean_bits": mean if status == "finite" else None,
        "se_bits": 0.01 if status == "finite" else None,
    }


def test_tables_preserve_synthetic_infinite_and_undefined_results(tmp_path):
    summary = {"config": {"sample_set_id": "synthetic"}, "targets": {
        "uniform": {"1000": {"methods": {
            "absolute_discounting": synthetic_cell("infinite"),
            "kt": synthetic_cell(),
        }}},
        "step": {"1000": {"methods": {
            "absolute_discounting": synthetic_cell(),
            "kt": synthetic_cell("undefined"),
        }}},
    }}
    render.write_loss_table(summary, tmp_path / "nonfinite", n=1000,
                            targets=["uniform", "step"],
                            methods=["absolute_discounting", "kt"], label="synthetic")
    text = (tmp_path / "nonfinite.tex").read_text()
    assert r"$\infty^{*}$" in text and r"n.a.$^{*}$" in text
    with (tmp_path / "nonfinite.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    nonfinite = [row for row in rows if row["status"] != "finite"]
    assert {row["status"] for row in nonfinite} == {"infinite", "undefined"}
    assert all(row["mean_bits"] == row["se_bits"] == "" for row in nonfinite)


@pytest.mark.parametrize("nonfinite", [False, True])
def test_synthetic_curves_break_nonfinite_lines_and_condition_the_note(
    tmp_path, monkeypatch, nonfinite,
):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    summary = {"config": {"n_values": [1000, 2000, 3000]}, "targets": {"step": {}}}
    for i, n in enumerate(summary["config"]["n_values"]):
        summary["targets"]["step"][str(n)] = {"methods": {
            "absolute_discounting": synthetic_cell(mean=0.2 + i / 10),
            "kt": synthetic_cell(mean=0.3 + i / 10),
        }}
    if nonfinite:
        summary["targets"]["step"]["2000"]["methods"]["absolute_discounting"] = synthetic_cell("infinite")
        summary["targets"]["step"]["3000"]["methods"]["kt"] = synthetic_cell("undefined")
    captured = {}

    def capture(figure, *args, **kwargs):
        captured["footer"] = " ".join(text.get_text() for text in figure.texts)
        captured["lines"] = [line.get_ydata() for line in figure.axes[0].lines]

    monkeypatch.setattr(Figure, "savefig", capture)
    details = render.plot_learning_curves(summary, tmp_path / "synthetic.pdf",
                                         targets=["step"],
                                         methods=["absolute_discounting", "kt"],
                                         uniform_floor=1e-14)
    assert ("Nonfinite cases marked" in captured["footer"]) is nonfinite
    if nonfinite:
        assert {(p["method"], p["n"], p["status"]) for p in details["nonfinite_points"]} == {
            ("absolute_discounting", 2000, "infinite"), ("kt", 3000, "undefined"),
        }
        assert any(len(y) == 3 and math.isnan(y[1]) for y in captured["lines"])
    else:
        assert details["nonfinite_points"] == []


def test_posterior_intervals_describe_mass_not_sampling_confidence(families):
    posterior = families["benchmark_primary"][0]["targets"]["zipf_1.5"]["1000"][
        "mean_posteriors"
    ]["depth"]
    result = render.posterior_summary(
        posterior["indices"],
        posterior["weights"],
        mass=0.95,
        broad_min_depths=5,
        individual_min_weight=0.05,
    )
    assert result["interval"] == [12, 19]
    assert result["enclosed_mass"] == pytest.approx(0.959790814244331, abs=1e-9)
    assert result["kind"] == "interval"
