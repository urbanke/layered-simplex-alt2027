from __future__ import annotations

import copy
import gzip
import importlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import recompute_saved_ad as runner
BASELINE = importlib.import_module("lsa.alt.baselines")


def pin(path):
    return {"bytes": path.stat().st_size, "sha256": runner.digest_file(path)}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n")


def independent_summary(losses):
    counts = {status: sum(x["status"] == status for x in losses) for status in runner.STATUSES}
    status = "undefined" if counts["undefined"] else "infinite" if counts["infinite"] else "finite"
    values = [x["bits"] for x in losses]
    return {"status": status, "trials": len(losses), "counts": counts,
            "mean_bits": statistics.mean(values) if status == "finite" else None,
            "se_bits": statistics.stdev(values) / math.sqrt(len(values))
            if status == "finite" and len(values) > 1 else None}


def old_ad(counts, target):
    n, d, s = int(sum(counts)), len(counts), int(np.count_nonzero(counts))
    c1, c2 = int(np.count_nonzero(counts == 1)), int(np.count_nonzero(counts == 2))
    if c1 + 2 * c2 == 0:
        return {"status": "undefined", "bits": None, "reason": "c1_and_c2_are_zero"}
    delta = c1 / (c1 + 2 * c2)
    if s == d and delta > 0:
        return {"status": "undefined", "bits": None, "reason": "no_unseen_symbol_for_reserved_mass"}
    q = np.zeros(d)
    q[counts > 0] = (counts[counts > 0] - delta) / n
    if s < d:
        q[counts == 0] = delta * s / (n * (d - s))
    if np.any((target > 0) & (q == 0)):
        return {"status": "infinite", "bits": None, "reason": "zero_predictive_probability"}
    return {"status": "finite", "bits": float(sum(p * math.log2(p / qi) for p, qi in zip(target, q))), "reason": None}


@pytest.fixture
def fixture(tmp_path):
    package = tmp_path / "accepted"
    metadata, trials, originals = {}, {}, {}
    count_cases = {3: [[3, 0, 0, 0], [2, 1, 0, 0], [1, 1, 1, 0]],
                   6: [[3, 3, 0, 0], [2, 2, 2, 0], [3, 1, 1, 1]]}
    for exp_index, name in enumerate(runner.EXPERIMENTS):
        root = package / "merged" / name
        methods = ["good_turing_hybrid", "absolute_discounting", "lsa_fixed"]
        if exp_index:
            methods.append("power_mixture")
        config = {"d": 4, "n_values": [3, 6], "trials": 3, "seed": 123 + exp_index,
                  "sample_set_id": "primary" if exp_index == 0 else "powers",
                  "targets": ["uniform", "step"], "methods": methods, "depths": [22],
                  "dirichlet_target_policy": "redraw_per_trial_shared_across_n",
                  "sample_size_policy": "independent_multinomial_per_n"}
        rows, entries = [], []
        for target_index, target in enumerate(config["targets"]):
            p = np.asarray([.25] * 4) if target == "uniform" else np.asarray([.125, .125, .375, .375])
            for trial in range(config["trials"]):
                arrays = {"target": p, **{f"counts_{n}": np.asarray(count_cases[n][trial]) for n in config["n_values"]}}
                path = root / "data" / "samples" / f"{target}-trial-{trial:04d}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
                entry = {"path": path.name, "target_id": target, "trial": trial,
                         "rng_coordinates": [config["seed"], target_index, trial], "sha256": runner.digest_file(path)}
                entries.append(entry)
                for n in config["n_values"]:
                    losses = {method: {"status": "finite", "bits": .2 + .01 * n + .03 * trial + .01 * k, "reason": None}
                              for k, method in enumerate(methods)}
                    losses[runner.AD] = old_ad(arrays[f"counts_{n}"], p)
                    rows.append({"sample_set_id": config["sample_set_id"], "target_id": target, "trial": trial,
                                 "n": n, "sample_file": path.name, "sample_sha256": entry["sha256"],
                                 "losses": losses, "paired_differences": {
                                     pair: runner.difference(losses[left], losses[right])
                                     for pair, (left, right) in runner.pair_names(config).items()},
                                 "diagnostics": {"discarded_large_trace": "x" * 2048}})
        sample_manifest = {"files": entries, "sampling": {key: config[key] for key in (
            "d", "n_values", "trials", "seed", "sample_set_id", "targets",
            "dirichlet_target_policy", "sample_size_policy")}}
        write(root / "data/samples/manifest.json", sample_manifest)
        write(root / "data/benchmark-config.json", config)
        trial_path = root / "data/trials.jsonl"
        trial_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        trials[name] = pin(trial_path)
        def targets_for(ids, extras):
            targets = {}
            for target in config["targets"]:
                targets[target] = {}
                for n in config["n_values"]:
                    selected = [row for row in rows if row["target_id"] == target and row["n"] == n and row["trial"] in ids]
                    cell = {"methods": {method: independent_summary([row["losses"][method] for row in selected]) for method in methods},
                            "paired_differences": {name: independent_summary([row["paired_differences"][name] for row in selected])
                                                   for name in runner.pair_names(config)}}
                    if extras:
                        cell.update(mean_posteriors={"unchanged": [0.25, 0.75]}, derived={"unchanged": "evidence"})
                    targets[target][str(n)] = cell
            return targets
        summary = {"config": config, "targets": targets_for(range(3), True), "seconds": 999,
                   "batching": {"original": "timing"}, "units": "next-symbol KL bits",
                   "trial_records_sha256": pin(trial_path)["sha256"],
                   "sample_manifest_sha256": pin(root / "data/samples/manifest.json")["sha256"]}
        blocks = {"config": config, "diagnostic_only": True, "requested_blocks": 10, "block_count": 3,
                  "pooled_targets": targets_for(range(3), False), "blocks": [
                      {"block_id": trial, "trial_start": trial, "trial_stop_exclusive": trial + 1, "trials": 1,
                       "targets": targets_for([trial], False)} for trial in range(3)]}
        write(root / "data/summary.json", summary)
        write(root / "data/block-diagnostics.json", blocks)
        outputs = {str(p.relative_to(root)): pin(p) for p in root.rglob("*") if p.is_file()}
        write(root / "manifest.json", {"outputs": outputs})
        for relative in ("manifest.json", "data/benchmark-config.json", "data/samples/manifest.json", "data/summary.json", "data/block-diagnostics.json"):
            metadata[f"merged/{name}/{relative}"] = pin(root / relative)
        originals[name] = {"summary": summary, "blocks": blocks, "rows": rows}
    spec = {"schema_version": 1, "source_package": str(package), "corrected_repo": str(REPO),
            "baseline_module_sha256": runner.digest_file(REPO / "src/lsa/alt/baselines.py"),
            "baseline_rule": BASELINE.ABSOLUTE_DISCOUNTING_RULE,
            "baseline_version": BASELINE.ABSOLUTE_DISCOUNTING_VERSION,
            "require_campaign_scope": False, "metadata_pins": metadata, "trial_pins": trials}
    spec_path = tmp_path / "spec.json"
    write(spec_path, spec)
    return spec_path, spec, originals


def test_full_recomputation_reuses_exact_samples_and_changes_only_ad(fixture, tmp_path):
    spec_path, spec, originals = fixture
    before = {str(p): pin(p) for p in Path(spec["source_package"]).rglob("*") if p.is_file()}
    output = tmp_path / "corrected"
    completed = runner.run(spec_path, output)
    assert completed["accepted_full_result"] is True
    assert completed["totals"]["recomputed_ad_paired_rows"] == 20
    for experiment in runner.EXPERIMENTS:
        summary = json.loads((output / experiment / "summary.json").read_text())
        original = originals[experiment]["summary"]
        assert summary["config"] == original["config"]
        assert summary["source_seconds"] == 999 and "seconds" not in summary
        assert summary["source_batching"] == original["batching"] and "batching" not in summary
        for target, ns in summary["targets"].items():
            for n, cell in ns.items():
                old_cell = original["targets"][target][n]
                assert cell["mean_posteriors"] == old_cell["mean_posteriors"]
                assert cell["derived"] == old_cell["derived"]
                for method in original["config"]["methods"]:
                    if method != runner.AD:
                        assert cell["methods"][method] == old_cell["methods"][method]
                for name, sides in runner.pair_names(original["config"]).items():
                    if runner.AD not in sides:
                        assert cell["paired_differences"][name] == old_cell["paired_differences"][name]
                assert cell["methods"][runner.AD]["counts"] == {"finite": 3, "infinite": 0, "undefined": 0}
        overlay = [json.loads(line) for line in (output / experiment / "ad-trials.jsonl").read_text().splitlines()]
        old = {(row["target_id"], row["trial"], row["n"]): row for row in originals[experiment]["rows"]}
        for row in overlay:
            identity = row["target_id"], row["trial"], row["n"]
            assert row["original_ad_loss"] == old[identity]["losses"][runner.AD]
            assert row["sample_sha256"] == old[identity]["sample_sha256"]
            assert row["source_non_ad_losses"] == {m: v for m, v in old[identity]["losses"].items() if m != runner.AD}
            with np.load(Path(spec["source_package"]) / "merged" / experiment / "data/samples" / row["sample_file"]) as sample:
                counts, p = sample[f"counts_{row['n']}"], sample["target"]
            s = int(np.count_nonzero(counts))
            if s == len(counts):
                q = counts / row["n"]
            else:
                delta = min(max(int(np.count_nonzero(counts == 1)), 1) / s, .9)
                q = np.where(counts > 0, (counts - delta) / row["n"], delta * s / (row["n"] * (len(counts) - s)))
            expected = sum(float(pi) * math.log2(float(pi / qi)) for pi, qi in zip(p, q))
            assert row["loss"]["bits"] == pytest.approx(expected, abs=1e-14)
        compact = json.loads(gzip.decompress((output / "compact-data" / f"{experiment}.json.gz").read_bytes()))
        assert compact == summary
    assert before == {str(p): pin(p) for p in Path(spec["source_package"]).rglob("*") if p.is_file()}
    reasons = json.loads(gzip.decompress((output / "compact-data/paired-reasons.json.gz").read_bytes()))
    assert all(not counter for family in reasons.values() for target in family.values() for cell in target.values() for counter in cell.values())
    with pytest.raises(ValueError, match="already exists"):
        runner.run(spec_path, output)


def test_pilot_is_bounded_and_cannot_certify_full_inputs(fixture, tmp_path):
    spec_path, spec, _ = fixture
    output = tmp_path / "pilot"
    completed = runner.run(spec_path, output, pilot=True, pilot_records=1)
    assert completed["accepted_full_result"] is False
    assert not (output / "completed.json").exists()
    timing = json.loads((output / "timing.json").read_text())
    for name in runner.EXPERIMENTS:
        assert timing[name]["extraction"]["bytes_read"] < spec["trial_pins"][name]["bytes"]
        assert timing[name]["extraction"]["rows"] == 1
        assert timing[name]["extraction"]["full_hash_verified"] is False
        assert timing[name]["evaluation"]["evaluations"] == 4


def test_sample_corruption_preserves_failure_without_completion(fixture, tmp_path):
    spec_path, spec, _ = fixture
    path = next((Path(spec["source_package"]) / "merged/benchmark_primary/data/samples").glob("*.npz"))
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    output = tmp_path / "bad-sample"
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        runner.run(spec_path, output)
    assert (output / "failed.json").exists() and not (output / "completed.json").exists()


def test_trial_checksum_failure_detected_in_single_scan(fixture, tmp_path):
    spec_path, spec, _ = fixture
    path = Path(spec["source_package"]) / "merged/benchmark_primary/data/trials.jsonl"
    data = path.read_bytes()
    path.write_bytes(data.replace(b"xxx", b"yyy", 1))
    output = tmp_path / "bad-trace"
    with pytest.raises(ValueError, match="trial log checksum"):
        runner.run(spec_path, output)
    assert not (output / "completed.json").exists()


def test_wrong_baseline_identity_rejected_before_output(fixture, tmp_path):
    spec_path, spec, _ = fixture
    spec["baseline_module_sha256"] = "0" * 64
    write(spec_path, spec)
    with pytest.raises(ValueError, match="source hash mismatch"):
        runner.run(spec_path, tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


def test_public_spec_builder_pins_local_paths_without_network(fixture, tmp_path):
    import make_ad_correction_spec as builder

    _, original, _ = fixture
    package = Path(original["source_package"])
    for relative in ("acceptance/manifest.json", "acceptance/result.json", "merged/merge.json"):
        write(package / relative, {"fixture": True})
    output = tmp_path / "public-spec.json"
    spec = builder.make(package, package, REPO, REPO, output)
    assert spec["source_package"] == str(package)
    assert spec["corrected_repo"] == str(REPO)
    assert spec["metadata_pins"] == {
        **original["metadata_pins"],
        **{relative: pin(package / relative) for relative in (
            "acceptance/manifest.json", "acceptance/result.json", "merged/merge.json")},
    }
    assert spec["baseline_module_sha256"] == original["baseline_module_sha256"]
    assert spec["required_runtime"] == {"python": "3.13.15", "numpy": "2.5.1", "scipy": "1.18.0"}
    assert spec["require_campaign_scope"] is True
    assert spec["scope"]["full_trial_rows"] == 99000
    with pytest.raises(FileExistsError):
        builder.make(package, package, REPO, REPO, output)


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "unexpected/duplicate"),
    ("sample_identity", "sample mismatch"),
    ("paired_value", "stored pair disagrees"),
])
def test_trial_identity_and_pair_validation(fixture, tmp_path, mutation, match):
    _, spec, _ = fixture
    inputs = runner.load_inputs(spec, np)["benchmark_primary"]
    path = inputs["root"] / "data/trials.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "sample_identity":
        rows[0]["sample_sha256"] = "0" * 64
    else:
        rows[0]["paired_differences"]["good_turing_hybrid_minus_lsa_fixed"]["bits"] = 99
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    progress = runner.Progress(tmp_path / "progress.jsonl")
    try:
        with pytest.raises(ValueError, match=match):
            runner.extract("benchmark_primary", inputs, pin(path), progress)
    finally:
        progress.close()
