#!/usr/bin/env python3
"""Recompute only AD on hash-bound saved samples and preserve accepted evidence.

Full runs stream/hash each original trial log exactly once. Pilot runs bound the
trial-log read and deliberately cannot issue a completion/acceptance certificate.
No evaluator, sampler, scheduler, network, or manuscript code is called.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

sys.dont_write_bytecode = True
AD = "absolute_discounting"
EXPERIMENTS = ("benchmark_primary", "benchmark_powers")
STATUSES = ("finite", "infinite", "undefined")
TARGETS = ("uniform", "step", "zipf_1", "zipf_1.5", "zipf_2", "zipf_3",
           "zipf_4", "zipf_5", "geometric", "dirichlet_1", "dirichlet_half")
NS = (1000, 2000, 3000, 5000, 7000, 10000, 14000, 20000)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def safe_path(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"unsafe relative input path: {relative}")
    path = root / relative
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe file: {path}")
    require(path.resolve().is_relative_to(root), f"input escapes source: {path}")
    return path


def checked_bytes(path, pin):
    path = Path(path)
    require(not path.is_symlink(), f"symlink input: {path}")
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), f"input changed: {path}")
    require(len(data) == pin["bytes"] and digest_bytes(data) == pin["sha256"],
            f"input checksum/size mismatch: {path}")
    return data


class Progress:
    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf8")
        self.last = 0.0

    def emit(self, stage, *, force=False, **values):
        if self.stream.closed:
            return
        now = time.monotonic()
        if not force and now - self.last < 15:
            return
        event = {"utc": utc(), "stage": stage, **values}
        self.stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        print(json.dumps(event, sort_keys=True), flush=True)
        self.last = now

    def close(self):
        self.stream.close()


def pair_names(config):
    methods = config["methods"]
    pairs = {f"{left}_minus_{right}": (left, right)
             for i, left in enumerate(methods) for right in methods[i + 1:]}
    if "power_mixture" in methods and "lsa_depth_mixture" in methods:
        pairs["power_mixture_minus_lsa_depth_mixture"] = (
            "power_mixture", "lsa_depth_mixture")
    if "lsa_depth_mixture" in methods and 0 in config["depths"] and len(config["depths"]) > 1:
        pairs["lsa_depth_mixture_minus_positive_depth_mixture"] = (
            "lsa_depth_mixture", "positive_depth_mixture")
    return pairs


def difference(left, right):
    if left["status"] != "finite" or right["status"] != "finite":
        return {"status": "undefined", "bits": None,
                "reason": f"nonfinite_pair:{left['status']}:{right['status']}"}
    return {"status": "finite", "bits": left["bits"] - right["bits"], "reason": None}


def valid_loss(loss):
    require(set(loss) == {"status", "bits", "reason"}, "invalid loss schema")
    require(loss["status"] in STATUSES, "invalid loss status")
    if loss["status"] == "finite":
        require(isinstance(loss["bits"], (int, float)) and math.isfinite(loss["bits"]),
                "nonfinite finite-loss value")
    else:
        require(loss["bits"] is None, "nonfinite loss must have null bits")


def summarize(losses, np):
    counts = {status: sum(x["status"] == status for x in losses) for status in STATUSES}
    require(sum(counts.values()) == len(losses) and losses, "invalid summary inputs")
    status = "undefined" if counts["undefined"] else ("infinite" if counts["infinite"] else "finite")
    result = {"trials": len(losses), "counts": counts, "status": status,
              "mean_bits": None, "se_bits": None}
    if status == "finite":
        values = np.asarray([x["bits"] for x in losses], dtype=float)
        result.update(mean_bits=float(values.mean()),
                      se_bits=float(values.std(ddof=1) / np.sqrt(len(values)))
                      if len(values) > 1 else None)
    return result


def summary_equal(actual, expected, label):
    require(set(actual) == set(expected), f"summary fields differ: {label}")
    for key in actual:
        a, b = actual[key], expected[key]
        if key in ("mean_bits", "se_bits") and a is not None and b is not None:
            require(math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-14),
                    f"saved aggregate mismatch: {label}/{key}: {a} != {b}")
        else:
            require(a == b, f"saved aggregate mismatch: {label}/{key}")


def load_inputs(spec, np):
    package = Path(spec["source_package"]).resolve()
    checked = {}
    for relative, pin in spec["metadata_pins"].items():
        checked[relative] = json.loads(checked_bytes(safe_path(package, relative), pin))
    experiments = {}
    for experiment in EXPERIMENTS:
        prefix = f"merged/{experiment}/"
        manifest = checked[prefix + "manifest.json"]
        config = checked[prefix + "data/benchmark-config.json"]
        samples = checked[prefix + "data/samples/manifest.json"]
        summary = checked[prefix + "data/summary.json"]
        blocks = checked[prefix + "data/block-diagnostics.json"]
        if spec.get("original_source_commit"):
            require(manifest["source"]["commit"] == spec["original_source_commit"],
                    "wrong original scientific source commit")
        if spec.get("original_source_tree_sha256"):
            require(manifest["source"]["tree_sha256"] == spec["original_source_tree_sha256"],
                    "wrong original scientific source tree")
        for relative in ("data/benchmark-config.json", "data/samples/manifest.json",
                         "data/summary.json", "data/block-diagnostics.json"):
            require(manifest["outputs"][relative] == spec["metadata_pins"][prefix + relative],
                    f"manifest/pin mismatch: {prefix}{relative}")
        require(manifest["outputs"]["data/trials.jsonl"] == spec["trial_pins"][experiment],
                f"trial pin mismatch: {experiment}")
        require(summary["config"] == blocks["config"] == config,
                f"saved config mismatch: {experiment}")
        require(summary["trial_records_sha256"] == spec["trial_pins"][experiment]["sha256"],
                "summary does not bind original trials")
        require(summary["sample_manifest_sha256"] ==
                spec["metadata_pins"][prefix + "data/samples/manifest.json"]["sha256"],
                "summary does not bind saved samples")
        for key in ("d", "n_values", "trials", "seed", "sample_set_id", "targets",
                    "dirichlet_target_policy", "sample_size_policy"):
            require(samples["sampling"][key] == config[key], "sample sampling mismatch")
        require(config.get("trial_start", 0) == 0 and AD in config["methods"], "unexpected cohort")
        if spec.get("require_campaign_scope", True):
            require(config["targets"] == list(TARGETS) and config["trials"] == 1000 and
                    config["d"] == 10000, "wrong full campaign scope")
            require(config["n_values"] == (list(NS) if experiment == EXPERIMENTS[0] else [1000]),
                    "wrong full campaign sample-size grid")
            require(config["seed"] == (2026100701 if experiment == EXPERIMENTS[0] else 2026100702),
                    "wrong full campaign seed")
            require(config["sample_set_id"] == ("primary" if experiment == EXPERIMENTS[0] else "powers"),
                    "wrong full campaign sample identity")
        require(config["sample_size_policy"] == "independent_multinomial_per_n" and
                config["dirichlet_target_policy"] == "redraw_per_trial_shared_across_n",
                "unexpected sampling policies")
        entries = {}
        for entry in samples["files"]:
            key = (entry["target_id"], entry["trial"])
            require(key not in entries, "duplicate saved sample identity")
            require(entry["rng_coordinates"] == [config["seed"], TARGETS.index(key[0]), key[1]],
                    "saved sample RNG identity mismatch")
            relative = "data/samples/" + entry["path"]
            require(relative in manifest["outputs"] and
                    manifest["outputs"][relative]["sha256"] == entry["sha256"],
                    "sample manifest/output hash mismatch")
            entries[key] = entry
        expected_samples = {(target, trial) for target in config["targets"]
                            for trial in range(config["trials"])}
        require(set(entries) == expected_samples, "incomplete saved sample coverage")
        require(len({entry["path"] for entry in entries.values()}) == len(entries),
                "saved sample paths are not unique")
        require(blocks["diagnostic_only"] is True and blocks["block_count"] == min(10, config["trials"]),
                "unexpected block diagnostics")
        require(blocks["pooled_targets"] == {
            target: {n: {key: cell[key] for key in ("methods", "paired_differences")}
                     for n, cell in ns.items()} for target, ns in summary["targets"].items()},
            "pooled block summaries do not match source")
        block_trials = []
        for i, block in enumerate(blocks["blocks"]):
            require(block["block_id"] == i and block["trial_start"] == len(block_trials),
                    "blocks are not consecutive")
            ids = list(range(block["trial_start"], block["trial_stop_exclusive"]))
            require(len(ids) == block["trials"], "block trial count mismatch")
            block_trials.extend(ids)
        require(block_trials == list(range(config["trials"])), "blocks do not partition trials")
        experiments[experiment] = dict(root=package / "merged" / experiment,
                                       manifest=manifest, config=config, samples=samples,
                                       summary=summary, blocks=blocks, entries=entries)
    require(experiments[EXPERIMENTS[0]]["config"]["seed"] !=
            experiments[EXPERIMENTS[1]]["config"]["seed"], "sample sets are not independent")
    return experiments


def extract(experiment, inputs, pin, progress, *, limit=None):
    """One binary read supplies both the full digest and bounded-memory parsing."""
    config, entries = inputs["config"], inputs["entries"]
    pairs = pair_names(config)
    ad_pairs = {name: sides for name, sides in pairs.items() if AD in sides}
    path = safe_path(inputs["root"], "data/trials.jsonl")
    before = path.stat()
    require(before.st_size == pin["bytes"], "trial log size mismatch before reading")
    sha, immutable_sha = hashlib.sha256(), hashlib.sha256()
    records, total_bytes = {}, 0
    reasons = {target: {str(n): {name: {} for name in pairs} for n in config["n_values"]}
               for target in config["targets"]}
    start = time.monotonic()
    with path.open("rb") as stream:
        for line in stream:
            total_bytes += len(line)
            sha.update(line)
            if not line.strip():
                continue
            require(line.endswith(b"\n"), "unterminated source trial record")
            row = json.loads(line)
            key = (row["target_id"], row["trial"], row["n"])
            require(key not in records and key[:2] in entries and key[2] in config["n_values"],
                    "unexpected/duplicate source trial identity")
            sample = entries[key[:2]]
            require(row["sample_set_id"] == config["sample_set_id"] and
                    row["sample_file"] == sample["path"] and
                    row["sample_sha256"] == sample["sha256"], "source trial sample mismatch")
            require(set(row["losses"]) == set(config["methods"]), "original loss coverage mismatch")
            require(set(row["paired_differences"]) == set(pairs), "original pair coverage mismatch")
            for loss in row["losses"].values():
                valid_loss(loss)
            for name, item in row["paired_differences"].items():
                valid_loss(item)
                if item["reason"] is not None:
                    counter = reasons[key[0]][str(key[2])][name]
                    counter[item["reason"]] = counter.get(item["reason"], 0) + 1
            # Every stored ordinary pair is independently checked from saved
            # losses, including original nonfinite reason semantics.
            for name, (left, right) in pairs.items():
                if left in row["losses"] and right in row["losses"]:
                    require(row["paired_differences"][name] == difference(row["losses"][left], row["losses"][right]),
                            f"stored pair disagrees with its saved losses: {name}")
            immutable = {"identity": list(key), "losses": {m: value for m, value in row["losses"].items() if m != AD},
                         "paired_differences": {name: value for name, value in row["paired_differences"].items() if name not in ad_pairs}}
            immutable_sha.update(canonical(immutable) + b"\n")
            # Huge numerical diagnostics, posteriors and unchanged pair values
            # stay in the original trace; they are never regenerated.
            records[key] = {"losses": row["losses"],
                            "original_ad_pairs": {name: row["paired_differences"][name] for name in ad_pairs}}
            progress.emit("extract", experiment=experiment, rows=len(records),
                          bytes_read=total_bytes, expected_bytes=pin["bytes"])
            if limit is not None and len(records) >= limit:
                break
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), "source trial log changed during scan")
    if limit is None:
        require(total_bytes == pin["bytes"] and sha.hexdigest() == pin["sha256"],
                "source trial log checksum mismatch")
        require(len(records) == len(entries) * len(config["n_values"]), "incomplete source trial coverage")
        for target in config["targets"]:
            for n in config["n_values"]:
                for name in pairs:
                    item = inputs["summary"]["targets"][target][str(n)]["paired_differences"][name]
                    require(sum(reasons[target][str(n)][name].values()) == item["counts"]["undefined"],
                            "original paired reason count does not match summary")
    inputs["source_pair_reasons"] = reasons
    result = {"rows": len(records), "bytes_read": total_bytes, "seconds": time.monotonic() - start,
              "full_hash_verified": limit is None, "observed_sha256": sha.hexdigest(),
              "observed_sha256_scope": "full_file" if limit is None else "read_prefix_only",
              "unchanged_source_records_canonical_sha256": immutable_sha.hexdigest()}
    progress.emit("extract_complete", force=True, experiment=experiment, **result)
    return records, result


def evaluate(experiment, inputs, records, baseline, np, progress, *, extra_first_trial=False):
    config, manifest = inputs["config"], inputs["manifest"]
    keys = {key[:2] for key in records}
    if extra_first_trial:
        keys.update((target, 0) for target in config["targets"])
    predictions, cases = {}, {}
    sample_bytes = 0
    start = time.monotonic()
    for i, key in enumerate(sorted(keys)):
        entry = inputs["entries"][key]
        relative = "data/samples/" + entry["path"]
        payload = checked_bytes(safe_path(inputs["root"], relative), manifest["outputs"][relative])
        sample_bytes += len(payload)
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            require(set(saved.files) == {"target", *(f"counts_{n}" for n in config["n_values"])},
                    "wrong saved sample array coverage")
            p = saved["target"]
            baseline.validate_probabilities(p, size=config["d"])
            for n in config["n_values"]:
                identity = (*key, n)
                if identity not in records and not (extra_first_trial and key[1] == 0):
                    continue
                counts = saved[f"counts_{n}"]
                require(counts.shape == (config["d"],) and np.issubdtype(counts.dtype, np.integer)
                        and np.all(counts >= 0) and int(counts.sum()) == n,
                        "invalid saved count vector")
                prediction = baseline.absolute_discounting(counts)
                require(prediction.probabilities is not None and prediction.reason is None,
                        "robust AD returned undefined prediction")
                q = baseline.validate_probabilities(prediction.probabilities, size=config["d"])
                loss = baseline.kl_bits(p, q)
                require(math.isfinite(loss), "robust AD is not finite on the saved campaign")
                diagnostics = prediction.diagnostics
                require(diagnostics["estimator_rule"] == baseline.ABSOLUTE_DISCOUNTING_RULE and
                        diagnostics["estimator_version"] == baseline.ABSOLUTE_DISCOUNTING_VERSION,
                        "robust AD identity missing from diagnostics")
                predictions[identity] = {"loss": {"status": "finite", "bits": float(loss), "reason": None},
                                         "diagnostics": diagnostics}
                cases[diagnostics.get("boundary_case") or "ordinary"] = cases.get(diagnostics.get("boundary_case") or "ordinary", 0) + 1
        progress.emit("evaluate_ad", experiment=experiment, sample_files=i + 1,
                      expected_sample_files=len(keys), evaluations=len(predictions))
    result = {"sample_files": len(keys), "sample_bytes": sample_bytes,
              "evaluations": len(predictions), "boundary_cases": cases,
              "seconds": time.monotonic() - start}
    progress.emit("evaluate_ad_complete", force=True, experiment=experiment, **result)
    return predictions, result


def corrected_cell(target, n, trial_ids, records, predictions, original, pairs, np):
    """Only AD and pairs containing AD may change in a copied cell."""
    result = copy.deepcopy(original)
    identities = [(target, trial, n) for trial in trial_ids]
    for method, expected in original["methods"].items():
        summary_equal(summarize([records[key]["losses"][method] for key in identities], np),
                      expected, f"{target}/{n}/{method}")
    result["methods"][AD] = summarize([predictions[key]["loss"] for key in identities], np)
    for name, (left, right) in pairs.items():
        if AD not in (left, right):
            continue
        summary_equal(summarize([records[key]["original_ad_pairs"][name] for key in identities], np),
                      original["paired_differences"][name], f"{target}/{n}/{name}")
        values = []
        for key in identities:
            losses = {**records[key]["losses"], AD: predictions[key]["loss"]}
            values.append(difference(losses[left], losses[right]))
        result["paired_differences"][name] = summarize(values, np)
    for method in original["methods"]:
        if method != AD:
            require(result["methods"][method] == original["methods"][method], "non-AD summary changed")
    for name, sides in pairs.items():
        if AD not in sides:
            require(result["paired_differences"][name] == original["paired_differences"][name],
                    "non-AD pair summary changed")
    return result


def save_experiment(experiment, inputs, records, predictions, output, correction, np):
    config = inputs["config"]
    pairs = pair_names(config)
    output.mkdir()
    trial_path = output / "ad-trials.jsonl"
    reasons = copy.deepcopy(inputs["source_pair_reasons"])
    for target in config["targets"]:
        for n in config["n_values"]:
            for name, sides in pairs.items():
                if AD in sides:
                    reasons[target][str(n)][name] = {}
    with trial_path.open("x", encoding="utf8") as stream:
        for target in config["targets"]:
            for trial in range(config["trials"]):
                for n in config["n_values"]:
                    key = (target, trial, n)
                    old = records[key]["losses"]
                    new = {**old, AD: predictions[key]["loss"]}
                    changed_pairs = {name: difference(new[left], new[right])
                                     for name, (left, right) in pairs.items() if AD in (left, right)}
                    for name, value in changed_pairs.items():
                        if value["reason"] is not None:
                            counter = reasons[target][str(n)][name]
                            counter[value["reason"]] = counter.get(value["reason"], 0) + 1
                    sample = inputs["entries"][(target, trial)]
                    row = {"sample_set_id": config["sample_set_id"], "target_id": target,
                           "trial": trial, "n": n, "sample_file": sample["path"],
                           "sample_sha256": sample["sha256"], "source_non_ad_losses": {m: old[m] for m in config["methods"] if m != AD},
                           "original_ad_loss": old[AD], "loss": predictions[key]["loss"],
                           "paired_differences": changed_pairs, "diagnostics": predictions[key]["diagnostics"]}
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    summary, blocks = copy.deepcopy(inputs["summary"]), copy.deepcopy(inputs["blocks"])
    for target in config["targets"]:
        for n in config["n_values"]:
            summary["targets"][target][str(n)] = corrected_cell(
                target, n, range(config["trials"]), records, predictions,
                inputs["summary"]["targets"][target][str(n)], pairs, np)
            blocks["pooled_targets"][target][str(n)] = {
                key: summary["targets"][target][str(n)][key] for key in ("methods", "paired_differences")}
            for old_block, block in zip(inputs["blocks"]["blocks"], blocks["blocks"], strict=True):
                block["targets"][target][str(n)] = corrected_cell(
                    target, n, range(block["trial_start"], block["trial_stop_exclusive"]),
                    records, predictions, old_block["targets"][target][str(n)], pairs, np)
    # Make the altered estimator and overlay record format explicit. Historical
    # whole-run timings are preserved under source_seconds, never misattributed.
    if "seconds" in summary:
        summary["source_seconds"] = summary.pop("seconds")
    if "batching" in summary:
        summary["source_batching"] = summary.pop("batching")
    summary["source_trial_records_sha256"] = summary["trial_records_sha256"]
    summary["trial_records_sha256"] = digest_file(trial_path)
    summary["trial_records_kind"] = "ad_replacement_overlay_with_original_non_ad_losses"
    summary["correction"] = blocks["correction"] = correction
    write_json(output / "summary.json", summary)
    write_json(output / "block-diagnostics.json", blocks)
    write_json(output / "benchmark-config.json", config)
    return summary, blocks, reasons


def write_compact(results, output):
    """Five replacements use exactly the public paper_results.py data schema.

    Other public results/templates must be copied unchanged by the publication
    assembler; this directory is intentionally not a complete public bundle.
    """
    root = output / "compact-data"
    root.mkdir()
    values = {"paired-reasons.json.gz": {name: item[2] for name, item in results.items()}}
    for name, (summary, blocks, _) in results.items():
        values[f"{name}.json.gz"] = summary
        values[f"{name}-blocks.json.gz"] = blocks
    for filename, value in values.items():
        with (root / filename).open("xb") as stream:
            stream.write(gzip.compress(canonical(value) + b"\n", mtime=0))


def write_tables(results, output):
    common = ["experiment", "sample_set_id", "target", "n"]
    fields = ["status", "mean_bits", "se_bits", "trials", "finite_trials", "infinite_trials", "undefined_trials"]
    method_rows, pair_rows, ad_pair_rows = 0, 0, 0
    def values(item):
        return [item["status"], item["mean_bits"], item["se_bits"], item["trials"],
                *[item["counts"][s] for s in STATUSES]]
    with (output / "methods-all.tsv").open("x", newline="") as mf, (output / "paired-differences-all.tsv").open("x", newline="") as pf:
        mw, pw = csv.writer(mf, delimiter="\t"), csv.writer(pf, delimiter="\t")
        mw.writerow(common + ["method", "recomputed", *fields])
        pw.writerow(common + ["comparison", "left", "right", "recomputed", *fields])
        for experiment, (summary, _, _) in results.items():
            config = summary["config"]
            for target in config["targets"]:
                for n in config["n_values"]:
                    prefix = [experiment, config["sample_set_id"], target, n]
                    cell = summary["targets"][target][str(n)]
                    for method, item in cell["methods"].items():
                        mw.writerow([*prefix, method, method == AD, *values(item)])
                        method_rows += 1
                    for name, (left, right) in pair_names(config).items():
                        changed = AD in (left, right)
                        pw.writerow([*prefix, name, left, right, changed,
                                     *values(cell["paired_differences"][name])])
                        pair_rows += 1
                        ad_pair_rows += int(changed)
    return {"method_rows": method_rows, "paired_rows": pair_rows, "recomputed_ad_paired_rows": ad_pair_rows}


def run(spec_path, output, *, pilot=False, pilot_records=20):
    require(not Path(output).is_symlink(), "output directory cannot be a symlink")
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    raw_spec = spec_path.read_bytes()
    spec = json.loads(raw_spec)
    require(spec["schema_version"] == 1, "unsupported recomputation spec")
    require(not output.exists(), "output directory already exists; never retry over it")
    package, repo = Path(spec["source_package"]).resolve(), Path(spec["corrected_repo"]).resolve()
    require(not output.is_relative_to(package) and not output.is_relative_to(repo),
            "output must be outside source package and corrected checkout")
    module_path = safe_path(repo, "src/lsa/alt/baselines.py")
    require(digest_file(module_path) == spec["baseline_module_sha256"], "corrected baseline source hash mismatch")
    sys.path.insert(0, str(repo / "src"))
    baseline = importlib.import_module("lsa.alt.baselines")
    np = importlib.import_module("numpy")
    require(Path(baseline.__file__).resolve() == module_path, "wrong imported baseline module")
    require(baseline.ABSOLUTE_DISCOUNTING_RULE == spec["baseline_rule"] and
            baseline.ABSOLUTE_DISCOUNTING_VERSION == spec["baseline_version"], "wrong robust AD version")
    runtime = {"python": platform.python_version(), "numpy": np.__version__,
               "scipy": importlib.import_module("scipy").__version__, "platform": platform.platform()}
    for key, value in spec.get("required_runtime", {}).items():
        require(runtime[key] == value, f"runtime mismatch: {key}")
    output.mkdir(parents=True)
    progress = Progress(output / "progress.jsonl")
    started = time.monotonic()
    correction = {"method": AD, "rule": spec["baseline_rule"], "version": spec["baseline_version"],
                  "baseline_module_sha256": spec["baseline_module_sha256"],
                  "source_package": str(package), "spec_sha256": digest_bytes(raw_spec),
                  "estimator_reevaluated": [AD], "lsa_reevaluated": False,
                  "samples_regenerated": False, "original_artifacts_modified": False}
    try:
        write_json(output / "spec.json", spec)
        write_json(output / "started.json", {"utc": utc(), "pilot": pilot, "runtime": runtime,
                                            "correction": correction, "script_sha256": digest_file(__file__)})
        progress.emit("verify_metadata", force=True)
        inputs = load_inputs(spec, np)
        results, timings = {}, {}
        for experiment in EXPERIMENTS:
            records, extraction = extract(experiment, inputs[experiment], spec["trial_pins"][experiment],
                                          progress, limit=pilot_records if pilot else None)
            predictions, evaluation = evaluate(experiment, inputs[experiment], records, baseline, np,
                                               progress, extra_first_trial=pilot)
            timings[experiment] = {"extraction": extraction, "evaluation": evaluation}
            if pilot:
                examples = []
                for key, row in records.items():
                    losses = {**row["losses"], AD: predictions[key]["loss"]}
                    examples.append({"identity": list(key), "sample_set_id": inputs[experiment]["config"]["sample_set_id"],
                                     "original_ad_loss": row["losses"][AD], **predictions[key],
                                     "paired_differences": {name: difference(losses[left], losses[right])
                                                            for name, (left, right) in pair_names(inputs[experiment]["config"]).items() if AD in (left, right)}})
                write_json(output / f"{experiment}-pilot.json", {"paired_examples": examples,
                    "evaluated_identities": [list(key) for key in sorted(predictions)],
                    "full_source_hash_verified": False, "accepted_full_result": False})
            else:
                progress.emit("aggregate", force=True, experiment=experiment)
                results[experiment] = save_experiment(experiment, inputs[experiment], records,
                    predictions, output / experiment, correction, np)
            del records, predictions
        # Recheck small pins and module identity without a second huge-file scan.
        for relative, pin in spec["metadata_pins"].items():
            checked_bytes(safe_path(package, relative), pin)
        require(digest_file(module_path) == spec["baseline_module_sha256"], "baseline source changed during run")
        totals = {} if pilot else write_tables(results, output)
        if not pilot:
            write_compact(results, output)
        if not pilot and spec.get("require_campaign_scope", True):
            require(totals == {"method_rows": 902, "paired_rows": 3773,
                              "recomputed_ad_paired_rows": 803}, "incorrect final campaign row totals")
        timings["total_seconds"] = time.monotonic() - started
        write_json(output / "timing.json", timings)
        progress.emit("pilot_complete" if pilot else "complete", force=True, **totals)
        progress.close()
        files = {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": digest_file(path)}
                 for path in sorted(output.rglob("*")) if path.is_file()}
        write_json(output / "files-manifest.json", {"schema_version": 1, "files": files})
        completion = {"utc": utc(), "status": "pilot_complete" if pilot else "complete",
                      "accepted_full_result": not pilot, "full_source_hash_verified": not pilot,
                      "correction": correction, "totals": totals,
                      "files_manifest_sha256": digest_file(output / "files-manifest.json")}
        write_json(output / ("pilot-complete.json" if pilot else "completed.json"), completion)
        return completion
    except BaseException as exc:
        progress.emit("failed", force=True, error=repr(exc))
        progress.close()
        write_json(output / "failed.json", {"utc": utc(), "error": repr(exc), "accepted_full_result": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-records", type=int, default=20)
    args = parser.parse_args()
    require(args.pilot_records > 0, "pilot-records must be positive")
    run(args.spec, args.output, pilot=args.pilot, pilot_records=args.pilot_records)


if __name__ == "__main__":
    main()
