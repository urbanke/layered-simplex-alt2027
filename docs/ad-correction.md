# Recompute the AD correction from original saved trials

The v1.1.0 correction evaluates robust absolute discounting on the paper's exact
original saved samples. It reuses every other method's saved trial losses,
including Good–Turing and LSA, and recomputes the paired comparisons containing
AD. The [protocol](protocol.md#absolute-discounting) defines the corrected rule.

This route needs the **full original saved samples and trial records**. The
compact `results/` bundle contains summaries, not the per-trial information
needed to compute a new paired standard error. For fresh experiments without
the original large outputs, use [full reproduction](full-reproduction.md),
which evaluates the robust AD rule directly alongside the other methods.

## Required saved package

These tools implement the correction of the originally accepted paper campaign.
The specification builder deliberately pins that campaign's source identity,
sample sets and runtime. It is not a converter for arbitrary v1.0.0 experiments.
The original scientific source commit is
`1c67b7af1928f0fdec7e2a9cb5abae631a9e00cb`.

The package must retain the original layout:

```text
saved-v1.0-package/
  acceptance/
    manifest.json
    result.json
  merged/
    merge.json
    benchmark_primary/
      manifest.json
      data/
        benchmark-config.json
        summary.json
        block-diagnostics.json
        trials.jsonl
        samples/
          manifest.json
          TARGET-trial-NNNN.npz
    benchmark_powers/
      manifest.json
      data/
        ... same layout ...
```

Each sample NPZ stores the labelled target probability vector as `target`,
and labelled count vectors as `counts_N`. Each trial record identifies its
sample set, target, global trial number, sample size, sample filename and SHA256,
and retains method losses and paired differences. Source manifests bind these
files. Keep the original package immutable.

The primary cohort contains 11 targets, eight sample sizes
(1,000; 2,000; 3,000; 5,000; 7,000; 10,000; 14,000; 20,000), and 1,000 trials per
cell. The powered cohort contains 11 targets and 1,000 trials at N=1,000 from
its independent sample set. Together these are 99 cells and 99,000 trial rows.
The accepted Bible experiments did not include AD and are outside this correction.

The original logs occupy about 73.8 GB and saved NPZ payloads about 1.4 GB.
The correction streams and hashes each trial log once; parsing and storage
throughput can dominate its runtime. It does not build numerical tables or
reevaluate LSA. Do not estimate its duration from the original campaign or a
small pilot.

## Freeze the specification

Use the paper's Python 3.13.15, NumPy 2.5.1 and SciPy 1.18.0 environment.
The generated specification enforces these versions, the exact corrected
baseline source hash, original input hashes, and full campaign coverage.
See [installation](installation.md) for environment setup. The commands below
run from the repository root and use only local paths. The correction output
must be outside both the repository and the saved input package; this example
uses a sibling directory:

```sh
mkdir -p ../ad-correction-results
python -B scripts/make_ad_correction_spec.py \
  --local-package "$PWD/saved-v1.0-package" \
  --remote-package "$PWD/saved-v1.0-package" \
  --local-repo "$PWD" \
  --remote-repo "$PWD" \
  --output ../ad-correction-results/spec.json
```

Despite their names, the `--remote-package` and `--remote-repo` arguments are
simply paths to use on the machine that runs the computation. The builder
performs no network operation. They can differ from the local paths when
preparing a specification for another machine. Use absolute paths and move or
copy the inputs before freezing the specification. The builder needs only the
small metadata files locally; the computation needs the complete saved payloads.

The JSON specification contains `source_package`, `corrected_repo`,
`metadata_pins`, `trial_pins`, `baseline_rule`, `baseline_version`,
`baseline_module_sha256`, `required_runtime`, and the original source identity.
Each input pin has `sha256` and `bytes` fields. Do not edit the scientific scope,
relax the hash checks, or substitute regenerated samples to make a mismatch pass.

## Run a bounded pilot, then the full correction

```sh
python -B scripts/recompute_saved_ad.py \
  --spec ../ad-correction-results/spec.json \
  --output ../ad-correction-results/pilot \
  --pilot --pilot-records 20
```

The pilot reads at most 20 nonblank trial records from each experiment. It also
evaluates the first saved trial of every target at every original sample size,
checking all 99 cells with bounded work. Its paired checks cover only the read
prefix of each trial log. **A pilot does not fully hash those logs and cannot
certify complete results.** It writes `pilot-complete.json`, explicitly marking
`full_source_hash_verified` and `accepted_full_result` false.

After checking the pilot, run the complete correction into a different new
directory:

```sh
python -B scripts/recompute_saved_ad.py \
  --spec ../ad-correction-results/spec.json \
  --output ../ad-correction-results/full
```

The full run checks every trial identity, sample hash and original ordinary
paired difference. It hashes each sample before evaluating AD on the verified
bytes. The original AD summaries and affected paired summaries are reconstructed
and checked against the saved aggregates before replacement. Other marginal
summaries are also checked against their original losses. Unchanged paired
summaries, posterior summaries and derived results remain copied values.

The result contains 902 method/cell rows and 3,773 pair/cell rows. Of these,
99 AD method/cell rows and 803 pairs containing AD are recomputed. Standard
errors use all 1,000 trials with `ddof=1`. The original ten consecutive blocks
of 100 trials remain diagnostic partitions; they do not replace pooled estimates.
Primary and powered samples remain independent.

## Inspect and preserve the output

`progress.jsonl` records phase, byte and row progress. `timing.json` separates
extraction and AD evaluation costs. A successful full run writes
`completed.json` and an exact `files-manifest.json`. Failures preserve their
partial output and `failed.json` without a completion marker. Existing output
directories are refused; use a fresh path for another attempt.

For each experiment, `ad-trials.jsonl` is a compact replacement overlay. It
contains the original sample identities and hashes, original AD loss, unchanged
non-AD losses, corrected AD loss and diagnostics, and corrected AD pairs. It is
not a replacement for the retained original full trace.

The corrected `summary.json` and `block-diagnostics.json` preserve the original
configs and carry an explicit top-level `correction` identity. The original
trial-log hash is kept as `source_trial_records_sha256`; the new
`trial_records_sha256` and `trial_records_kind` identify the overlay. Historical
evaluation timings are labelled `source_seconds` and `source_batching` and
must not be attributed to the correction.

`methods-all.tsv` and `paired-differences-all.tsv` include all cells and identify
recomputed rows. The `compact-data/` directory supplies both corrected summaries,
both block files and complete paired reason counts in the public reporting
schema. These five files are an overlay, not a complete reporting bundle. A
publication assembler must retain the other scientific results and templates,
select the public scientific fields and safe correction metadata, and generate
a new verified bundle manifest. Execution records contain the operator's paths;
do not copy those records into a public scientific bundle indiscriminately.

The scripts themselves do not modify the package, overwrite original results,
submit jobs, or publish a new release. The original v1.0.0 results remain a
separate historical record.

## Test the correction path

```sh
python -B -m pytest -q tests/test_saved_ad_correction.py
```

The small fixture includes original finite, infinite and undefined AD cases,
independently calculated robust probabilities, unchanged non-AD values,
bounded pilots, hash failures, and duplicate or mismatched sample identities.
No full trial payload or numerical store is required for these tests.
