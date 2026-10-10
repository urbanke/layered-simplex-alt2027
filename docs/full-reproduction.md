# Recompute the experiments

This route builds the numerical tables and generates fresh experimental records
from the paper's fixed settings and seeds. It does not require the authors' saved
large outputs or a private cluster account. It requires substantial computing
time and disk space; use the [small example](quick-start.md) first.

Version 1.1.0 uses the robust AD rule and explicit boundary conventions in the
[protocol](protocol.md#absolute-discounting). A fresh run evaluates that rule
alongside the other methods. The included reference correction separately
reuses the original non-AD trial results; it is not evidence that this full
campaign has been rerun.
To reproduce only that correction from the original saved samples and losses,
follow the [saved-sample correction guide](ad-correction.md).

Run from the repository root after [installation](installation.md). Keep the
source unchanged throughout a run and cap numerical library threads at one per
worker as described there. Choose new output paths for a new execution. The
examples below use one worker, which can be increased to fit available memory
and CPUs. Increasing workers changes scheduling, not the trial grid or seeds.

## Prepare the corpus

Follow the source-download and checksum instructions in [data](data.md). Prepare
both the canonical token stream and the secondary tokenization control before
running corpus validation or the full campaign. A changed download must fail
verification; it must not silently become a different experiment.

```sh
python scripts/get_kjv.py
```

## Build and verify the numerical store

```sh
python scripts/alt_build_kernel_store.py plan --anchors experiments/alt2027/count-anchors.json --levels 2-138 --out tables/paper-sealed
python scripts/alt_build_kernel_store.py build --store tables/paper-sealed --workers 1 --seal
python scripts/alt_build_kernel_store.py verify --store tables/paper-sealed
python scripts/alt_build_sealed_native.py --out output/native.so --cc cc
```

The plan pins the count anchors and construction geometry. Building all levels
is a substantial computation. The complete store has 50,553 columns and about
11.63 GB of binary payload. The native build prints its `sha256`; substitute that
value for `NATIVE_SHA256` below. Keep the adjacent native metadata file together
with the library.

```sh
python scripts/alt_experiments.py configure-engine --store-directory tables/paper-sealed --native-library-path output/native.so --native-library-sha256 NATIVE_SHA256 --out output/engine.json
python scripts/alt_experiments.py configure-reference --engine-config output/engine.json --out output/reference
```

Configuration checks the newly built store against the declared geometry and
binds its actual file hashes. A fresh build requires fresh numerical validation;
the paper's certificate is not reused. Native binaries must be built for the
machine that runs the experiments.

## Run the numerical checks

```sh
python scripts/alt_experiments.py calibrate --suite kernel --config experiments/alt2027/kernel-validation.json --engine-config output/engine.json --out output/validation/kernel
python scripts/alt_experiments.py calibrate --suite prior --config experiments/alt2027/prior-validation.json --engine-config output/engine.json --out output/validation/prior
python scripts/alt_experiments.py calibrate --suite depth --config experiments/alt2027/depth-validation.json --engine-config output/engine.json --workers 1 --out output/validation/depth
python scripts/alt_experiments.py calibrate --suite power --config experiments/alt2027/power-validation.json --engine-config output/engine.json --workers 1 --out output/validation/power
python scripts/alt_experiments.py calibrate --suite chain --config experiments/alt2027/bible-chain-validation.json --engine-config output/engine.json --out output/validation/chain
python scripts/alt_validate_kernel_store.py --engine-config output/engine.json --config experiments/alt2027/sealed-store-validation.json --workers 1 --out output/validation/sealed_store
python scripts/alt_validate_sealed_profiles.py --config experiments/alt2027/sealed-profile-validation.json --candidate-engine output/engine.json --legacy-engine output/reference/engine.json --out output/validation/sealed_profile
python scripts/alt_admission.py regressions --repo . --out output/validation/regressions
```

The `--legacy-engine` argument names the direct numerical reference implementation;
it does not require a previous checkout or previous experiment. These finite
checks measure the domains declared in their configuration files, rather than
certifying arbitrary inputs.

Create `output/validation/evidence.json` with this content:

```json
{
  "kernel": "kernel",
  "prior": "prior",
  "depth": "depth",
  "power": "power",
  "chain": "chain",
  "sealed_store": "sealed_store",
  "sealed_profile": "sealed_profile",
  "regressions": "regressions"
}
```

Then assemble a certificate binding the completed evidence, source, engine and
protocol. All required checks must pass; do not reduce tolerances to obtain a
certificate.

```sh
python scripts/alt_admission.py assemble --repo . --protocol experiments/alt2027/protocol.json --engine-config output/engine.json --evidence output/validation/evidence.json --out output/admission
```

## Execute the full fixed campaign

```sh
python scripts/alt_distributed.py prepare --protocol experiments/alt2027/protocol.json --purpose production --engine-config output/engine.json --block-size 20 --factorial-block-size 500 --batch-size 20 --split-benchmark-n --out output/plan.json
python scripts/alt_distributed.py work --plan output/plan.json --engine-config output/engine.json --calibration output/admission/calibration.json --host-profile local --workers 1 --out output/raw
```

The plan contains 7,754 disjoint jobs. Each completed Run records its input hashes,
scientific settings, saved samples, numerical identities and output inventory.
The word `production` here describes a new execution that passed its own admission
checks. The paper's original raw Runs retained their validation labels and were
accepted separately; their labels are not changed by these instructions.

For multiple allocated machines, use the same source, engine, plan and raw output
root, with a distinct `--worker-index I --worker-count K` for each controller,
where `I` ranges from zero to `K-1`. Under Slurm use `--host-profile slurm` inside
an allocation. Choose the worker count within each allocation's resource limits.
Completed outputs are checked and skipped on resumption; claim locks prevent
overlapping ownership. Preserve incomplete attempts and do not remove locks to
force concurrent execution.

Store immutable numerical tables on fast local storage when available. Shared
filesystem access can dominate runtime. If moving inputs, preserve their bytes,
update paths before configuration and admission, and retain the resulting input
identities. A single short example is not a reliable whole-campaign runtime
estimate.

## Merge, report and check

After every planned job has completed:

```sh
python scripts/alt_distributed.py merge --plan output/plan.json --runs output/raw --out output/merged
python scripts/alt_experiments.py report --protocol experiments/alt2027/protocol.json --purpose production --sources output/merged --out output/report
python scripts/alt_experiments.py verify output/report
python scripts/export_comparisons.py --runs output/merged --out output/comparisons
python scripts/reproduce_paper.py --runs output/merged --validation-root output/validation --out output/paper-displays
```

The merge checks job coverage and saved records; the report command verifies its
input Runs before rendering. Comparison export reads saved primary and power
trials, preserves nonfinite outcomes and exports saved paired differences and
standard errors within each sample set. This stage can read tens of gigabytes; run it on a compute allocation
when using a shared cluster.

The final command applies the current paper's figure and table layouts to the
fresh experiment and validation results. It does not substitute the included
reference results. Runtime measurements depend on hardware and must be measured
for a new execution; the authors' timing numbers are not assigned to that run.

See [paper results](paper-results.md) for the current display conventions and the
compact reference route. Full reproduction uses the fresh records above, rather
than treating a successful rendering of included reference summaries as a new
experiment. Numerical values should agree within the declared numerical accuracy;
runtime, file metadata and PDF bytes need not be identical across machines.
