# A Layered Simplex Architecture for Large Alphabets

Code, experiment settings and results accompanying the paper by Meir Feder,
Yaniv Fogel and Ruediger Urbanke.

The layered simplex architecture is a simple architecture-induced prior: multiply
independent uniform simplex vectors coordinate-wise and normalize. This package
studies how depth changes that prior and its predictive behavior. It includes
the LSA predictors, the comparison methods, numerical validation and the programs
for the paper's figures and tables.

Version 1.1.0 corrects the absolute-discounting (AD) comparison to use the
robust experimental rule described in the [protocol](docs/protocol.md).
The correction reuses the original samples and saved results for every other
method. See the [change log](CHANGELOG.md) for its scope and the distinction
from the original v1.0.0 results.
The [saved-sample correction guide](docs/ad-correction.md) describes how to
verify the original inputs and reproduce this correction separately.

## Install

Use Python 3.13 or 3.14 on Linux or macOS. Native numerical acceleration requires
a C compiler. From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

The paper used Python 3.13 on Linux x86-64. See [installation](docs/installation.md)
for the environment, native build and test instructions.

## Three ways to use the package

| Route | What it does | Guide |
|---|---|---|
| Small example | Runs a reduced experiment through prediction, reporting and verification | [Quick start](docs/quick-start.md) |
| Reproduce displays | Regenerates the current paper's plots and tables from saved scientific results | [Paper results](docs/paper-results.md) |
| Full experiments | Builds numerical tables, runs the declared sampling grids and validates the results | [Full reproduction](docs/full-reproduction.md) |

The example uses explicitly reduced settings. The full paper settings are in
[`experiments/alt2027/protocol.json`](experiments/alt2027/protocol.json).
The paper contains four figures and six numbered tables, including the numerical
validation table. The factorial table has two labels.

## Data and numerical conventions

Compact scientific reference results (about 1.6 MB) are included in
`results/` to check and regenerate the paper displays. Numerical tables and full
simulation outputs are generated locally; they are not stored in this repository.
Corpus text is downloaded separately from its original source. The repository
records the source links, preparation procedure and expected checksums so that
the input can be checked against the paper's exact text. See
[inputs and generated files](docs/data.md).

All reported losses are in bits unless stated otherwise. Comparisons use paired
samples within an experiment. The primary and powered experiments use independent
sample sets. Standard errors, genuine infinite losses and undefined outcomes
remain distinct. The [experimental protocol](docs/protocol.md) describes these
conventions and the fixed sampling grids.

## Layout

- `src/`: predictors, numerical evaluation, baselines and independent checks.
- `experiments/`: full paper settings, validation cases and the small example.
- `scripts/`: experiment, download, validation and reporting entry points.
- `tests/`: numerical, integrity and reporting tests.
- `data/`: corpus source notices and checksum manifests; no corpus text.
- `results/`: compact scientific results and the paper display map.
- `docs/`: installation, reproduction and interpretation instructions.

## License and citation

The software is under the [MIT license](LICENSE). Corpus data have their own
[notice](data/NOTICE.txt); the software license does not change their status.
See [third-party notices](THIRD_PARTY_NOTICES.txt) and [CITATION.cff](CITATION.cff).
