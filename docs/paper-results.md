# Regenerate the paper displays

After installation, from the repository root:

```sh
python scripts/reproduce_paper.py --check
python scripts/reproduce_paper.py --out output/paper-displays
python scripts/export_comparisons.py --out output/paper-comparisons
```

Use new output directory names if repeating these commands. The compact input
bundle in `results/` contains saved scientific summaries, uncertainty/block
summaries, corpus controls and numerical validation measurements. The commands
verify their checksums and recreate the current paper's displays. They do not
resample data, evaluate the models again or establish new numerical validation.

The v1.1.0 reference correction changes AD and every paired comparison that
contains AD. It uses the original 99,000 benchmark trials and preserves the
saved non-AD results, posterior summaries and numerical validation. Its rule,
boundary conventions and historical difference are given in the
[protocol](protocol.md#absolute-discounting). Regenerating these displays does
not rerun the correction or certify a new full campaign. The original v1.0.0
release retains the previous AD definition and results.
See the [saved-sample correction guide](ad-correction.md) for recomputing AD
from the original trial records and building the correction artifacts.

The architecture illustration is regenerated as TikZ source, as used in the
paper. The other three figures are PDFs. Six numbered table objects are generated,
including the numerical validation table; the factorial table carries two labels.
The paper map records the display roles and data definitions.

The comparison export includes 902 method rows and 3,773 paired rows across 99
benchmark cells. Each cell summarizes 1,000 trials with ten disjoint blocks of
100. Negative A_minus_B favors A. Preserve infinite/undefined outcome counts,
their reasons, and the distinction between a standard error and a confidence
interval. The primary and powered benchmark sample sets are independent.

Plot values and numerical table entries are reproducible from these inputs;
PDF byte identity can depend on renderer versions and metadata. The factorial
offset uses the saved slope-one fit, not the free-slope intercept. See the
[protocol](protocol.md) for definitions and [full reproduction](full-reproduction.md)
for generating fresh numerical results from the exact paper configuration.

For a newly completed campaign and its own numerical checks, use:

```sh
python scripts/reproduce_paper.py --runs output/merged --validation-root output/validation --out output/paper-displays
python scripts/export_comparisons.py --runs output/merged --out output/paper-comparisons
```

This reads verified fresh Runs. It does not borrow the included reference
validation or timing measurements. Without `--validation-root`, only the
experimental displays can be generated; the numerical-validation table is
explicitly unavailable.
