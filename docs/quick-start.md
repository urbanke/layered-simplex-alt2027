# A small complete example

Install the package as described in the README, then run:

```sh
python scripts/get_kjv.py
python scripts/alt_experiments.py plan --protocol experiments/alt2027/smoke.json
python scripts/alt_experiments.py campaign --protocol experiments/alt2027/smoke.json --purpose smoke --out output/example
python scripts/alt_experiments.py report --protocol experiments/alt2027/smoke.json --purpose smoke --sources output/example --out output/example-report
python scripts/alt_experiments.py verify output/example-report
```

The example exercises the experiment families and reporting pipeline on small
alphabets, shallow depths and few trials. It does not build the full numerical
table store and does not reproduce the paper's numerical values. For those
values use the saved-results route or the full experiment protocol.

Each execution writes into a new directory and preserves its inputs and results.
To run the example again, choose new output directory names. A successful
verification checks the saved file inventory, contents and protocol identity.
