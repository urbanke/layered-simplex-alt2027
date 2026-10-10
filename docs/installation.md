# Installation and environment

The scientific implementation supports Linux and macOS. The published numerical
experiment environment was Python 3.13.15 on Linux x86-64, with NumPy 2.5.1,
SciPy 1.18.0, Matplotlib 3.11.1, mpmath 1.4.1 and ReportLab 5.0.0. The dependency
file pins those versions. Python 3.14 on macOS is used for release checks.
Windows is not a tested target: execution and claim locking use POSIX facilities.

Create a fresh virtual environment using the README commands. Run commands from
the root of this repository. A downloaded source ZIP is also supported; Git is
not required for scientific file verification. `release-source.json` records the
released source files. Output directories must be new for each execution.

Use one numerical thread per worker for the declared experimental execution:

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Run the tests with:

```sh
python -m pytest -q
```

The sealed numerical evaluator includes C source, not a platform-specific binary.
Build it explicitly with `scripts/alt_build_sealed_native.py`; the command's
`--help` describes compiler and output arguments. The generated library and its
metadata bind compiler flags and implementation bytes. Use the resulting library
and its SHA-256 when configuring the engine. A library from another operating
system or architecture must be rebuilt, not copied into place.

The Python interpolation implementation is retained for independent native/Python
checks. The direct numerical reference routines are retained for the paper's
validation; they are not a substitute for the table-backed full benchmark when
estimating runtime.
