# Inputs and generated files

This repository contains programs and reproduction instructions, not a bulk
simulation archive. Full numerical tables and per-trial outputs are created
locally by the full reproduction commands.

## Included inputs

- Exact experiment grids and seeds in `experiments/alt2027/protocol.json`.
- Explicit count-anchor lists for constructing the numerical table store.
- Corpus source links and byte-hash/tokenization manifests in `data/`, without
  the corpus text itself. The preparation command downloads the source, applies
  the exact tokenization and checks the resulting bytes against the paper's
  input. See [NOTICE.txt](../data/NOTICE.txt) for source and licensing.
- About 1.6 MB of scientific reference summaries in `results/`, used
  to check results and regenerate paper displays without rerunning predictions.
  These summaries are not the full saved trial data and do not replace independent
  experimental reproduction.

## Download and verify the Bible inputs

```sh
python scripts/get_kjv.py
```

The source is [Project Gutenberg eBook 10](https://www.gutenberg.org/ebooks/10),
using its [UTF-8 text download](https://www.gutenberg.org/ebooks/10.txt.utf-8).
The command checks the downloaded source, prepares both tokenizations and checks
their exact text hashes before saving `data/kjv.txt` and `data/kjv-secondary.txt`.
For an already downloaded copy, use `--from-file PATH`.

The canonical prepared text has SHA-256
`27c79dc3ad00db78e55a09f9d0140470e42ce11bd41099a2972712fbe2e48196`;
the secondary control has SHA-256
`b6f472f38d88a2efcfc000295560a92483da3e7b0ade00fe3f30160cf27f9560`.
These are the exact token streams used for the paper. The manifests record
token counts, type counts and the separate preparation recipes. If a source
download changes, preparation stops rather than accepting a new edition.
Downloaded and prepared text is ignored by Git.

## Generated locally

The complete numerical store covers depths 2–138, with 50,553 stored columns and
approximately 11.63 GB of binary numerical payload, plus indices/metadata. Its
builder is supplied. Native libraries are built for the machine on which the
experiments will run.

The full saved scientific experiment outputs occupied approximately 76 GB before
compression in the paper's execution. Space used by a new execution also depends
on temporary files, validation output and whether raw and merged records are
retained together. Allow additional free space for those stages; the numerical
store size alone is not a full storage budget.

Generated numerical tables belong in `tables/`; experiment records and displays
belong in `output/` or another chosen work directory. Those directories are ignored
by Git. The programs record input settings and file checksums and reject changed
or incomplete results. New executions use new output directories.

No cluster account or private download is required. Full reproduction does
require sufficient computing resources and disk space. The small example and
saved-summary display regeneration do not build the full numerical store.
