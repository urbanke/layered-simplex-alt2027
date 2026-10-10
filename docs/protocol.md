# Experimental protocol

The executable settings are in `experiments/alt2027/protocol.json`. Its seeds,
grids, trial counts and numerical settings are fixed before sampling. The
following is a reading guide to that file; the JSON is the exact configuration.

| Experiment | Fixed workload |
|---|---|
| Architecture illustration | Alphabet 24, four layers, seed 7 |
| Complexity spectrum | Three alphabet/sample-size panels; 11 Zipf exponents; 1,000 profiles per cell; all declared depths through 69, 92 or 138 |
| Primary benchmark | Alphabet 10,000; 11 targets; eight sample sizes from 1,000 to 20,000; 1,000 trials per cell; depths 0–80 |
| Powered benchmark | Independent 1,000-trial samples for the 11 targets at sample size 1,000; powers 0–80 |
| Bible | Canonical 915,860-token stream; five main prefixes and two short-prefix controls; alphabet 100,000; depths 0–54; 2,000-step chain check |
| Tokenization control | Separate 917,868-token stream and its declared prefixes; classical and Dirichlet methods |
| Factorial scaling | Five alphabet sizes × five sample sizes × four Zipf exponents; 5,000 profiles per cell |
| Depth coefficient | 17 depth coefficients; Zipf exponents 2, 3 and 4; 1,000 profiles per exponent |

## Comparisons and uncertainty

Methods within a trial use the same target and observed counts. For random
Dirichlet targets, a target is redrawn for each trial and shared across that
trial's sample sizes; counts are independently drawn at the different sample
sizes. The primary and powered benchmarks use independent target/sample sets
and separate seeds. Do not form a paired comparison across those two sets.

A method mean describes its saved per-trial loss. A paired difference is computed
within each common-sample trial before aggregation. A negative `A_minus_B`
favors A. Its standard error describes the uncertainty of that paired mean;
it is not a confidence interval. The ten disjoint blocks of 100 benchmark trials
are stability diagnostics. They do not add independent trials, replace the pooled
estimate or define a stopping rule.

Genuine infinite KL loss is retained. Undefined outcomes and their reasons are
also retained. A method with nonfinite outcomes is not summarized by silently
averaging only its finite trials. Reporting distinguishes these cases from
numerical evaluation errors.

## Model and plotting conventions

Depth zero is the uniform point-mass prior; it is included in the depth average.
The average is a Bayesian mixture of marginal likelihoods, so its predictive
weights depend on the observed counts. It is not an arithmetic average of losses
or an oracle choice of the best depth for each target.

The complexity-spectrum plot reports cumulative excess loss per sample, while
the learning curves report next-symbol KL loss. Their depth-mixture comparisons
have different meanings. Log-axis plots cannot display an exact zero; the display
rules and zero omissions are recorded with the corresponding result generator.

The factorial table uses the paper's separate fitted slope and slope-one offset
conventions. The offset is not the intercept of the free-slope regression.
All rounding belongs to the report renderer; saved numerical summaries retain
full precision.

## Numerical validation

The validation configurations cover endpoint identities, exchangeability,
normalization, independent low-depth calculations, Mellin/contour comparisons,
powered kernels, profile/depth refinement, table interpolation and Bible chain
consistency. Their finite tested domains and measured residuals are stated with
the numerical validation table. Numerical accuracy thresholds are not sampling
standard errors, and a normalization guard is not a universal accuracy bound.

The full experiment route retains native/Python and independent reference checks.
Changing a numerical implementation, domain or tolerance requires fresh validation
of that change rather than reuse of a success statement for different inputs.
