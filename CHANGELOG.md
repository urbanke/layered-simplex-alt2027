# Change log

## 1.1.0 — robust absolute-discounting correction

- Use `delta = min(max(c1, 1) / s, 0.9)` for partially observed alphabets,
  with a uniform empty-sample prediction and empirical prediction when all
  symbols have been observed. The method is identified as `robust-ad-v1`.
- Recompute AD on the original 99,000 trials in the 99 primary and powered
  benchmark cells, including its paired differences and ten-block summaries.
  Reuse every other method's saved losses and all LSA/power posteriors.
- Update the affected table columns, learning-curve series and comparisons
  from the corrected summaries. Show a nonfinite-curve note only when the
  plotted data contain such outcomes.
- Preserve the original v1.0.0 results and accepted execution records. The
  correction is separate from a fresh full experiment or numerical validation
  campaign; sampling grids, seeds and numerical engines are unchanged.

See the [protocol](docs/protocol.md#absolute-discounting) for the source and
complete estimator definition.
