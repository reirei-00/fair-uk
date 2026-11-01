# WarBias baseline metrics

Fair-UK calculates the triplet and QA metrics, severity slices, group summaries
and confidence intervals. The experiment walkthrough lives in
[FairForget](https://github.com/reirei-00/fair_forget/blob/aaed72f54ba19f4678caa834f1940247ed701c76/notebooks/warbias_baseline.ipynb)
and imports `lm_eval.fair_uk.warbias.baseline`.

Analyze verified saved runs with the library CLI:

```sh
fair-uk-eval warbias baseline \
  --release /path/to/frozen-release \
  --results /path/to/results \
  --output /path/to/analysis
```

`metric_definitions()` provides the definitions and units used by the notebook.
`analyze_baselines()` returns the report; `write_baseline_results()` exports it.

For runs from a frozen compatible collector, set `--inference-release` to that
snapshot. The library checks its checksums and unchanged inference and native
scoring code, then records the inference and analysis identities separately.
