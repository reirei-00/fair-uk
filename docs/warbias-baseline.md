# WarBias baseline metrics

Fair-UK calculates the triplet and QA metrics, severity slices, group summaries
and confidence intervals. The companion repository's walkthrough is
`warbias-artifacts/notebooks/warbias_baseline.ipynb`; it imports
`lm_eval.fair_uk.warbias.baseline`.

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

Schema-1 native-chat fingerprints include the rendered prompts. For the known
collector, language checks compare the verified checkpoint assets, model options
and fixed runtime settings. The report records this policy and preserves the
original input-bound hashes. Other collectors require matching tokenizer hashes.

For declared metadata corrections, add `--metadata-release CORRECTED_RELEASE`
and `--inference-release ORIGINAL_COLLECTOR`. Native files are verified against
their original release. The corrected release must preserve model assets,
prompts, answer roles, pairing and components. Author-bound case/profile
corrections change grouping and severity only; QA review-status updates are
retained. Reports bind both releases and list changed IDs. Original native files
remain unchanged.

Add `--neutral-release CONTROL_RELEASE --neutral-results CONTROL_RESULTS` for
prior-normalised SS. This separate release contains the two bilingual neutral
tasks and binds the active triplet texts, profiles and source manifest. Its
reviewed edits replace target mentions only; exact source spans and a bound rule
catalog preserve the remaining text. Assistant acceptance is recorded separately
from human dataset review. Original runs are verified first. Controls must use
the same pinned assets and likelihood settings, with repeated smoke qualification.
The report adds `scoring=prior_normalised`, including severity slices and paired
language contrasts. Missing or invalid controls retain an explicit per-model
status; existing raw and QA results remain available.
