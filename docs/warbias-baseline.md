# WarBias baseline metrics

Fair-UK calculates the triplet and QA metrics, severity slices, group summaries
and confidence intervals for verified WarBias benchmark predictions.

Analyze verified saved runs with the library CLI:

```sh
fair-uk-eval warbias baseline \
  --release /path/to/frozen-release \
  --results /path/to/results \
  --output /path/to/analysis
```

`metric_definitions()` provides metric definitions and units.
`analyze_checkpoints()` returns the report; `write_baseline_results()` exports it.
`analyze_baselines()` is an alias for the same scorer.

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

Use `--qa-answer-policy abc_explicit_choice_v3` to parse explicit answer prefixes
and a separate first answer block before an explanation. Exact unique options
may have one terminal period. Conflicting choices and unmatched prose remain
invalid. The same policy applies to every model and language. Native v2 runs
are verified before rescoring; raw files and strict bare-letter diagnostics stay
unchanged. The report records both policies, parse methods and changed choices.
Omit this option to reproduce `abc_option_text_v2` results.

## Benchmark subsets

Set `triplet_ids` to select complete semantic components. Full native files are
verified before selection. Raw and neutral scores use the same IDs; all QA rows
remain included. The report records the selected IDs and their hash.

Use `--triplet-ids selected_ids.txt` for the equivalent CLI option.
`--neutral-inference-release` declares a compatible frozen collector for control
runs when it is separate from the control input release. Original run identities
are preserved.

Primary SS and QA accuracy exports include `scope=actor_severity` with `actor=idp` or
`actor=veteran` and `severity_level=1..5`. Each slice uses its own component
weights and bootstrap interval. Empty slices stay empty; a single component
has no interval. `scope=severity` retains the combined-group results.
