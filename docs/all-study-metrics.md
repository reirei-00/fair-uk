# Saved analysis of all study metrics

The study report tracks all 18 commitments reconciled from Tetiana's sheet.
`paper all-metrics` keeps every entry visible even when its input, review or
calibration is missing. It performs no model inference or training. Missing
effects stay pending or partial and never become zero.

The preserved uncertainty kernels recompute native summaries per metric and
scope. Large BBQ panels can require substantial CPU time at 1,000 draws. Budget
independent arms in parallel on CPU and measure the actual runtime; a local
synthetic timing is not a cluster completion estimate. Keep the selected draw
count and scientific rules unchanged when choosing execution resources.

The six native tasks alone provide useful measurements, but do not cover all
18 commitments. Their registered metric catalog remains available through
`python -m lm_eval.fair_uk metrics --format json`. `paper compare-native`
compares those native scores and supplies paired component intervals. The
all-metrics layer adds the study effects and honest dependency records.

## Calculation source and validation

The pure calculation modules `intervention_metrics.py`, `transfer_metrics.py`
and `analysis_inputs.py` are restored byte-for-byte from Fair-UK commit
`710ae774a327e948e8957879dcc1ce071bbc5eae`. The [source receipt](../lm_eval/fair_uk/study_metric_sources.json) lists their hashes.
Only the pure `resolve` and `measurement_policy` helpers from its baseline
module are retained in a small compatibility module. The old collector,
`measure` CLI, JEV and external capability package are not activated.

The new analysis source is separate from the immutable collection source.
For each pair, a subprocess runs the new export helper with the selected frozen
collector interpreter and only its evaluator on `PYTHONPATH`. The helper calls
that collector's `load_verified_run` and paired-source checks. It requires:

- Exact collector code identity in both saved run manifests.
- The same complete source IDs, metadata, tokenizer, effective inputs and fixed
  inference settings before and after the intervention.
- Active source eligibility and the explicitly requested source split.
- All three native files: `run.json`, `report.json`, `predictions.jsonl`.

An evaluation pair must contain evaluation rows. A complete two-item development
qualification bundle cannot stand in for a full evaluation. Explicit development
analysis is allowed and remains separate from evaluation transfer. No minimum
row count is invented. The source bundle defines the inventory.

When actual training evidence is supplied, every pair also binds to that method's
execution receipt. The post checkpoint path and full file-hash map must match the
saved training output. The base path must match the frozen recipe. For adapter
arms, every base file in the native model identity must match the trainer's saved
source-file hashes. The RMU receipt lacks base-file hashes, so its base binding is
explicitly path plus recipe pin only; its post hashes remain strict. A valid NPO
pair cannot be labelled WGA and combined with WGA exposure. Null task-vector
selection is allowed when it matches its own actual saved output.

The normalized export preserves the collector's answer policy, source rows,
predictions, scored rows and provenance. Analysis records its own code hashes
and the export digest; it never edits the original native files. Exports can
contain private source text and responses. Keep them with the study evidence,
outside a code-only repository export.

## Invocation and configuration

```bash
python -m lm_eval.fair_uk paper all-metrics \
  --config path/to/frozen-analysis-config.json \
  --output path/to/new-analysis-directory
```

The output directory must be new and outside saved inputs. The main output is
`all_metrics.json`; each available task has `TASK/analysis.json`, a normalized
`validated_pair.json` and the isolated validation log.

The configuration has these fields. Paths resolve relative to the configuration.

| Field | Meaning |
| --- | --- |
| `schema_version` | `1` |
| `kind` | `fair_uk_all_metrics` |
| `collector.python` | Explicit interpreter for saved-run validation |
| `collector.evaluator_root` | Immutable collector's evaluator root; it may differ from the analysis source |
| `intervention` | `id`, `method`, `training_language`, explicit `target_values` and any separately declared QA harm metric |
| `required_tasks` | Every intended task, including unavailable or source-blocked views |
| `task_blocks` | Optional task-to-reason mapping for source-policy blocks; blocked tasks cannot have an active pair |
| `pairs` | Records with `task`, `phase`, `before_run`, `after_run`, `dataset_bundle`; optional `annotations`, `before_judgments`, `after_judgments` |
| `rules` | `tau_by_task`, `min_independent_support`, `min_target_improvement`, `bootstrap`, `seed`; support and practical gates default to null |
| `exposure` | Optional paths to `execution`, `data_manifest`, `training_binding`, `method_directory`, `recipe`, `evidence_manifest` |

Different pairs may use different frozen bundles. This permits an original
native collection and an additive collection to retain their original identities.
A missing pair or absent native directory is recorded as pending. Invalid
evidence raises an error rather than silently disappearing. With no pairs, an
exposure-only report still contains all 18 metric entries and task states.

For the Ukrainian all-reviewed-gender integration, `target_values={}` preserves
the actual intervention scope. It does not invent an IDP, veteran, men, actor or
topic target to enable a redistribution ratio. StereoSet category/target scopes,
including gender and non-target categories, remain separate in the native and
association records. Descriptive cell changes are not a target-qualified M3 or
RI claim. Training language `uk` can support M3d when matched English outputs
actually exist. Unpublished English StereoSet remains blocked under its source
policy; WarBias language transfer is reported on its own panel.

## Inventory and dependencies

The current six native views expose the following registered metric IDs. Units,
scopes, weighting and denominator definitions come from `metrics --format json`.
Counts are diagnostics rather than intervention effects.

| Native view | Registered metric IDs |
| --- | --- |
| StereoSet UK | `ss`, `lms`, `icat`, `worst_target_ss_distance_from_50`, `stereotype_tie_rate` |
| BBQ UK | `accuracy_all_rows`, `accuracy`, `unknown_rate`, `raw_bias`, `bias_score`, `signed_bias_macro`, `absolute_bias_macro`, `tie_rate` |
| WinoBias Natural and Controlled, each separately | `pro_accuracy`, `anti_accuracy`, `primary_accuracy`, `signed_bias_gap`, `absolute_bias_gap`, `pair_consistency`, `agreement_control_accuracy`, `cross_control_accuracy`, `tie_rate` |
| WinoPron UK | `prompt_accuracy`, `prompt_attempt_rate`, `option_order_consistency`, `double_accuracy`, `double_attempt_rate`, `single_accuracy`, `single_attempt_rate`, `accuracy`, `attempt_rate`, `pair_consistency`, `female_minus_male_accuracy`, `gender_form_consistency`, `full_gender_pair_consistency`, `n_prompts`, `n_rows`, `independent_pairs` |
| WarBias triplets UK | `ss`, `lms`, `icat`, `stereotype_tie_rate`, `worst_profile_ss_distance_from_50` |

WarBias triplets also expose length-bucket SS and length/margin rank-correlation
diagnostics. Additional bounded group rates, worst values, gaps and min/max
ratios are descriptive group analyses; they are distinct from the study's
support-qualified M1/M1b/M2 common-content requirements. WinoPron consistency is
not a stereotype score or a refusal classifier.

| Metric | Calculation and currently required evidence |
| --- | --- |
| Forget efficacy | `abs(SS_pre-50)-abs(SS_post-50)` in percentage points, preserving native target/component weights; absolute SS, ties, crossing 50, item margins and each completion's likelihood change are retained |
| NNR | Fixed baseline-eligible `margin_pre > tau` cohort; cured minus reversed, with complete S/N/C transitions; requires a frozen calibration identity, exact baseline prediction SHA and eligible IDs |
| Retain utility | Native absolute and paired correctness/LMS/consistency outcomes; benign helpfulness requires judgments; these in-scope measures do not qualify general capability |
| LMS | Native related-versus-unrelated comparison and paired change; unrelated likelihood diagnostics remain separate |
| ICAT | Native composite from the same scope's SS and LMS, with both components reported |
| M1 | Minimum post quality on one matched common-content panel and an explicit independent-support rule |
| M1b | Minimum/maximum post quality on that qualified panel; zero maximum leaves the ratio null |
| M2 | Range and population standard deviation of `Q_pre-Q_post` group losses on the qualified common panel |
| M3a | Target and matched non-target actor changes; requires a declared actor target and matched content |
| M3b | Within-actor topic changes; requires reviewed topic membership and target/near/far labels |
| M3c | Matched explicit gender/age profile changes; requires actual target values and support |
| M3d | Opposite-language pre/post effect after single-language training, using aligned content; a baseline language gap is a different quantity |
| RI | Per-axis sum of adverse non-target changes divided by stable positive target reduction; requires a named harm scale, practical minimum and a positive paired target interval; raw changes remain visible without a ratio |
| M4a | Full/partial/any benign refusal, task success and judgment coverage; requires raw benign responses, response-bound judgments and human calibration |
| M4b | Accuracy, unknown and invalid outcomes under sufficient evidence; factual favorable/adverse slices require explicit source-bound assignments rather than inferred aligned/conflicting labels |
| BBQ | Original BBQ formulas by native category/condition and a separately labelled WarBias QA adaptation; undefined non-unknown mass cannot establish neutrality |
| Training imbalance | Actual stream/family/token events and declared objective weights; exposure-outcome association or causal imbalance requires additional matched outcomes/design |
| Severity | Reviewed high/standard slices bound to the adapted dataset SHA; partial coverage is descriptive and intervals are not supplied |

With the six native panels, F/LMS/ICAT and in-scope correctness changes can be
calculated after complete predictions exist. M1/M1b/M2 still need qualified
common content and an explicit support decision. No support threshold of five
is silently applied by this layer. BBQ unknown outcomes support QA caution;
they are neither free-text refusal nor automatically favorable/adverse evidence.

WarBias QA, expanded QA, intersectional/cross-actor panels, benign responses and
aligned English runs require their own collector inputs when absent. No saved
triplet likelihood can replace a missing generated benign answer. Calibration,
semantic reviews and judgment coverage remain independent gates.

For a non-null `tau_by_task[TASK]`, `rules.neutrality_calibration[TASK]` must
provide `calibration_id`, `baseline_predictions_sha256` and
`baseline_eligible_ids`. The exact cohort is verified against the saved baseline.
A calibration name without a threshold, or a threshold without a frozen
calibration, does not qualify NNR. Empty eligible cohorts remain undefined.

The restored study kernels use native mean-token triplet margins. They do not
convert raw WarBias triplets into the separate primary prior-normalised WarBias
endpoint. That endpoint still uses `paper compare` with original and neutral
likelihood runs at both checkpoints and its own frozen calibration. Those scales
and datasets must not be pooled.

## Actual exposure

The new exposure adapter uses the current reviewed data manifest and trainer
binding, not the archived baseline train inventory. It verifies the original
review-manifest hash, frozen stream bytes, recipe hash, receipt and collected
evidence hashes. Every event must name a permitted source ID and family in its
declared stream, and its valid-token count must equal its recorded completion
mask. Adversarial derived IDs remain linked to their source family.

Positive acquisition, forgetting, correction and preservation are separate roles.
The report retains observed IDs, source IDs, families, steps, token totals and
objective-weighted token totals. The declared human-validation status remains
unchanged. These summaries do not infer exposure-outcome correlation, missing
groups, applied-gradient causality or broad utility preservation.
Logged scalar stream weights are separate from per-token confidence weights and
gradient magnitudes. A corrective-derived adversarial view retains its original
source ID and family even when presented in the forget role.

The main report contains exactly 18 `metrics` records with `id`, `status`,
`reason`, real file/JSON-pointer `references` and `missing_tasks`. Available
scopes retain their values while unavailable scopes retain their reasons.
`task_status`, detailed `pairs`, `transfer` and `training_exposure` allow the
reader to distinguish collected inputs, calculable effects and unqualified
dependent interpretations.
