# fairForget Evaluation

Intersectional dataset release status: [WarBias PR #21](https://huggingface.co/datasets/FairForget/WarBias/discussions/21) is uploaded and verified; merge to `main` is pending. The new configurations described below become available on `main` after that merge.

Development plan updated 2026-09-12. The first runner now implements all eight
registered dataset configurations, native scoring, per-item outputs and clustered
worst-group comparisons. See [implementation and commands](fairforget.md).
Human validation, inference-coverage simulations, paired between-model intervals
and a substantive model case study remain outstanding. The sections below retain
the broader September plan; proposed generic harness task names are not registered.

## Purpose

Evaluate unsupported stereotypical attribution and evidence-following in
language models, with explicit reporting for the worst-performing supported
groups. The core suite includes Ukrainian StereoSet, BBQ, WinoBias and WarBias
QA. WarBias QA also has a separately reported English track. All four benchmark
families belong to the September scope.

This project builds on EleutherAI's Language Model Evaluation Harness. Keep
the upstream MIT license and attribution. Benchmark data retains its own
license and provenance; the software license does not relicense datasets.

Initial upstream revision: `ad8737ae7fad24cf64e50fc7fc31397bff586b9e`.

## September 30 deliverable

- Versioned Ukrainian task definitions for StereoSet, BBQ, WinoBias and WarBias
  QA, plus a separately reported English WarBias QA task.
- Dataset-specific scoring protocols; generated A/B/C answers and the existing
  strict parser remain the primary WarBias QA protocol.
- Condition-specific scores, group scores, worst-group summaries and uncertainty.
- Per-item output export and a reproducible report.
- Metric stress tests and one end-to-end model case study.

Interpretability and debiasing experiments follow the benchmark release.
Human validation is a release dependency; model-reviewed examples remain
explicitly labeled as pilot data until their review is complete.

## Implementation boundaries

Reuse upstream model backends, batching, task loading and sample logging.
Keep fairness additions in dedicated task, metric and reporting modules with
minimal edits to upstream internals. Preserve an upstream remote for updates.
Use explicit, versioned datasets rather than copying annotation workbooks or
private review records into the code repository.

Proposed task names include `stereoset_uk_eval`, `bbq_uk`,
`winobias_uk_controlled`, `winobias_uk_natural`, `warbias_qa_uk_generate` and
`warbias_qa_en_generate`. These names are not registered yet. Freeze and name
each scoring protocol before implementation. Likelihood and generation tracks
must have distinct names and must not be pooled.

## Dataset integrations

These are candidate integration sources identified in the local project, not
a claim that their latest remote revisions or human validation were verified.
Pin the approved artifact and split for each task before running it.

| Family | Existing source | Scoring to preserve or validate |
| --- | --- | --- |
| StereoSet-UK | `FairForget/StereoSet-UK-Eval`; local `evaluation/stereoset_uk/evaluate.py` | Candidate likelihoods, SS, LMS and ICAT; preserve target-level aggregation and identify the causal-LM adaptation explicitly. |
| BBQ-UK | `FairForget/BBQ-UK`; local `evaluation/bbq_uk/evaluate.py`; upstream harness BBQ tasks | Ambiguous/disambiguated accuracy and bias metrics; preserve target eligibility, polarity handling, denominators and unknown-answer conventions after reference checks. |
| WinoBias-UK Controlled | `FairForget/WinoBias-UK-Controlled` | A/B coreference evaluation; pro/anti accuracy, signed and absolute gaps, paired results and type/gender strata. |
| WinoBias-UK Natural | `FairForget/WinoBias-UK-Natural`; local `evaluation/winobias_uk/evaluate.py` | Existing counterbalanced A/B likelihood protocol; primary balanced variants and separate agreement/cross controls. |
| WarBias QA | Approved bilingual QA artifact and existing `score_predictions.py` | Generated answers; ambiguous, aligned and conflicting conditions and both polarities. |

The local `evaluation/` sources above are under the sibling project's
`fair_forget_tranlsation/` directory. Reuse and test their logic rather than
assuming the English upstream tasks already implement the Ukrainian adaptation.
WinoBias is distinct from the leaderboard's WinoGrande task. Controlled and
Natural are separately named configurations, never silently mixed.

Reviewed subsets and full machine-translated releases need distinct dataset
versions and validation labels. The former 200/220/120 annotation samples are
not automatically the benchmark sizes or evidence that full sets are validated.
The latest WarBias-only annotation allocation does not remove the other three
datasets from the evaluation-tool scope.

## Evaluation contract

- Each item runs in a fresh context with no answer keys or condition labels.
- Score Ukrainian and English separately.
- For WarBias QA, preserve ambiguous, stereotype-aligned and stereotype-conflicting conditions,
  both question polarities, target group, scenario ID and source case ID.
- For WarBias QA, retain the existing rule: only A/B/C after trimming and uppercasing is valid.
  Other responses are invalid, not insufficient-information answers.
- Missing, duplicate or unexpected prediction IDs fail scoring.
- Publish denominators and eligibility rules for every task. WarBias QA rates
  include all items in their stated condition. Do not silently apply that rule
  to another dataset's conditional bias score.
- Pin model and dataset revisions, harness revision, chat template, prompt,
  generation settings and seeds. Save rendered inputs and raw outputs.

## Initial metrics

For WarBias QA, retain the existing ambiguous accuracy, ambiguous stereotype and counter-
stereotype choice rates, informative accuracy, aligned and conflicting accuracy,
their signed difference, informative unknown rate and invalid-response rate.

Add worst-group ambiguous stereotype-choice rate (maximum across supported
groups) and worst-group conflicting-evidence accuracy (minimum). Report the
group identity and all constituent group scores. Base QA supports IDP and
veteran groups. The separate matched intersectional configurations add 11
status–gender–age cells, each with 120 rows from 20 source cases, using ages
25/45/65 and retaining the older-women-veteran exclusion. They have no human
validation or annotation assignments. Report these configurations separately
from base QA, with shared-profile interpretation and coverage limits explicit.

Apply group-aware reporting separately within each benchmark, retaining its
native metrics and capability measures. Identify demographic groups, bias
categories and linguistic controls distinctly. Worst-category performance is
not automatically worst-demographic-group performance. For StereoSet retain
signed preference as well as distance from its benchmark balance point; do not
assume lower SS is always better. For WinoBias report competence alongside
pro/anti parity. Define supported groups and outcome direction per task before
computing extremes. Do not publish a pooled score across the four families.

Use Ghosh, Genuit and Reagan's worst-case min/max comparison as the basis for
the new disparity metrics. Report ratio, absolute group gap and absolute
worst-group performance together, with dataset-specific event definitions.
The adaptation and edge-case policies are specified in
[Worst-group comparisons](worst-group-metrics.md). This supersedes the earlier
weighted mean–worst composite as the initial implementation direction.

Separate group-balanced averages from pooled averages. Good average performance
cannot establish good worst-group performance; equal group performance does
not establish low harm or competence. Do not apply min/max ratios to signed
bias scores or log-likelihoods, and do not infer intersectional coverage from
category names alone. The specification remains to be implemented and validated.

## Statistical and validity checks

- Freeze group definitions and scoring before final evaluation.
- Keep all WarBias QA source-case variants together in partitions, including
  baseline rows, all demographic profiles, six conditions and translations.
  Use `profile_scenario_id` for six-variant matched panels and `source_case_id`
  for independent resampling. For BBQ preserve source question families and context pairs; for
  WinoBias preserve source items and their gender variants; for StereoSet retain
  complete triplets and related source-item families.
- Use paired model comparisons and resample independent source cases, retaining
  all their scenarios and variants. Show both scenario and source-case counts.
- Design uncertainty for the maximum/minimum across groups explicitly; do not
  present ordinary item-level error bars as cluster-aware worst-group intervals.
- Report sparse coverage and avoid unsupported claims about unseen groups.
- Distinguish new wording of a training association from unseen-association
  generalization. Preserve source-case mappings for leakage audits.
- Verify answer keys and condition relationships after human text corrections.

## Acceptance checks

1. Each harness adapter reproduces its reference scorer on identical inputs:
   counts and rates for predictions, and documented numerical tolerances for
   likelihoods. Include invalid responses, ties and empty eligible subsets.
2. Synthetic predictors test always-stereotype, always-counter, always-unknown,
   random, fixed-position, and one-group-failure behavior.
3. Answer-order checks remap all answer and stereotype indices consistently.
4. The report retains poor task competence and invalid responses even when
   stereotypical-choice rates look low.
5. A pinned model run is reproducible within documented backend tolerances.

## References

- https://github.com/EleutherAI/lm-evaluation-harness
- https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md
- https://aclanthology.org/2022.findings-acl.165/
- https://aclanthology.org/2021.acl-long.81/
- https://arxiv.org/abs/2101.01673
