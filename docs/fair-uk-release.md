# Fair-UK v0.1.0 — pilot software release

Fair-UK provides the evaluation tooling for a WarBias-centered study in Ukrainian and English. The four WarBias configurations are the core tracks; BBQ-UK, StereoSet-UK Eval and WinoBias-UK Natural/Controlled provide complementary evaluations. The toolkit does not depend on an unbiased model or unlearning methods.

## Included

- Eight checksum-verified HF task configurations and their native scoring protocols.
- Strict generated A/B/C WarBias evaluation; mean-token likelihood adapters for the complementary tasks.
- Condition-specific rates, worst-group performance, disparity gaps and min/max comparisons, with source-case bootstrap intervals and explicit sparse/undefined cases.
- Immutable model/data fingerprints, source-code hashes, resume checks, raw predictions and offline rescoring.
- Paired UK–EN WarBias comparisons with exact case matching and shared source-case resampling across translations and profiles.
- Native-score Markdown tables, subgroup CSV exports and model-by-task experiment summaries.
- CLI `fair-uk-eval`; Python module `lm_eval.fair_uk`; installation extra `.[fair-uk]`.

The old development module `lm_eval.fairforget` and command `fairforget-eval` have been replaced. Use a fresh environment or reinstall this checkout. The upstream package distribution name remains `lm_eval`; Fair-UK has its own tool version and does not publish a replacement package to PyPI. Install from this repository's `fair-uk-v0.1.0` release tag. The upstream `lm-eval` command, MIT license and attribution remain intact.

## Validation

On 2026-09-12:

- All 26 focused tests passed, including actual local harness generation, checkpoint resumption, offline rescoring, likelihood/token-boundary agreement, paired language comparisons and experiment exports. Integration models are tiny random fixtures.
- Static checks and formatting passed.
- All eight registered dataset files passed checksum, row-count and complete-case structure checks.
- Earlier native-score parity checks used constructed candidate scores against project reference evaluators: 100 BBQ rows, all 949 StereoSet items and 12 WinoBias Natural rows.
- A local tiny-random-model software smoke check exercised 132 matched intersectional WarBias items per language over four source cases, followed by paired comparison and report export. This is not a substantive model experiment or paper result.
- A clean wheel build was inspected for the renamed module, registry and CLI entry point, with the old module excluded.
- The merged WarBias extension was verified on HF `main` at `c5aceb6265e8c780549cfc1b8dc623c490b86b4c`: 13 updated files matched the audited extension; all 40 other existing files, including eight annotation artifacts, were preserved.

To repeat the local software tests: `pytest tests/fair_uk -q`.

## Release boundaries

All registered datasets remain **pilot, human-unvalidated** releases. The software release does not certify construct validity or population-level fairness. No new annotator evaluations or assignments were created. Dataset licenses and existing HF namespaces remain unchanged.

Percentile intervals are exploratory; zero-event data can yield zero-width intervals. Confirmatory coverage studies, paired between-model inference and activation analysis remain future work. GPU/vLLM execution has not been validated. Paper model experiments are deferred by the project owner and are not part of this release.

The [focused GitHub Actions template](fair-uk-ci.yml) is included but not installed: the current GitHub credential lacks workflow permission. The release is locally validated; no successful remote CI run is claimed.

See [usage](fair-uk.md), [metric definitions](worst-group-metrics.md) and [remaining September work](fair-uk-roadmap.md).
