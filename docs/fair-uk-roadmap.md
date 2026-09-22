# Fair-UK implementation and September plan

Updated 2026-09-12. **WarBias is the main research contribution. Fair-UK is the complementary evaluation toolkit**, built on EleutherAI's Language Model Evaluation Harness. WarBias has Ukrainian and English tracks. The toolkit is independent of unlearning experiments and needs no unbiased reference model.

## Implemented for the v0.1 pilot

- `fair-uk-eval` and `python -m lm_eval.fair_uk`, with isolated fairness modules and preserved upstream model backends.
- Four core task configurations: `warbias_uk`, `warbias_en`, `warbias_intersectional_uk`, `warbias_intersectional_en`.
- Four complementary configurations: `bbq_uk`, `stereoset_uk`, `winobias_uk_natural`, `winobias_uk_controlled`. They retain distinct native protocols and reporting scales.
- Immutable HF revisions, checksums, row counts, structural validation and whole-case smoke subsets.
- Native and condition-specific scores; source-case weighted group rates; worst-group harm/success, gaps and min/max comparisons with exploratory clustered intervals.
- Strict generated A/B/C WarBias scoring, invalid-response reporting and checkpointed model runs.
- Paired WarBias UK–EN comparisons: exact source IDs and semantic metadata, matching model/settings, re-scoring raw predictions and paired source-case bootstrap intervals.
- Markdown/JSON reports, subgroup CSV exports, and model-by-task experiment tables without a pooled fairness composite.
- Offline regression tests and real local harness integration tests using a tiny random model. These are software validation, not paper results.

Installation, commands and interpretation are in [fair-uk.md](fair-uk.md). Metric definitions and limitations are in [worst-group-metrics.md](worst-group-metrics.md).

## Dataset release

[WarBias PR #21](https://huggingface.co/datasets/FairForget/WarBias/discussions/21) is merged. HF `main` was verified at `c5aceb6265e8c780549cfc1b8dc623c490b86b4c`. The 13 updated files match the audited extension, and all 40 other existing files, including eight annotation artifacts, retain their previous contents.

Each intersectional language track contains 1,320 rows over 40 source cases and 11 status–gender–age cells. Each cell uses 20 source cases with six evidence/polarity variants. Both people share their age and gender profile. Older women veterans remain an explicit design exclusion. Base and intersectional tracks reuse cases and are reported separately.

The registry retains its original immutable content revisions; a merge does not silently change a benchmark version. Existing HF `FairForget/...` repository names identify data provenance and are not toolkit branding. All registered releases remain pilot, human-unvalidated data. This software work adds no annotator tasks or assignments.

## Work remaining before September 30

| Stage | Work | Completion criterion |
| --- | --- | --- |
| Experiment setup | Choose model checkpoints, immutable revisions and the compute environment; freeze raw prompt protocol and full-run matrix. | A reproducible run manifest for every selected model and task. |
| Full evaluations | Run both WarBias languages and both base/intersectional tracks; evaluate complementary tasks separately. | All expected prediction IDs present; no incomplete cases; task competence and invalid responses visible. |
| Statistical checks | Exercise sparse/high-harm/zero-event settings and investigate bootstrap coverage for the intended claims. | Limitations documented; no unsupported confidence or population-fairness claim. |
| Analysis | Produce model-by-task tables and paired UK–EN comparisons; inspect strongest failures and linguistic confounds. | WarBias-centered results with source-case counts, native scores and uncertainty. |
| Paper freeze | Archive model, dataset, code and output fingerprints; reconcile any existing human corrections through a new versioned release. | Reproducible experiment bundle and accurate validation status. |

The current dataset is a pilot software release; its publication does not certify human validation. Paper model experiments are deferred by the project owner; no substantive model runs are scheduled. Between-model paired inference is a separate extension; the present `compare` command compares languages for one matching model run.

Activation analysis follows the benchmark and evaluation milestone. A probe or activation difference alone will not establish the strength, depth or causal role of bias. That later study should use within-model interventions and controls, with evaluation examples kept separate from any probe training or intervention selection.

## Repository and CI

Repository: [reirei-00/fair-uk](https://github.com/reirei-00/fair-uk). Initial upstream revision: `ad8737ae7fad24cf64e50fc7fc31397bff586b9e`. Upstream MIT attribution is preserved; dataset licenses remain separate.

The focused [CI template](fair-uk-ci.yml) is ready to install at `.github/workflows/fair-uk-tests.yml`. The current GitHub publishing credential lacks workflow permission, so this workflow is not enabled. Local tests and packaging checks are recorded in the [pilot release notes](fair-uk-release.md).
