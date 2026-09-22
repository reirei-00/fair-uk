# Fair-UK implementation plan

Updated 2026-09-22. This plan supersedes the earlier experiment-first September plan. The [completion checklist](completion-checklist.md) tracks individual requirements and their current status.

## Delivery target

A model-agnostic evaluation suite built on EleutherAI's harness that lets users select benchmarks, a compatible model/backend and Ukrainian, English or both. Every benchmark reports **its own metrics plus applicable worst-group analysis**. Bilingual runs additionally support comparisons on verified matched cases.

WarBias remains the main paper contribution. The toolkit also covers BBQ, StereoSet and WinoBias Natural/Controlled. No unbiased reference model is required. No new annotator tasks are included. Full model experiments remain paused; this plan concerns implementation, dataset preparation and software validation.

## Current starting point

- Eight public pinned task configurations remain unchanged. A local bundle adds four matched English tasks, enabling twelve tasks and six UK–EN comparisons.
- Checkpoint and generic hosted runners accept user-selected compatible models. Legacy hosted commands retain their historical pilot settings.
- Dataset-specific metrics, additional worst-group calculations, raw predictions, provenance, resumption and offline rescoring exist.
- Paired Ukrainian–English comparison now supports every benchmark family, including dataset-specific metric differences. Four new English counterparts are available through the mechanically validated local bundle; public distribution remains pending.
- The suite launcher and hosted-runner additions are development-branch work associated with draft PR #2, not all part of the tagged pilot release.
- Registered datasets remain pilot, human-unvalidated releases. Existing software checks do not establish content or construct validity.

Implementation progress and evidence are recorded in [the September 22 implementation notes](implementation-20260922.md). Remaining scientific validation and public distribution are tracked separately from completed code.

## Work packages, in dependency order

| Order | Design / code / validation work | Deliverable and acceptance criterion |
| --- | --- | --- |
| 1. Dataset and protocol audit | Audit existing HF evaluation files and original English source mappings. Define task inputs, prompts, scoring, groups, case IDs and language alignment. | Versioned task/protocol specification and English-counterpart coverage map for every family; unmatched or non-equivalent cases identified. |
| 2. Dataset-specific metrics | Finalize formulas and reference checks for WarBias, BBQ, StereoSet and both WinoBias variants. Identify original metrics versus adaptations and diagnostics. | Each dataset has a tested metric set with documented denominators, units, aggregation, ties and invalid-answer handling. |
| 3. English adapters and paired data | Register verified English releases and implement matching adapters for the complementary tasks. Preserve existing WarBias pairing. | Pinned data, explicit pair mappings and structural tests; full-language sets and matched comparison subsets remain distinguishable. |
| 4. Model-agnostic execution | Generalize provider adapters, remove pilot allowlists, declare backend capabilities and centralize configuration. Add language selection, persistent suite manifests, progress, estimates and completeness checks. | A fresh installation can prepare and resume supported model/task/language combinations without private scripts. Unsupported operations fail clearly before inference. |
| 5. Group and language analysis | Finalize additional worst-group reporting. Extend bilingual comparisons to each dataset's own metrics and applicable group summaries. | Reports show UK, EN and consistently signed EN-minus-UK differences, matched support, group identities and validated or explicitly exploratory uncertainty. |
| 6. Verification and release | Complete reference regression tests, metric stress tests, paired-alignment tests, clean-install checks, packaging and CI. Synchronize guides and review the changes for release. | A versioned software release with reproducible fixture outputs and accurate supported/tested/unverified status. |

English-data auditing and metric specification come first because they determine which comparisons are meaningful. Generic backend work can proceed independently once the required model operations are defined. Full bilingual comparison tests depend on the aligned adapters and finalized metrics.

## Dataset-specific metric requirements

| Dataset | Required benchmark-specific outputs | Protocol qualification |
| --- | --- | --- |
| WarBias | Accuracy by evidence condition; stereotype/counter-stereotype/unknown rates; aligned-minus-conflicting accuracy | Our specified QA protocol, with invalid and format diagnostics reported separately |
| BBQ | Ambiguous/disambiguated accuracy and bias scores | Preserve documented formulas and eligible denominators; identify any adapted scoring protocol |
| StereoSet | SS, LMS and ICAT | Preserve documented target aggregation and tie rules; record the sentence-scoring protocol |
| WinoBias Natural | Pro/anti accuracy and signed/absolute gaps, with paired correctness and control diagnostics | Candidate-selection adaptation; distinguish it from original coreference-system F1 evaluation |
| WinoBias Controlled | The same applicable metric family, evaluated separately | Its own fixed-candidate protocol; do not pool it with Natural |

Additional group reports provide worst success/harm, gaps and suitable ratios, with subgroup support and uncertainty. They supplement the dataset-specific outputs. Signed bias scores and balance-at-50 measures must not automatically receive the same min/max-rate treatment.

## Bilingual comparison requirements

- Select Ukrainian, English or both explicitly. Show missing counterparts before execution.
- Use the same model identity and equivalent scoring/prompt settings across languages.
- Compare verified matched source cases, preserving semantic roles even when answer positions differ.
- Report full-language results separately from matched-subset results. Recompute both languages' metrics on the matched subset before taking differences.
- Preserve source-case pairing and shared variants during resampling; recompute nonlinear metrics such as ICAT within each resample.
- Show each dataset's metrics and applicable group outcomes in both languages, then their differences, support and uncertainty.
- Record translation, cultural adaptation and grammatical differences. Results describe observed differences on benchmark cases; they do not isolate a causal effect of language.

## September 30 milestone

Target a reproducible software pilot covering the agreed tasks, metrics and bilingual workflow. Prioritize work packages 1–3 to establish the metric and data contract, followed by execution, reporting and release verification. This is a target, not a claim that all remaining validation fits the available time.

English counterpart availability and semantic equivalence are unresolved dependencies. If a pairing cannot be validated in time, record it as outstanding and expose the limitation; do not silently substitute unmatched English data or mark bilingual coverage complete. Automated checks can validate software and structure, while human/content validation remains a separate evidence requirement.

## Experiments and later work

After explicit authorization to resume model runs: freeze the model/task/language matrix and compute budget, refresh prepared manifests against the final code, execute complete datasets, audit outputs and archive reproducible results. Existing pilots and the stopped attempt remain separate. No experiments or recurring runs are scheduled by this plan.

Between-model statistical superiority claims require separately validated paired comparisons. Activation analysis, probes, interventions, a public leaderboard UI, new benchmark families and a pooled fairness score are outside this milestone. English counterparts of the existing families are inside it.

## References and release context

Implementation details: [suite guide](benchmark-suite.md), [usage](fair-uk.md), [group-metric definitions](worst-group-metrics.md), [completion checklist](completion-checklist.md). Historical validation remains recorded in the [v0.1 pilot release notes](fair-uk-release.md).

Repository: [reirei-00/fair-uk](https://github.com/reirei-00/fair-uk). The fork preserves EleutherAI's MIT attribution; dataset licenses and HF provenance remain separate. The focused [CI template](fair-uk-ci.yml) exists but is not enabled; the earlier publishing attempt lacked workflow permission.
