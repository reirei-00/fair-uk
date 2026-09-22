# Fair-UK completion checklist

Implementation and offline validation updated on 2026-09-22. Checked items mean implemented, not scientifically validated or released on `main`. Suite and hosted-runner additions are still on the development branch associated with draft PR #2.

**Target:** a model-agnostic fairness evaluation suite covering WarBias, BBQ-UK, StereoSet-UK and WinoBias-UK Natural/Controlled, with Ukrainian-only, English-only and bilingual evaluation options. WarBias already has registered Ukrainian and English tracks; matched English tracks for the other families are prepared and mechanically validated in an explicit local bundle, pending public distribution and content validation. All benchmark families are part of software completion; WarBias's role as the main paper contribution does not restrict the toolkit. No unbiased reference model is required.

**Model scope:** users choose the model. The toolkit must support compatible models through configurable backends, not a whitelist of model names. Named models used in pilots are examples and validation fixtures, not the scope of the product. Compatibility depends on required operations: text generation or scoring supplied candidate sequences. Checkpoint and generic hosted interfaces accept user-selected compatible models. Historical hosted pilot commands retain their old restrictions for reproducibility; new runs use the configurable API interface.

This checklist authorizes no model experiments. Full runs remain paused. It adds no annotator tasks or assignments.

## Required benchmark and model coverage

| Registered tasks | Required backend operation | Completion requirement |
| --- | --- | --- |
| `warbias_uk`, `warbias_en`, `warbias_intersectional_uk`, `warbias_intersectional_en` | Generate text | Run through any compatible configured generation backend |
| `bbq_uk` | Score supplied answer candidates | Run through any compatible backend providing the token scores required by the protocol |
| `stereoset_uk` | Score supplied sentences | Run through any compatible backend providing the token scores required by the protocol |
| `winobias_uk_natural`, `winobias_uk_controlled` | Score supplied candidates | Run both variants through any compatible backend providing the token scores required by the protocol |

- [x] Declare required capabilities in task metadata and expose them in the dataset catalog and suite preview.
- [x] Accept user-selected compatible model identifiers without editing source code or adding model-name allowlists.
- [x] Validate compatibility before execution and explain unsupported operations. Do not silently omit selected benchmarks or substitute a different scoring protocol.
- [ ] Distinguish supported, tested and unverified backend/task combinations in the documentation. Model-agnostic does not mean every provider exposes every required operation.

## Ukrainian–English evaluation and comparison

**Goal:** make both languages selectable for every benchmark family where valid counterparts can be established, and compare the same model on matched cases. Enabling bilingual evaluation is part of the software scope; running experiments remains deferred.

| Benchmark family | Current English support | Remaining work |
| --- | --- | --- |
| WarBias base and intersectional | Registered English tracks and paired language comparison code | Integrate bilingual selection and comparisons into the full suite workflow; validate generalized provider support |
| BBQ | Matched English adapter in local bundle | Public distribution and content validation remain pending |
| StereoSet | Matched English adapter in local bundle | Public distribution and content validation remain pending |
| WinoBias Natural and Controlled | Matched English adapters in local bundle; Ukrainian-only controls excluded from pairing | Public distribution and interpretation/content validation remain pending |

- [ ] **Audit counterpart availability before assigning task names or counts.** Inspect existing HF releases and source mappings first. Register verified English files with immutable revisions, checksums, licenses and provenance. Any newly prepared evaluation artifacts belong in the versioned dataset workflow; no annotator tasks are added.
- [x] **Create explicit pair mappings.** Record English/Ukrainian item IDs, shared source-case IDs, semantic answer roles, conditions, group attributes and retained/excluded cases. Do not assume equal answer letters or matching row order establishes a pair. Existing WarBias exact-ID checks can remain for its already aligned format.
- [x] **Separate full-language results from matched comparisons.** Report each language's available full-set benchmark scores separately. Calculate paired differences only on verified matched subsets, recomputing both languages' metrics on that same subset. Publish alignment coverage and exclusion reasons. An unmatched English reference set can provide context, but cannot be presented as a paired language effect.
- [x] **Add language selection to suite planning.** Support Ukrainian, English or both; show resolved tasks and counterpart availability before execution. Existing explicit WarBias task selection works today. Do not silently run only one language when both were requested.
- [ ] **Match evaluation settings.** Use the same model identity, corresponding prompt templates, answer policy, scoring protocol and suitable generation configuration across languages. Localize prompt text consistently. Record truncation and other validity diagnostics by language; do not compare raw cross-language token likelihoods as bias scores.
- [x] **Extend comparisons to dataset-specific metrics.** Report Ukrainian and English values plus consistently signed `EN − UK` differences for the comparable WarBias outcomes, BBQ scores, StereoSet SS/LMS/ICAT and WinoBias pro/anti scores and gaps. Label units and interpretation; a positive delta does not have the same meaning for every metric.
- [x] **Add the applicable worst-group comparisons alongside native scores.** Use matched group definitions and common coverage. Report differences for each group and for each language's worst-group value, retaining both worst-group identities because they may differ.
- [ ] **Validate paired uncertainty and language confounds.** Preserve paired translations and shared source cases during resampling; recompute the actual dataset-specific metric, including nonlinear aggregates, on each draw. Test missing pairs, changed gold roles and incompatible protocols. Record translation/cultural adaptation and grammatical differences, and report task competence alongside bias. Describe observed language differences on these cases, rather than treating them as an isolated causal language effect.

**Done when:** users can select either language or both; the suite produces each dataset's own scores and applicable group analysis in each language, plus a comparison report wherever validated matching permits it. Unsupported pairings are explicit. Full English datasets need not have the same size as Ukrainian datasets, but paired analysis must use the same underlying cases.

## Current verification evidence

The focused suite passes 137 tests at this checkpoint, covering metric reference fixtures, local tiny CPU harness integration, mocked hosted APIs, paired native/group reporting, bundle integrity, resume and stop behavior. Synthetic scores were also passed through the real matched BBQ, StereoSet and both WinoBias adapters. No model experiments or live provider requests were made. See the [implementation notes](implementation-20260922.md) for the final verification record and remaining limits.

## Already implemented

- [x] Extend EleutherAI's harness with an isolated `fair-uk-eval` entry point and dataset-specific adapters.
- [x] Register eight pinned task configurations: four WarBias tracks, BBQ-UK, StereoSet-UK and two WinoBias-UK variants. Provide a readable dataset catalog.
- [x] Check dataset checksums, row counts, IDs and complete case structures; select partial runs by whole source cases.
- [x] Provide a checkpoint-model suite launcher with a preview mode, per-task outputs and combined tables.
- [x] Implement native scores, condition-specific worst-group performance, gaps, min/max ratios, group support and exploratory source-case bootstrap intervals.
- [x] Preserve raw predictions and provenance; support checkpoint resumption and offline rescoring with versioned answer parsing and separate format diagnostics.
- [x] Produce Markdown/JSON/CSV reports and paired Ukrainian–English WarBias comparisons.
- [x] Package OpenAI and Gemini WarBias runners. The new configurable OpenAI-compatible adapter supports those endpoint configurations without dedicated model-specific code.
- [x] Have unit tests, mocked provider tests and tiny-model harness integration tests. These are software checks, not full model evaluations.

## Metrics: dataset-specific metrics plus worst-group analysis

Every benchmark must report **its own benchmark-specific metrics first, plus applicable worst-group metrics as an additional analysis**. Both are required for completion. Benchmark metrics retain their own formulas, units and aggregation; a common worst-group score does not replace them.

The current implementation is in [`metrics.py`](../lm_eval/fair_uk/metrics.py), with benchmark adapters in [`data.py`](../lm_eval/fair_uk/data.py) and exports in [`reporting.py`](../lm_eval/fair_uk/reporting.py). The functions exist; the unchecked items below concern final specification, completeness and validation.

### A. Dataset-specific metrics

| Dataset | Benchmark-specific metrics already implemented | Additional diagnostics already implemented |
| --- | --- | --- |
| WarBias UK/EN, base and intersectional | Accuracy by evidence condition; stereotype-consistent, counter-stereotypical and unknown-answer rates; aligned-minus-conflicting accuracy | Invalid-answer rate, strict-format accuracy and format violations |
| BBQ-UK | Accuracy and BBQ bias scores for ambiguous/disambiguated contexts | Eligible-item counts, non-unknown answer mass and zero-denominator convention |
| StereoSet-UK | Stereotype Score (SS), Language Modeling Score (LMS), and ICAT | Stereotype ties and largest target-level SS distance from 50 |
| WinoBias-UK Natural | Pro-stereotypical and anti-stereotypical accuracy; signed and absolute pro–anti accuracy gaps | Paired correctness, ties and available grammatical-control accuracy |
| WinoBias-UK Controlled | Pro-stereotypical and anti-stereotypical accuracy; signed and absolute pro–anti accuracy gaps | Paired correctness, ties and available grammatical-control accuracy |

- [x] Implement benchmark-specific scoring for all five dataset variants listed above, including both WarBias languages/tracks.
- [x] **Specify every metric:** formula, eligible denominator, aggregation, direction, units, ties and invalid-answer handling. Distinguish source-benchmark metrics from our adaptations and added diagnostics.
- [ ] **WarBias:** verify each evidence condition and question polarity, semantic answer roles and the aligned/conflicting gap. Verify paired language differences separately from per-language scores.
- [x] **BBQ-UK:** verify both context-specific bias formulas, accuracy, answer-order permutations, eligible cases and zero-denominator behavior against the documented reference.
- [x] **StereoSet-UK:** verify SS, LMS and ICAT formulas, target aggregation and strict tie conventions against the documented reference.
- [x] **WinoBias-UK Natural:** verify pro/anti accuracy and gaps by supported construction/type; document and test paired correctness and grammatical controls as additional diagnostics.
- [x] **WinoBias-UK Controlled:** independently verify the same applicable metrics under its own fixed-candidate protocol. Keep Natural and Controlled outputs separate.
- [x] **Export the complete metric set:** verify that readable reports, JSON and experiment tables retain each dataset's benchmark-specific metrics, definitions/versions and units, not only accuracy or worst-group summaries.

### B. Additional worst-group metrics

Apply these to explicitly defined outcomes and supported groups within each dataset, retaining the dataset-specific metrics above.

- [x] Implement worst-group success/harm, maximum–minimum gap, minimum/maximum rate ratio and source-case weighting.
- [x] Record worst-group identities, case support, missing coverage and exploratory bootstrap intervals.
- [ ] **Define applicability per dataset:** WarBias condition-specific accuracy/harm; BBQ accuracy and stereotype-selection rates across registered categories; StereoSet group LMS and separately interpreted target imbalance; WinoBias accuracy within comparable construction/condition slices.
- [ ] **Validate the additional calculations:** group membership, weighting, sparse/undefined cases, extrema selection and uncertainty. Do not apply nonnegative-rate ratios directly to signed bias scores or conflate categories with demographic intersections.
- [ ] **Attribute the method:** document the adopted min/max comparison and our task-specific adaptation.

**Metric completion criterion:** every dataset has a specified, tested and exported benchmark-specific metric set **and** its applicable additional worst-group analysis. Reference parity and uncertainty checks are detailed in section 3.

## 1. Finalize the evaluation design

- [ ] **Freeze the protocol per task.** Specify prompt/chat-template behavior, likelihood normalization, answer parsing, generation limits and refusal/truncation handling. Resolve the current 16-token packaged runners versus 128-token prepared API script. Record changes in versioned manifests; preserve prior results under their original settings.
- [ ] **Freeze the primary outcomes.** For WarBias, designate ambiguous-context stereotype selection and conflicting-evidence accuracy as primary candidates; retain unknown, counter-stereotypical, invalid and format rates as companions. Decide before full runs which comparisons support the paper. Keep benchmark scores separate rather than adding a single fairness composite.
- [ ] **Freeze group definitions and comparable summaries.** Document status × gender × age, the 11 supported cells and the older-women-veteran exclusion. Define status summaries over the five shared profiles, with the extra IDP older-women cell separate. Specify weights for marginal age/gender summaries if included. Both people share the profile, so these cases measure status associations across profiles, not direct within-item age/gender preference.
- [ ] **Specify groups for the other benchmarks.** Register BBQ bias categories, StereoSet targets and WinoBias's explicit gender groups within comparable conditions. Document what each comparison measures; do not reuse WarBias's demographic labels or infer unsupported intersections.
- [ ] **Define support and inference rules.** Set rules using independent source-case counts, not expanded row counts. Distinguish missing, excluded, sparse and undefined comparisons. Decide whether the release offers exploratory intervals only or supports confirmatory comparisons; no universal fairness pass/fail threshold is needed.

**Done when:** one versioned protocol specification determines the prompt, scoring, groups, weighting and interpretation of every reported result.

## 2. Finish the runner and reporting code

- [x] **Generalize hosted model access.** Accept a user-selected model ID, endpoint and supported generation settings through provider adapters, including an OpenAI-compatible endpoint adapter. Remove pilot model-name allowlists as execution gates; keep pricing metadata optional and separate from compatibility. Integrate existing Lapa/Mamay access as examples of that configurable adapter, not dedicated model-specific implementations. Preserve environment-only credentials, provider/model provenance, metering, bounded retries and resumable checkpoints.
- [x] **Separate model access from scoring.** Backends return raw generations or candidate scores in a common format. Dataset adapters define inputs; metric functions consume standardized predictions independently of model/provider names. Extend the existing separation to all hosted paths and retain offline rescoring.
- [x] **Route all benchmarks by backend capabilities.** Declare the required operations per task and check them before requests. Support generation tasks through generation-capable adapters and candidate-scoring tasks through likelihood-capable adapters. Under the current protocols WarBias uses generation; BBQ, StereoSet and WinoBias require scoring supplied candidates. A generated-answer version of those tasks would require a separately designed and validated protocol. Ordinary generated-token log probabilities alone do not establish candidate-scoring support.
- [x] **Centralize generation configuration.** Expose the agreed output limit and relevant model settings consistently, include them in resume identity, and report truncation/refusal separately where provider metadata permits. Do not selectively regenerate only wrong answers.
- [x] **Add a persistent suite manifest and completion check.** Record expected tasks, settings and output identities; distinguish prepared, partial, failed and complete runs. Validate all expected prediction IDs and complete source cases before declaring a suite complete. The suite now persists identity/progress, verifies completion and artifact hashes, reuses completed tasks, and forwards termination to its child process group.
- [ ] **Complete agreed subgroup views.** Expose shared-profile status comparisons and any prespecified marginal reports alongside intersectional cells. Make exclusions, source-case counts, missing coverage and the worst group's identity visible in the readable report, not only JSON. Audit existing native summaries before adding duplicate calculations.
- [x] **Add preflight work estimates.** Report rows, source cases and actual model-request counts; estimate tokens/cost only where supported by verified provider information. Separate estimates from observed usage. Likelihood tasks may require multiple requests per row.

**Done when:** a fresh installation can prepare, execute, interrupt, resume, rescore and summarize each advertised backend/task combination without private scripts.

## 3. Validate the measurements

- [ ] **Complete reference-score regression coverage.** Turn existing reference parity checks into durable fixtures covering BBQ polarity and permutations, StereoSet ties/target aggregation, and both WinoBias variants. Check tokenizer boundaries and mean-token normalization. Label translation/protocol adaptations explicitly; matching an adapted reference does not establish equivalence to every original benchmark protocol.
- [ ] **Expand metric stress tests where coverage is missing.** Audit existing tests first. Cover hidden intersections, equal poor performance, all-invalid responses, missing cells, zero events, tied extrema and duplicated variants. Verify that variant duplication cannot increase independent support or spuriously narrow intervals.
- [ ] **Validate uncertainty for the intended claims.** Simulate rare events, correlated profiles, unequal support and changes in the worst group. Check interval coverage and undefined bootstrap draws. Keep intervals exploratory unless stronger coverage evidence supports confirmatory use; implement simultaneous bounds/multiplicity handling only if those claims require them.
- [ ] **Audit the registered HF data and existing validation evidence.** Check bilingual semantic alignment, answer keys, demographic consistency, grammatical cues, duplicate/leaking IDs and provenance/licenses. Separate automatic structural checks from human/content validation. Use existing reviewed corrections where available; publish changes as a new pinned version. No new annotator evaluations are part of this work, so absent human validation remains a stated limitation.
- [ ] **Verify supported backends end to end.** Run the full focused software test suite and a clean-install smoke check across all task families. Test configuration with model IDs outside the former pilot allowlists and cover incompatible capabilities, interrupted runs and resume mismatches. Validate vLLM on suitable hardware before advertising it as verified. Live provider checks remain deferred until model runs are authorized; retain mocked coverage in the meantime.

**Done when:** release claims match recorded evidence, and each advertised score can be traced from saved model output through a tested calculation.

## 4. Release and usability

- [ ] **Synchronize documentation.** Keep the README short; place protocol detail here and in linked guides. Update roadmap/release notes that still describe only strict parsing or older test counts. Clearly separate tagged-release capabilities from development-branch additions.
- [ ] **Enable focused CI and verify packaging.** Install the existing workflow template when repository permissions permit, obtain a successful run, and verify the wheel includes the registry and entry points. Check optional provider dependencies in clean environments.
- [ ] **Review and publish a versioned release.** Review the pending changes, merge through the project workflow, tag the tested revision and document upstream attribution, dataset licenses, supported backends and known limitations. Archive a reproducible software example using fixtures; no substantive model result is required for a software release.

**Done when:** another researcher can install a named version and reproduce the documented workflow and fixture outputs.

## 5. Paper experiments — later, after authorization

- [ ] Freeze the model/task/language matrix, model revisions or available provider identities, compute, budget and generation settings.
- [ ] Refresh the prepared full-run manifests against the final code; preserve old pilots and the stopped attempt separately.
- [ ] Run the authorized full datasets, verify completeness, and produce WarBias-centered tables with complementary benchmark results kept distinct.
- [ ] Inspect failures and linguistic confounds, then archive raw outputs, manifests, metric versions and the analysis. Avoid treating translations and demographic variants as independent observations.
- [ ] If making statistical between-model superiority claims, implement and validate paired source-case comparisons. The current paired comparison command is for languages within one model.

## Scope and order for September

Follow the [implementation plan](fair-uk-roadmap.md): dataset/protocol audit and dataset-specific metrics → English adapters and verified pair mappings → generic backend configuration and bilingual suite execution → native/group metric reports and paired comparisons → validation → release. Full paper experiments are a separate milestone and remain paused.

A complete **software pilot** is possible while dataset human validation remains pending, provided it is labeled accurately. A **scientifically validated benchmark** requires evidence for content and construct validity in addition to working code; automatic checks cannot provide that by themselves.

Activation analysis, probes, interventions, a public leaderboard website, new benchmark families and an aggregate fairness score are outside this completion milestone. They are not prerequisites for the evaluation suite.

English counterparts of the existing benchmark families are included in this milestone; unresolved counterpart availability or alignment must remain visible as incomplete work.
