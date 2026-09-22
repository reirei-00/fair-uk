# Implementation and verification — 22 September 2026

The evaluation workflow now supports user-selected compatible models, every existing benchmark family, dataset-specific metrics plus additional group analysis, and matched Ukrainian–English comparisons. This is development-branch software, not a new published release or a claim of scientifically validated fairness measurement.

## Implemented

- A versioned metric catalog exposed by `fair-uk-eval metrics`. Reports retain each dataset's own formulas, denominators and units, with additional worst-group analysis in a separate section.
- Completed BBQ all-row accuracy alongside eligible bias scores, category-macro summaries, StereoSet target/category SS/LMS/ICAT, and WinoBias stratum-macro summaries for both variants.
- Configurable hosted generation for OpenAI-compatible endpoints and Gemini, with arbitrary model IDs. Endpoint, generation settings and optional pricing are configuration, not model-name gates. Historical pilot commands remain for reproduction.
- Dataset capability checks and language selection in suite planning. The default public catalog still contains eight tasks; an explicit local bundle enables twelve tasks and six bilingual comparisons.
- Source-matched English subsets for BBQ, StereoSet, WinoBias Natural and WinoBias Controlled. Every prepared file and pair map is checksummed. No public HF file or annotation artifact was changed.
- Native metric and group-level language differences on matched cases, with paired source-case resampling and recomputation of nonlinear metrics. Known mixed/different returned hosted model versions are rejected; missing version evidence is explicit.
- Persistent suite identity/progress, complete prediction-ID verification, artifact hashes, reuse of completed tasks, and termination forwarding to active child processes.
- Hosted raw response checkpoints, request verification, resume, usage/refusal/truncation diagnostics, rate limits and bounded temporary-quota retry. Uncertain transport outcomes require resolution before repeating a possibly charged request.
- New checkpoint and generic hosted generation defaults to 128 output tokens. Original 16-token pilot settings are preserved under their original run identities.

## Verification

- **137 focused tests passed**, including hand-calculated benchmark metric cases, paired nonlinear metric tests, missing/misaligned pairs, local tiny CPU harness integration, mocked hosted provider/CLI workflows, checkpoint resume and stop handling.
- Lint, formatting and whitespace checks passed.
- All eight files in the local complementary bilingual bundle passed checksum, row-count, case-structure and pair-map validation. Registered WarBias sources remain unchanged.
- Synthetic candidate scores exercised both languages through the actual dataset adapters, native metrics, group comparisons and paired reports: 5,712 matched BBQ rows; 158 StereoSet rows; four primary Natural WinoBias rows with the corresponding Ukrainian controls retained in the input; eight Controlled WinoBias rows. These are software checks, not model results.
- A wheel was built and installed outside the checkout for CLI/catalog/planning checks. Runtime dependencies were inherited from the existing environment; this is not a fully isolated dependency-resolution test.

The local Transformers installation attempted to initialize an unrelated MLX/Metal runtime during the CPU fixture test. The test fixture now disables optional MLX detection for that CPU-only test. No production backend behavior was changed to bypass a model failure.

## Remaining boundaries

- The complementary English bundle is local and unpublished. An external user cannot reconstruct it from this repository alone without the documented source package. Public dataset distribution, CI activation, merge and a new release remain outstanding.
- Dataset translation/content validity remains human-unvalidated. Automatic source matching is not proof of semantic equivalence. No new annotator tasks were created.
- Natural WinoBias has 558 English counterparts for 1,674 Ukrainian rows. Its 1,116 Ukrainian-only controls remain separate. Controlled WinoBias compares original English occupation wording with Ukrainian neutral paraphrases; observed differences include adaptation.
- Intervals remain exploratory. This work does not establish confirmatory coverage, population fairness or between-model superiority. Native interval draws and group extreme draws are not simultaneous inference across the full metric catalog.
- vLLM/GPU execution and live compatibility of newly configured hosted models remain unverified. A compatible interface does not establish that every model accepts every generation parameter.
- Hosted prices are optional estimates. Unknown cost remains unavailable; no hard monetary spending guarantee is claimed.

No live model provider requests or substantive model experiments were run during implementation. Existing pilots and the previously stopped full run remain separate. Refresh experiment manifests against the final code and confirm the model/configuration matrix before starting experiments.
