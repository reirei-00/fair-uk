# Fair-UK evaluation tool

Fair-UK supports the WarBias study in Ukrainian and English, with BBQ-UK, StereoSet-UK and WinoBias-UK as complementary evaluations. It uses EleutherAI harness model backends and produces a reproducible report with native benchmark scores, subgroup rates, absolute worst-group performance, disparity gaps and minimum/maximum rate ratios.

This first implementation uses **pilot, human-unvalidated HF releases**. It reads evaluation datasets only; it does not create annotation assignments or modify annotation materials. It requires no unbiased reference model. Activation analysis is outside this release.

For hosted models, use the [OpenAI pilot guide](openai-warbias-pilot.md) or [Gemini pilot guide](gemini-warbias-pilot.md).

## Install and run

From this fork, use a dedicated Python environment:

```sh
pip install -e '.[fair-uk]'
fair-uk-eval list
fair-uk-eval validate --task warbias_intersectional_uk
fair-uk-eval run \
  --task warbias_intersectional_uk \
  --model hf \
  --model-args 'pretrained=YOUR_MODEL,revision=IMMUTABLE_40_CHARACTER_COMMIT,device=cuda' \
  --batch-size 8 \
  --output results/warbias_intersectional_uk
```

Replace the model and revision placeholders with an actual model and commit. A self-contained local model directory can be supplied as `pretrained=/path/to/model` without `revision`; its files are hashed. Use `device=cpu` for a CPU run. The equivalent module command is `python -m lm_eval.fair_uk`.

`--limit-clusters-per-stratum 2` provides a smoke subset while retaining complete paired cases. It takes two cases in **each** source stratum, not two rows in total, and labels the report as partial. For StereoSet, strata are lexical targets; for BBQ, they are reporting categories; for WinoBias, split/type; for WarBias, status. There is no random row-level limit.

Repeated runs with the same output directory resume completed predictions only if the model, dataset, protocol, runtime and selected IDs match the saved manifest. Changed model weights or revisions require a new output directory. A chunk is saved atomically after all its candidate requests finish. Interrupted chunks are recomputed. Changing only bootstrap settings does not rerun completed inference.

Hugging Face causal models are integration-tested. Likelihood execution also accepts the harness's vLLM token-scoring interface, but GPU/vLLM execution has not been validated in this release. It requires its own environment and dependencies. Separate tokenizer overrides, adapters, delta weights and GGUF artifacts are rejected until their provenance is supported. Prompts use the documented raw protocol; chat formatting is not silently applied.

## Registered datasets and protocols

Every task has an immutable dataset revision, filename, expected row count and SHA-256 in [`datasets.json`](../lm_eval/fair_uk/datasets.json). Local `--input` files must match that checksum. The registry deliberately pins known releases rather than following mutable `main`.

| Task | Rows | Protocol |
| --- | ---: | --- |
| `warbias_uk`, `warbias_en` | 240 each | Strict generated A/B/C; six context/polarity variants per source case |
| `warbias_intersectional_uk`, `warbias_intersectional_en` | 1,320 each | Same generation protocol, with 11 shared demographic profiles across the two statuses |
| `bbq_uk` | 58,492 | Mean answer-token log probability, averaged over three cyclic answer orders |
| `stereoset_uk` | 949 | Mean full-sentence causal token log probability; target-macro SS, LMS and ICAT |
| `winobias_uk_natural` | 1,674 | Mean A/B token log probability; deterministic target placement; primary and grammatical controls separate |
| `winobias_uk_controlled` | 1,580 | Mean A/B token log probability at fixed released candidate positions; test split, types separate |

The Natural release covers 279 validation/type1 cases. Controlled covers 790 test cases. They are not directly comparable pooled sets. The StereoSet release covers 949 items and 79 targets, not the full original development benchmark.

The intersectional WarBias revision is the verified content of [dataset PR #21](https://huggingface.co/datasets/FairForget/WarBias/discussions/21), now merged into HF `main`. The registry keeps the same immutable content revision. The original HF repository namespaces remain intact for provenance.

For likelihood tasks, the runner uses the harness's `_loglikelihood_tokens` backend interface to preserve exact token boundaries from the reference evaluators. Retokenized prompt suffixes belong to the scored continuation, and scores are divided by the actual number of scored tokens. Full-sentence StereoSet starts from BOS, otherwise PAD or EOS, as in the reference. Inputs that exceed the model context window fail rather than being silently truncated. This backend interface is covered by a smoke test and should be rechecked when updating upstream.

These adapters are accessed through **`fair-uk-eval`**, not registered as generic `lm-eval --tasks` YAML definitions. Upstream task loading and generic multiple-choice defaults do not implement all these protocols. The original `lm-eval` command remains available unchanged.

## Outputs and rescoring

A run writes:

- `run.json`: dataset/model fingerprints, toolkit and harness-interface source hashes, protocol, batch size, selected IDs and runtime versions.
- `predictions.jsonl`: generated text or candidate mean log probabilities, with token counts for likelihood requests.
- `records.jsonl`: semantic per-item results with source-case, grouping and condition metadata.
- `report.json`: native scores, all subgroup rates/counts, pooled and group-macro summaries, worst-group statistics and exploratory 95% intervals.
- `report.md`: native-score and worst-group tables with intervals and coverage.
- `groups.csv`: all subgroup rates, numerators, denominators and source-case counts.

Recalculate a report without loading a model:

```sh
fair-uk-eval score \
  --task warbias_intersectional_uk \
  --predictions results/warbias_intersectional_uk/predictions.jsonl \
  --bootstrap 2000 --seed 42 \
  --output results/warbias_intersectional_uk_rescored
```

For a limited run, repeat the same `--limit-clusters-per-stratum` setting when rescoring. Missing, duplicate and unexpected prediction IDs are errors. External prediction files are identified by checksum; the scorer does not independently certify their model identity.

WarBias external records use `{"id": "...", "response": "A"}`. Only A/B/C after trimming whitespace and uppercasing is accepted. Extra explanations, punctuation, refusals and Cyrillic lookalike letters are invalid, not unknown answers.

Likelihood records use `{"id": "...", "scores": [...]}`. These are **mean scored-token log probabilities**, not probabilities or summed sentence likelihoods. BBQ requires nine values ordered by cyclic display order `(0,1,2)`, `(1,2,0)`, `(2,0,1)`, then A/B/C within each order. WinoBias requires A/B scores; StereoSet requires stereotype/anti-stereotype/unrelated scores. All values must be finite. Tied QA candidates split selection mass; StereoSet preserves the reference strict-greater-than comparisons and separately reports ties.

## Paired WarBias UK–EN comparisons

Run the corresponding UK and EN tasks with the same pinned model, code, backend, runtime, batch size, seed and model arguments, and the same case-limit setting if using a smoke subset. Then compare their output directories:

```sh
fair-uk-eval compare \
  --uk-run results/warbias_intersectional_uk \
  --en-run results/warbias_intersectional_en \
  --bootstrap 2000 --seed 42 \
  --output results/warbias_intersectional_paired
```

The base tasks can be compared in the same way. `--uk-input` and `--en-input` optionally provide checksum-verified local copies of their registered data. The comparison reloads the pinned sources and scores raw predictions; it does not trust cached `records.jsonl` values. It rejects missing or duplicate IDs, different case selections, mismatched answer/group/condition metadata, base/intersectional mixing, and mismatched model identities or run settings. External `score` reports do not certify model identity and cannot replace fingerprinted `run` directories.

`comparison.json` and `comparison.md` report **EN minus UK** rates by condition and group, plus differences between language-specific worst-group rates. Positive means a higher English rate; whether that is better depends on whether the rate measures success or harm. The worst group can change between languages. The bootstrap resamples each source case once per draw within status, retaining both languages, all profiles and both polarities, and recomputes extrema. Exact pairing prevents treating translations as independent samples; it does not remove translation or tokenization confounds or establish a causal language effect.

## Experiment tables

```sh
fair-uk-eval summarize \
  --reports results/warbias_intersectional_uk/report.json \
            results/warbias_intersectional_en/report.json \
            results/bbq_uk/report.json \
  --output results/experiment_summary
```

This produces `summary.md`, `summary.json`, `coverage.csv`, `native.csv` and `worst_groups.csv`. Each task, protocol, language and model remains separate. Native metrics retain their original units; worst-group comparisons use rates in [0, 1]. Run identifiers distinguish revisions or settings that share a model name. Input report checksums and full report contents remain in the summary JSON. Duplicate experiment identities are rejected; these tables do not calculate paired between-model significance or a cross-benchmark fairness score.

## Metrics and grouping

For the same nonnegative rate across registered groups, the tool reports absolute worst-group harm (maximum) or success (minimum), the best–worst gap, and `minimum / maximum`. A ratio of one can coexist with uniformly high harm. A zero/zero ratio is undefined. Signed BBQ bias scores and raw StereoSet preference scores are not fed into this parity ratio.

WarBias reports ambiguous stereotype, counter-stereotype, unknown and invalid rates separately; they sum to one. It reports accuracy by evidence condition, stereotype errors under conflicting evidence, and the aligned-minus-conflicting accuracy difference within each group. The intersectional report also summarizes status over the five profiles shared by both statuses. The additional IDP older-women cell remains visible separately in the full cell report; older women veterans remain an explicit design exclusion.

Both people in an intersectional WarBias item share the same gender and age. The report measures status associations across those profiles, not a direct within-item preference for one gender or age. Age anchors 25/45/65 do not represent entire age ranges. Different source-case composition limits veteran–IDP comparisons.

BBQ native scores use only rows with `bias_score_eligible`, including native accuracy; general accuracy comparisons additionally retain all rows. Its released `target_loc` already defines the scoring target and must not be flipped a second time for positive questions. The reference zero-non-unknown convention is recorded explicitly. BBQ groups are **bias categories**, not inferred demographic intersections.

WinoBias separates pro/anti conditions, types and primary/agreement/cross controls, and reports paired correctness and signed/absolute gaps. Its groups denote grammatical gender. StereoSet retains target-macro SS/LMS/ICAT and reports maximum target SS distance from 50, with group comparisons for LMS. Lexical target disparities are not claimed to be demographic fairness.

## Uncertainty and limits

Within each subgroup, the comparison rate gives every source case equal weight after averaging its eligible variants. Row-level numerators, denominators and pooled rates remain visible. WarBias clusters by `source_case_id`; BBQ conservatively clusters by category/question-template index; WinoBias by split/type/item; StereoSet by original item ID, stratified by lexical target.

The bootstrap resamples whole source cases within source strata, preserving shared profile panels, and recomputes extrema in every replicate. All group rates and eligible-case counts are reported. Missing registered groups are explicit; no all-group interval is emitted when a registered group has fewer than two source cases. The JSON includes the number of usable bootstrap replicates, including the ratio's valid denominator count.

Intervals are **exploratory percentile intervals**, not simultaneous-coverage guarantees or evidence of population-wide fairness. Zero-event data can yield degenerate zero-width intervals. More samples, metric-coverage simulations and human validation are needed before making stronger claims. The `compare` command adds paired UK–EN intervals for one matching model. Paired between-model confidence intervals are not implemented. Languages and benchmark families are never pooled into one fairness score.

## Validation and attribution

Run `pytest tests/fair_uk`. A [GitHub Actions template](fair-uk-ci.yml) is included; CI is not enabled because the publishing credential lacks GitHub workflow permission. An authorized maintainer can install it as `.github/workflows/fair-uk-tests.yml`. Regression tests cover strict parsing, polarity, answer-order mapping, ties, unsupported groups, zero denominators, hidden intersectional disparities, source-case weighting, clustered extrema and token boundaries.

During implementation, all eight registered configurations passed checksum and structure checks. Native scoring was compared with the project reference evaluators on 100 BBQ rows, all 949 StereoSet items and 12 WinoBias Natural rows using constructed scores. A tiny random local causal model exercised generation and all four likelihood adapters. Those are software checks, not meaningful model fairness results. A substantive model case study and human benchmark validation remain future work.

Software retains the upstream MIT license. Dataset text is downloaded from its original HF repository and retains its separate license: WarBias and StereoSet-UK Eval CC BY-SA 4.0, BBQ-UK CC BY 4.0, and WinoBias-UK MIT. No annotation workbook is bundled. The native score definitions follow the Ukrainian reference implementations in the sibling project, and the worst-case comparison adaptation is documented in [worst-group-metrics.md](worst-group-metrics.md), based on [Ghosh et al.](https://proceedings.mlr.press/v142/ghosh21a.html).
