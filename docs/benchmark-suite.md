# Bias benchmark suite

Fair-UK is an extension of [EleutherAI's Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). Its installed entry point is `fair-uk-eval`; the equivalent module command is `python -m lm_eval.fair_uk`. Dataset adapters, scoring, reports and the suite launcher live in `lm_eval/fair_uk/`. No annotation workflow is required.

## Available datasets

`fair-uk-eval list --format table` shows the catalog. Add `--language uk` or `--language en` to filter it. The default JSON format includes immutable HF revisions, filenames, checksums and validation status.

| Dataset | Tasks | Rows per task | Required model output |
| --- | --- | ---: | --- |
| [WarBias base](https://huggingface.co/datasets/FairForget/WarBias) | `warbias_uk`, `warbias_en` | 240 | Generated answer |
| [WarBias intersectional](https://huggingface.co/datasets/FairForget/WarBias) | `warbias_intersectional_uk`, `warbias_intersectional_en` | 1,320 | Generated answer |
| [BBQ-UK](https://huggingface.co/datasets/FairForget/BBQ-UK) | `bbq_uk` | 58,492 | Candidate token log probabilities |
| [StereoSet-UK](https://huggingface.co/datasets/FairForget/StereoSet-UK-Eval) | `stereoset_uk` | 949 | Sentence token log probabilities |
| [WinoBias-UK Natural](https://huggingface.co/datasets/FairForget/WinoBias-UK-Natural) | `winobias_uk_natural` | 1,674 | Candidate token log probabilities |
| [WinoBias-UK Controlled](https://huggingface.co/datasets/FairForget/WinoBias-UK-Controlled) | `winobias_uk_controlled` | 1,580 | Candidate token log probabilities |

All registered releases are pilot, human-unvalidated datasets. The registry is versioned in `lm_eval/fair_uk/datasets.json`; these counts describe those pinned releases, not the latest mutable HF branches. Translations and demographic variants are not independent source cases.

## One model, multiple benchmarks

Preview the default Ukrainian suite without downloading datasets, loading model weights, or invoking an API:

```sh
fair-uk-eval suite
```

The default suite contains the six Ukrainian tasks, totalling 64,255 registered rows. It runs full datasets unless a source-case limit is explicitly supplied. Each task retains its own protocol and dataset-specific scores, supplemented by applicable group comparisons. Inspect formulas with `fair-uk-eval metrics --task TASK`; there is no aggregate fairness score across benchmarks.

Execute only when ready:

```sh
fair-uk-eval suite --execute \
  --model hf \
  --model-args 'pretrained=YOUR_MODEL,revision=IMMUTABLE_40_CHARACTER_COMMIT,device=cuda' \
  --batch-size 8 \
  --output results/my-model
```

A self-contained local model directory can replace the remote model and revision with `pretrained=/absolute/path/to/model,device=cuda`. Use environment variables for access credentials. Each benchmark runs in a separate process to release model memory before the next. Outputs go under `results/my-model/<task>/`; combined tables go under `results/my-model/summary/`. On failure, later tasks stop and completed checkpoints remain available. Repeating the same configuration verifies and reuses completed task artifacts and resumes unfinished tasks. `suite.json` records progress and configuration; changed settings or modified completed artifacts are rejected. Suite termination is forwarded to the active model process group.

To prepare only a bilingual WarBias selection:

```sh
fair-uk-eval suite \
  --tasks warbias_uk warbias_en warbias_intersectional_uk warbias_intersectional_en
```

Alternatively use `--benchmarks warbias warbias_intersectional --languages uk en`. For all six benchmark tracks in both languages, provide the prepared manifest:

```sh
fair-uk-eval suite --languages uk en --dataset-bundle /path/to/manifest.json
```

The local bundle adds four matched English tasks, giving twelve tasks and six language comparisons. It retains all 1,674 Ukrainian Natural WinoBias rows, but only its 558 primary rows have English counterparts; 1,116 Ukrainian grammatical controls remain separate. The bundle is unpublished, derived from pinned Ukrainian releases and audited local English sources. See [source requirements, provenance and limitations](bilingual-data-audit.md). Requesting missing counterparts without a bundle yields an incomplete plan and blocks execution.

Bilingual execution generates each task's full-set report and paired native/group metric reports under `comparisons/`. Pairing uses declared matched cases and rejects missing or incompatible records. Differences use EN minus UK; nonlinear metrics are recomputed in paired source-case bootstrap draws. These intervals remain exploratory. WinoBias comparisons include adaptation differences, so they do not isolate language effects.

Add the model arguments, output directory and `--execute` to run it. The suite accepts `--limit-clusters-per-stratum` for an explicitly partial smoke run, and `--answer-policy strict_abc_v1` for legacy WarBias scoring. The default is `abc_option_text_v2`. For one task, use the existing `fair-uk-eval run --task ...` command.

## Backend support

| Model access | Supported scope | Current status |
| --- | --- | --- |
| Hugging Face causal checkpoint, `--model hf` | All eight tasks | Harness integration tested; exact model compatibility depends on the backend/tokenizer |
| vLLM checkpoint, `--model vllm` | Matching generation/token-likelihood interfaces | Adapter support exists; GPU/vLLM execution is not validated in this release |
| Configurable OpenAI-compatible or Gemini API, `run-api` / suite `--model api` | Generation tasks (currently WarBias) | Arbitrary model IDs, configured endpoint/parameters; mocked integration validated, live checks deferred |
| Legacy `run-openai` / `run-gemini` | Historical WarBias pilot protocols | Fixed pilot model lists retained for reproduction; use the generic runner for new configurations |

A text-only chat API cannot provide the candidate token scores required by the current BBQ, StereoSet and WinoBias protocols. Replacing those measurements with generated answers would be a separately specified benchmark protocol. The multi-task `suite` launcher supports checkpoint and generic hosted backends and rejects unsupported task/backend combinations before inference. See [hosted configuration](hosted-models.md).

## Relationship to the Ukrainian LLM Leaderboard

The related public project located during this review is [lang-uk/ukrainian-llm-leaderboard](https://github.com/lang-uk/ukrainian-llm-leaderboard). Its README describes Ukrainian capability tasks and execution through `lm_eval` with a custom task include path, plus a leaderboard over result files. Fair-UK independently extends the same upstream harness with bias-specific adapters, source-case grouping, worst-group measures and paired WarBias language reports. It has not imported that leaderboard's task suite or become a compatible leaderboard frontend.

Use the leaderboard as a reference for packaging and reporting. Keep fairness metrics, generation settings and dataset versions explicit; a capability score is not a bias score.

## Generation limits and reproducibility

New checkpoint and generic hosted runs default to 128 output tokens. Set `--max-output-tokens` for checkpoint models or `max_output_tokens` in hosted configuration. Generation settings belong to the run identity; existing 16-token pilots retain their original manifests and cannot be silently resumed under new settings. Legacy hosted commands remain unchanged for reproduction.

The suite's preview mode estimates actual requests (BBQ uses nine per row, StereoSet three, and WinoBias two), without downloading datasets or touching model credentials. Hosted costs remain unavailable unless pricing is supplied; a published price estimate is not a hard spending cap.
