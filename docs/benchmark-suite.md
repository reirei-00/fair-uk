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

The default suite contains the six Ukrainian tasks, totalling 64,255 registered rows. It runs full datasets unless a source-case limit is explicitly supplied. Each task retains its own protocol and scores; there is no aggregate fairness score across benchmarks.

Execute only when ready:

```sh
fair-uk-eval suite --execute \
  --model hf \
  --model-args 'pretrained=YOUR_MODEL,revision=IMMUTABLE_40_CHARACTER_COMMIT,device=cuda' \
  --batch-size 8 \
  --output results/my-model
```

A self-contained local model directory can replace the remote model and revision with `pretrained=/absolute/path/to/model,device=cuda`. Use environment variables for access credentials. Each benchmark runs in a separate process to release model memory before the next. Outputs go under `results/my-model/<task>/`; combined tables go under `results/my-model/summary/`. On failure, later tasks stop and completed checkpoints remain available. Repeating the same configuration resumes through the existing per-task runner.

To prepare only a bilingual WarBias selection:

```sh
fair-uk-eval suite \
  --tasks warbias_uk warbias_en warbias_intersectional_uk warbias_intersectional_en
```

Add the model arguments, output directory and `--execute` to run it. The suite accepts `--limit-clusters-per-stratum` for an explicitly partial smoke run, and `--answer-policy strict_abc_v1` for legacy WarBias scoring. The default is `abc_option_text_v2`. For one task, use the existing `fair-uk-eval run --task ...` command.

## Backend support

| Model access | Supported scope | Current status |
| --- | --- | --- |
| Hugging Face causal checkpoint, `--model hf` | All eight tasks | Harness integration tested; exact model compatibility depends on the backend/tokenizer |
| vLLM checkpoint, `--model vllm` | Matching generation/token-likelihood interfaces | Adapter support exists; GPU/vLLM execution is not validated in this release |
| OpenAI, `run-openai` | WarBias generation | Packaged; live provider validation pending |
| Gemini, `run-gemini` | WarBias generation | Packaged; small live pilot completed |
| Lapa/Mamay hosted API | WarBias generation | Local pilot/prepared scripts exist outside this repository; not yet an installed CLI backend |

A text-only chat API cannot provide the candidate token scores required by the current BBQ, StereoSet and WinoBias protocols. Replacing those measurements with generated answers would be a separately specified benchmark protocol. The multi-task `suite` launcher currently supports checkpoint backends, not the hosted API runners.

## Relationship to the Ukrainian LLM Leaderboard

The related public project located during this review is [lang-uk/ukrainian-llm-leaderboard](https://github.com/lang-uk/ukrainian-llm-leaderboard). Its README describes Ukrainian capability tasks and execution through `lm_eval` with a custom task include path, plus a leaderboard over result files. Fair-UK independently extends the same upstream harness with bias-specific adapters, source-case grouping, worst-group measures and paired WarBias language reports. It has not imported that leaderboard's task suite or become a compatible leaderboard frontend.

Use the leaderboard as a reference for packaging and reporting. Keep fairness metrics, generation settings and dataset versions explicit; a capability score is not a bias score.
