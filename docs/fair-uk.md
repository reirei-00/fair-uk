# Fair-UK usage guide

[Run a suite](#run-a-suite) · [Datasets](#registered-datasets-and-protocols) · [Hosted models](#hosted-models) · [Bilingual data](#bilingual-data) · [Outputs](#outputs-and-rescoring) · [Comparisons](#paired-language-comparisons) · [Metrics](worst-group-metrics.md)

Fair-UK evaluates pilot, human-unvalidated datasets without requiring an unbiased reference model. It reads evaluation data only; activation analysis and annotation workflows are outside this tool.

## Run a suite

Install from this checkout in a Python 3.10+ virtual environment:

```sh
pip install -e '.[fair-uk]'
fair-uk-eval list --format table
fair-uk-eval metrics --task bbq_uk
fair-uk-eval validate --task warbias_intersectional_uk
fair-uk-eval suite
```

`list` shows available tasks; its JSON format includes pinned revisions and checksums. Add `--language uk` or `--language en` to filter it. `metrics` shows definitions, units and denominators. `validate` loads and checks one dataset. `suite` previews six Ukrainian tasks (64,255 rows) without downloading data, loading models or reading API credentials.

Run a compatible causal checkpoint by replacing the placeholders:

```sh
fair-uk-eval suite --execute \
  --model hf \
  --model-args 'pretrained=YOUR_MODEL,revision=IMMUTABLE_40_CHARACTER_COMMIT,device=cuda' \
  --batch-size 8 --output results/my-model
```

A self-contained local checkpoint can use `pretrained=/absolute/path/to/model,device=cuda`; its files are hashed. `device=cpu` is also supported. Separate tokenizer overrides, adapters, delta weights and GGUF artifacts are rejected until their provenance is supported. Checkpoint prompts use the raw protocol without silently applying chat templates. The equivalent CLI is `python -m lm_eval.fair_uk`.

| Selection | Arguments |
| --- | --- |
| Default full Ukrainian suite | No task-selection flags |
| Named tasks | `--tasks warbias_uk bbq_uk` |
| Both WarBias tracks in both languages | `--benchmarks warbias warbias_intersectional --languages uk en` |
| All tracks in both languages | `--languages uk en --dataset-bundle /path/to/manifest.json` |
| Partial smoke subset | `--limit-clusters-per-stratum 2` |

A source-case limit preserves complete panels and takes that many cases in **each** stratum: status for WarBias, reporting category for BBQ, lexical target for StereoSet, and split/type for WinoBias. It is not a random row sample. Partial runs are labeled. For one checkpoint task, use `fair-uk-eval run --task TASK` with the same model/output options; this command executes directly.

Checkpoint generation defaults to 128 output tokens; change it with `--max-output-tokens`. Each suite task runs in a separate process, with results under `<output>/<task>/`. Failure stops later tasks; termination is forwarded to the active model process. `suite.json` records progress. Repeating the same suite configuration verifies completed results and resumes unfinished tasks; changed settings or altered tracked results require a new output directory. An interrupted checkpoint chunk is recomputed.

## Registered datasets and protocols

[`datasets.json`](../lm_eval/fair_uk/datasets.json) pins each public dataset's immutable revision, filename, row count and SHA-256. A local `--input` must match that checksum. These counts refer to pinned releases, not mutable HF branches.

| Dataset | Public tasks | Rows per task | Scoring protocol |
| --- | --- | ---: | --- |
| [WarBias base](https://huggingface.co/datasets/FairForget/WarBias) | `warbias_uk`, `warbias_en` | 240 | A/B/C generation; six evidence/polarity variants per case |
| [WarBias intersectional](https://huggingface.co/datasets/FairForget/WarBias) | `warbias_intersectional_uk`, `warbias_intersectional_en` | 1,320 | Same protocol; 11 status–gender–age cells across both statuses |
| [BBQ-UK](https://huggingface.co/datasets/FairForget/BBQ-UK) | `bbq_uk` | 58,492 | Mean answer-token log probability over three cyclic answer orders |
| [StereoSet-UK Eval](https://huggingface.co/datasets/FairForget/StereoSet-UK-Eval) | `stereoset_uk` | 949 | Mean full-sentence causal token log probability |
| [WinoBias-UK Natural](https://huggingface.co/datasets/FairForget/WinoBias-UK-Natural) | `winobias_uk_natural` | 1,674 | Mean A/B token log probability; deterministic target placement |
| [WinoBias-UK Controlled](https://huggingface.co/datasets/FairForget/WinoBias-UK-Controlled) | `winobias_uk_controlled` | 1,580 | Mean A/B token log probability at released candidate positions |

Natural WinoBias covers 279 validation/type1 cases; Controlled covers 790 test cases, with types reported separately. StereoSet covers 949 original items and 79 targets. These subsets and protocols are not interchangeable with full original English benchmarks.

Likelihood adapters use the harness's token-scoring interface, count retokenized prompt suffixes in the continuation, and divide by the actual scored-token count. StereoSet prefixes each sentence with BOS, otherwise PAD or EOS. Inputs exceeding the context window fail rather than silently truncate. BBQ uses nine requests per row, StereoSet three, and WinoBias two.

These tasks run through `fair-uk-eval`; they are not generic `lm-eval --tasks` YAML tasks. Hugging Face integration has offline tests. The vLLM interface is supported with its own dependencies, but GPU/vLLM execution remains unverified. Text-only chat APIs cannot supply the candidate scores required by BBQ, StereoSet or WinoBias.

## Hosted models

Install `pip install -e '.[fair-uk-api]'`. Use `run-api` for any compatible OpenAI-style chat endpoint or Gemini `generateContent` model. Keep credentials in an environment variable; the configuration stores its **name**, never its value.

Example `provider.json`:

```json
{
  "provider": "openai-compatible",
  "model": "YOUR_MODEL_ID",
  "base_url": "https://YOUR_PROVIDER/API_BASE",
  "api_key_env": "MY_PROVIDER_API_KEY",
  "max_output_tokens": 128,
  "output_token_parameter": "max_tokens",
  "parameters": {},
  "requests_per_minute": 24
}
```

The chat adapter appends `/chat/completions` exactly; it does not insert `/v1`. Omitting `base_url` uses `https://api.openai.com/v1`. Use `max_completion_tokens` as `output_token_parameter` when required by the selected model. For Gemini, set `provider` to `gemini`, omit `base_url` and `output_token_parameter`, and use `api_key_env: "GEMINI_API_KEY"`; the default endpoint is `https://generativelanguage.googleapis.com/v1beta/models/MODEL_ID:generateContent` with `maxOutputTokens`.

`parameters` accepts supported provider-native generation settings, such as temperature, sampling seed or reasoning controls. None are set implicitly; defaults can be stochastic. The CLI's `--seed` controls evaluation uncertainty, not hosted generation. Each request contains the unchanged benchmark prompt in one user message without a system message. Multiple candidates, streaming, prompt replacements, tools and forced output schemas are rejected.

| Optional configuration | Default / meaning |
| --- | --- |
| `timeout_seconds` | 45 |
| `max_retries`, `retry_backoff_seconds` | 2 temporary-429 retries; 15-second fallback delay, bounded at 60 seconds |
| `expected_response_model` | Optional exact returned-model/version check |
| `prices` | Optional `input`, `output`, `cached_input` USD per million tokens |
| `pricing_source`, `pricing_date` | Optional price provenance |

Preview one task or a bilingual suite:

```sh
fair-uk-eval run-api --task warbias_uk --config provider.json \
  --output results/my-model/warbias_uk
fair-uk-eval suite --benchmarks warbias warbias_intersectional --languages uk en \
  --model api --api-config provider.json --output results/my-model
```

Add `--execute` to make requests. A single-task preview loads pinned data and may download it from HF; `--input` accepts a checksum-verified local copy. Suite preview does not download data. Missing prices mean unavailable cost, not zero; estimates are neither hard spending caps nor token upper bounds and exclude unrecorded failed-request charges.

Hosted runs save immutable item responses under `responses/`, a source-ordered `predictions.jsonl`, and progress in `status.json`. Resume checks exact source rows, settings, requests and raw responses. A process lock prevents simultaneous writers. A `STOP` file stops before the next request; an in-flight request may finish and be saved.

Only temporary HTTP 429s receive bounded retries. Authentication, daily quota and other errors stop. Transport errors and HTTP 5xx retain `inflight.json`, blocking automatic repetition of an uncertain request; resolve it using provider records before removing it. Malformed successful responses are preserved and stop scoring. Explicit empty answers, refusals and truncations remain scored outcomes with diagnostics. Gemini thought parts are excluded from visible answers. Unknown usage is left missing.

Requested and returned model IDs, finish reasons and usage remain in reports. Hosted aliases are not independently verifiable weight snapshots; matching identifiers and seeds do not guarantee determinism. The adapter has mocked integration tests; each live provider/model configuration still needs validation.

### Legacy pilot commands

`run-openai` and `run-gemini` retain fixed pilot model lists and 16-token generation settings for reproduction; use their `--help` for accepted snapshots/options. Install `.[fair-uk-openai]` or `.[fair-uk-gemini]`, set `OPENAI_API_KEY` or `GEMINI_API_KEY` (`GOOGLE_API_KEY` fallback), and preview before executing:

```sh
fair-uk-eval run-openai --task warbias_uk --snapshot gpt-4.1-mini-2025-04-14 \
  --limit-clusters-per-stratum 2 --output results/legacy-openai --dry-run
fair-uk-eval run-gemini --task warbias_uk --snapshot gemini-3.6-flash \
  --limit-clusters-per-stratum 2 --output results/legacy-gemini --dry-run
```

Removing `--dry-run` executes. `--max-estimated-usd` checks an estimate, not an account spending limit; `--concurrency` does not enforce requests per minute. These commands have their own checkpoint/retry behavior, without the generic runner's uncertainty markers or STOP handling. Saved predictions and report hashes use source order. Their original code/settings manifests cannot be silently changed on resume.

## Bilingual data

WarBias UK/EN tasks are public. The other four English tasks require an **unpublished local bundle**, adding `bbq_en`, `stereoset_en`, `winobias_en_natural` and `winobias_en_controlled`, alongside their matching Ukrainian inputs. This gives twelve tasks and six matched comparisons; requesting missing counterparts without a bundle blocks execution.

| Family | Ukrainian rows | Matched English rows | Exclusions |
| --- | ---: | ---: | --- |
| BBQ | 58,492 | 58,492 | None within the pinned release |
| StereoSet | 949 | 949 | Other original English items are outside this matched subset |
| WinoBias Natural | 1,674 | 558 | 1,116 UK agreement/cross controls have no original EN counterpart |
| WinoBias Controlled | 1,580 | 1,580 | None within the pinned test split |

Reconstruct a bundle from the existing project source package:

```sh
python -m lm_eval.fair_uk.bilingual \
  --source-root /path/to/fairForget --output /path/to/new-empty-bundle
fair-uk-eval suite --languages uk en \
  --dataset-bundle /path/to/new-empty-bundle/manifest.json
```

The source root must preserve the layout in [`bilingual.py`](../lm_eval/fair_uk/bilingual.py): pinned Ukrainian exports, original BBQ/StereoSet sources and source-commit records, and existing WinoBias source/construction records. **This repository alone cannot reconstruct the bundle** until those dependencies or a reviewed bundle are distributed. Preparation performs no inference, translation or downloads and reads no provider credentials.

Preparation matches BBQ category/example IDs and answer metadata, StereoSet original IDs and semantic candidate labels, and WinoBias split/type/line IDs with construction lineage. Both languages retain common scoring/grouping metadata. WinoBias removes gold antecedent brackets; Controlled marks only the queried pronoun. Canonical occupation labels avoid a gold-option formatting cue. Mechanical alignment does not establish translation validity.

The manifest records source/output/pair-map hashes, available source revisions and preparation-code identity; local revisions use `local-sha256:`. Loading verifies hashes, eligible IDs and documented exclusions. Full-language reports retain all UK rows, while paired reports exclude only declared unpaired rows. StereoSet retains the same 79 target groups across languages.

WinoBias Natural's primary variants include Ukrainian grammatical agreement; Controlled uses Ukrainian neutral occupation paraphrases against original English occupation wording. Differences therefore combine language and adaptation. These candidate-selection tasks do not reproduce original coreference F1. English subset scores must not be presented as full original benchmark results.

## Outputs and rescoring

| Artifact | Contents |
| --- | --- |
| `run.json` | Dataset/model/code fingerprints, settings, selected IDs and runtime versions |
| `predictions.jsonl` | Raw generated answers or mean candidate-token scores |
| `records.jsonl` | Semantic item outcomes, source cases, groups and conditions |
| `report.json`, `report.md` | Native metrics, group rates/support, worst groups, gaps, ratios and intervals |
| `groups.csv` | Group rates, numerators, denominators and source-case counts |

Rescore without loading a model, writing to a new directory:

```sh
fair-uk-eval score --task warbias_uk \
  --predictions results/my-model/warbias_uk/predictions.jsonl \
  --bootstrap 2000 --seed 42 --output results/rescored-warbias-uk
```

Repeat the source-case limit and `--dataset-bundle` when applicable. Missing, duplicate or unexpected IDs fail. External predictions are fingerprinted but their model identity is unverified; they cannot replace fingerprinted run directories in paired comparisons.

WarBias records use `{"id":"...","response":"A"}`. Default `abc_option_text_v2` accepts ASCII A/B/C (case/whitespace insensitive), one trailing `.`, `!` or `)`, exact unique option text, or a letter plus `.`, `)` or `:` followed by its matching option. Text normalization covers Unicode composition and whitespace only. Explanations, contradictory labels/text and Cyrillic lookalikes remain invalid. Semantic accuracy and bare-letter format compliance are separate. Use `--answer-policy strict_abc_v1` to reproduce original bare-letter scoring; the generation prompt stays unchanged.

Likelihood records use `{"id":"...","scores":[...]}` with finite **mean scored-token log probabilities**. BBQ expects nine values: cyclic orders `(0,1,2)`, `(1,2,0)`, `(2,0,1)`, then A/B/C within each. WinoBias expects A/B; StereoSet expects stereotype/anti-stereotype/unrelated. Ties and aggregation are defined in the [metrics reference](worst-group-metrics.md).

## Paired language comparisons

Use the same pinned model, code, backend, runtime, generation settings, batch size, seed and case selection in both languages. A bilingual suite creates comparisons automatically; to compare existing runs:

```sh
fair-uk-eval compare --uk-run results/my-model/warbias_uk \
  --en-run results/my-model/warbias_en --bootstrap 2000 --seed 42 \
  --output results/warbias-paired
```

Add `--dataset-bundle` for prepared counterparts. `--uk-input`/`--en-input` accept local pinned sources. Comparison reloads sources and rescores raw predictions under one answer policy; it checks matching IDs, semantic metadata, model identities/settings and known returned hosted versions. Missing version evidence remains explicit.

`comparison.json`/`comparison.md` report **EN minus UK** native scores and group rates, with paired source-case intervals. Nonlinear scores and each language's worst group are recomputed in every draw. Positive means higher English values; improvement depends on the metric. Exact pairing preserves dependence, but does not remove translation/adaptation confounds or establish a causal language effect.

## Experiment tables

```sh
fair-uk-eval summarize --reports results/my-model/warbias_uk/report.json \
  results/my-model/warbias_en/report.json --output results/summary
```

Outputs are `summary.md`, `summary.json`, `coverage.csv`, `native.csv` and `worst_groups.csv`; suites create these automatically. Tasks, languages, models, protocols and metric units stay separate. Source report checksums are retained; duplicate experiment identities fail. No cross-benchmark composite or paired between-model significance test is calculated.

## Local WarBias expansion

The draft expansion is separate from the pinned public releases. Preparation verifies artifact hashes, bilingual identity and scoring metadata, frozen splits and profiles, QA evidence/answer roles, both actor orders and the benign-task rubric. It does not certify human or construct validity.

```sh
fair-uk-eval prepare-warbias \
  --source /path/to/warbias_expansion/artifacts \
  --cases /path/to/warbias_preparation/artifacts/cases.jsonl \
  --output /path/to/new-bundle

fair-uk-eval suite --dataset-bundle /path/to/new-bundle/manifest.json \
  --benchmarks warbias_triplets warbias_cross_actor warbias_expanded_qa warbias_benign \
  --languages uk en
```

Preparation defaults to the frozen **evaluation** split. `--split development`, `train` or `all` creates a different bundle in an empty directory; do not label an all-splits run held-out evaluation. The suite command above only previews execution. Original default-suite tasks are unchanged.

| New track (each has `_uk` and `_en`) | Capability | Native outcomes |
|---|---|---|
| `warbias_triplets` | Candidate token scoring | Adapted SS, LMS, ICAT and ties |
| `warbias_cross_actor` | Candidate token scoring | Same formulas on matched actor-swapped claims |
| `warbias_expanded_qa` | Text generation | Accuracy, stereotype/counter/unknown/invalid rates and evidence gaps |
| `warbias_benign` | Text generation, then rubric judging | Task success, full/any refusal and judgment coverage |

Use `run` with a compatible `hf`/`vllm` backend for sentence scores. The generic `run-api` adapter also supports expanded QA and benign requests. Backend compatibility depends on capabilities, not a model-name list. Each report includes shared-case comparisons in `matched.csv`; the readable report shows up to twelve largest differences, while JSON retains every contrast and its case support.

Benign outputs initially remain **unscored**. Judge saved responses with a separately configured generation model using the same provider configuration format as `run-api`:

```sh
fair-uk-eval judge-benign --task warbias_benign_uk \
  --dataset-bundle /path/to/new-bundle/manifest.json \
  --predictions results/warbias_benign_uk/predictions.jsonl \
  --config judge.json --output results/benign-judge-uk

fair-uk-eval score --task warbias_benign_uk \
  --dataset-bundle /path/to/new-bundle/manifest.json \
  --predictions results/warbias_benign_uk/predictions.jsonl \
  --judgments results/benign-judge-uk/judgments.jsonl \
  --output results/benign-scored-uk
```

The judge command previews requests without accessing credentials; add `--execute` to create judgments before rescoring. Raw responses are preserved. Judgments bind to the exact request, response, criteria and rubric version. Malformed judge output becomes unscorable, and missing judgments lower coverage. No keyword heuristic classifies refusal. This adds no annotator task or assignment.

For UK–EN comparison, pass the original generation run directories to `compare`, plus `--uk-judgments` and `--en-judgments`. Judge identities/settings must match; utility/refusal deltas use only responses scorable in both languages. Automated judging itself requires validation before confirmatory claims. Inspect `content_validation.json` for the selected split's independent cases/tasks; rendered profiles do not increase that support.

## Validation and attribution

Run `pytest tests/fair_uk -q` for offline formula, CLI, checkpoint and tiny local-model integration checks. Live provider configurations, GPU/vLLM execution, human data validation and confirmatory interval coverage remain unverified. The [CI template](fair-uk-ci.yml) is supplied but not installed as a workflow.

Software retains the upstream MIT license. Dataset licenses are separate: WarBias and StereoSet-UK Eval CC BY-SA 4.0, BBQ-UK CC BY 4.0, WinoBias-UK MIT. Bundle sources retain their provenance and license obligations. See the [metrics reference](worst-group-metrics.md) for formula attribution and interpretation limits.
