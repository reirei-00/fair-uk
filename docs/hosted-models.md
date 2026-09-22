# Hosted models

The hosted adapter accepts **user-selected model IDs** through an OpenAI-compatible
chat endpoint or Gemini's `generateContent` endpoint. Model names and prices are
configuration, not compatibility allowlists. The older `run-openai` and
`run-gemini` commands retain the original pilot settings for reproducibility;
use `run-api` for new configurations.

Install the `fair-uk-api` extra. Keep the API credential in an environment
variable; the JSON file contains its **variable name**, never its value.

## Configuration

Example `provider.json` for any compatible chat service:

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

The adapter appends `/chat/completions` to the exact base URL. It does not insert
`/v1`. For example, Lapa's base is `https://api.lapathoniia.top`; select the model
using the provider's model ID. OpenAI's default base is
`https://api.openai.com/v1`. Set `output_token_parameter` to
`max_completion_tokens` for a model whose API requires that name.

Gemini configuration:

```json
{
  "provider": "gemini",
  "model": "YOUR_GEMINI_MODEL_ID",
  "api_key_env": "GEMINI_API_KEY",
  "max_output_tokens": 128,
  "parameters": {},
  "requests_per_minute": 24
}
```

The default Gemini base is `https://generativelanguage.googleapis.com/v1beta`.
The adapter appends `/models/MODEL_ID:generateContent` and uses `maxOutputTokens`.

`parameters` holds provider-native generation settings. Supply temperature,
sampling seed, reasoning controls or thinking settings **only when the selected
model supports them**. The adapter supplies none of these by default. For
example, supported Gemini thinking settings belong inside
`parameters.thinkingConfig`; an OpenAI-compatible model might instead accept
`parameters.reasoning_effort`. Provider-default sampling can be stochastic;
record explicit supported settings when repeatability matters. The CLI's
`--seed` controls evaluation sampling/uncertainty and does not implicitly become
a provider generation parameter.

Local validation rejects multiple candidates, streaming, prompt replacements,
tool use, forced output schemas, duplicate output caps and credential fields.
Other model-specific parameter support is checked by the provider on execution;
a configuration preview cannot prove that the provider will accept a model.

Optional configuration:

| Field | Default / meaning |
|---|---|
| `timeout_seconds` | `45`; per-request timeout |
| `max_retries` | `2`; temporary HTTP 429 retries, at most `3` |
| `retry_backoff_seconds` | `15`; used when the provider gives no retry delay, maximum `60` |
| `expected_response_model` | Absent; optional exact check against returned model ID/version |
| `prices` | Absent; `{ "input": ..., "output": ..., "cached_input": ... }` in USD per million tokens; cached input price optional |
| `pricing_source`, `pricing_date` | Optional provenance for user-supplied prices |

Hosted model aliases may resolve to provider versions with different names.
Both requested and returned identifiers are recorded. `expected_response_model`
can enforce a known immutable provider version; an ordinary model name does not
prove the hosted weights are immutable.

## Supported tasks and planning

The current hosted adapter exposes **generation**. It runs WarBias base and
intersectional tasks in Ukrainian and English. BBQ, StereoSet and WinoBias
currently require supplied-candidate log-likelihoods; a chat answer or a logprob
for one generated token cannot substitute for that scoring protocol. Those
tasks use the checkpoint backends described in [the suite guide](benchmark-suite.md).

Preview a bilingual selection without model calls or credential reads:

```bash
fair-uk-eval suite \
  --tasks warbias_uk warbias_en warbias_intersectional_uk warbias_intersectional_en \
  --model api --api-config provider.json --output results/my-model
```

To estimate a single task from its source prompts:

```bash
fair-uk-eval run-api --task warbias_uk --config provider.json \
  --output results/my-model/warbias_uk
```

This also makes no model calls or credential reads. It loads the pinned source
dataset, which may download from Hugging Face if not already cached. Use
`--input /path/to/the/pinned/file` for a local copy; its checksum is still verified.

**Execution is opt-in:** add `--execute` to the chosen command when experiments
are authorized. Keep the same model configuration for the UK and EN runs.
The suite produces matched language comparisons after both tasks finish.

## Evidence, progress and recovery

Each run directory contains:

- `run.json`: source fingerprint, exact model settings, runtime versions and estimate.
- `responses/`: one atomic, immutable request/raw-response checkpoint per item.
- `predictions.jsonl`: derived predictions in source order; reconstructed from
  saved raw checkpoints when necessary.
- `status.json`: state and completed/total counts.

The generated prompt is exactly the registered prompt, in one user message
without a system prompt. Reports retain visible response text, returned model
version, finish reason, truncation, explicit provider refusal/block and token
usage diagnostics. Gemini thought parts are excluded from the scored answer.
Unknown usage remains missing; it is not fabricated as zero usage.

Re-running the same command resumes completed work after checking the manifest,
source rows, exact requests and raw response consistency. A process lock
prevents simultaneous writers. Create a `STOP` file in the run directory to
stop before the next request; an in-flight request may still finish and be saved.

Only temporary HTTP 429 responses receive bounded retries. Daily quota,
budget/authentication and other errors stop the run; provider error bodies are
not printed. Transport errors and HTTP 5xx leave `inflight.json` because the
request may have been processed. Automatic resume then stops to avoid repeating
an uncertain request. Preserve the directory and resolve that item with provider
records before removing an uncertainty marker; do not simply delete the marker
and assume no charge occurred. A malformed successful response is also retained
and blocks automatic repetition.

Missing prices produce an **unavailable** cost, not zero. Estimates use a rough
prompt-byte proxy and the configured output allowance. They are not hard budget
limits or token upper bounds. Recorded usage estimates also exclude unrecorded
failed requests and may not reflect provider-specific reasoning charges or
account discounts.

The adapter is validated with mocked HTTP responses, including checkpoints,
resume, authentication failure, quota failure and uncertain transport. A live
request is still needed to verify a particular provider/model configuration;
implementing this adapter does not launch that experiment.
