# OpenAI WarBias pilot

The OpenAI runner evaluates WarBias through Chat Completions with a dated model snapshot. It uses the unchanged WarBias prompt as one user message, no system message, temperature 0, a requested seed, and at most 16 output tokens. The versioned answer parser retains refusals, explanations and ambiguous responses as invalid answers. API failures interrupt the run and preserve completed responses; they are not scored as model answers.

**Status:** the runner has offline tests using a mocked OpenAI SDK transport. A live pilot requires an accessible, funded OpenAI API key. No live API validation or pilot results are claimed yet.

## Setup and cost preview

Use the Fair-UK checkout and a dedicated Python environment:

```sh
pip install -e '.[fair-uk-openai]'
```

Configure `OPENAI_API_KEY` in the local environment. Keep it out of committed files, command-line arguments, reports and chat messages. This runner sends requests only to `https://api.openai.com/v1`.

Preview a run without an API key or model calls:

```sh
fair-uk-eval run-openai \
  --task warbias_intersectional_uk \
  --snapshot gpt-4.1-mini-2025-04-14 \
  --limit-clusters-per-stratum 5 \
  --output results/openai/mini/warbias_intersectional_uk \
  --dry-run
```

Remove `--dry-run` to execute. `--concurrency 4` controls simultaneous requests. `--max-estimated-usd 2` rejects a pending-request estimate above $2 before sending requests. The estimate uses UTF-8 byte counts plus a chat-envelope allowance and no cache discount; it is deliberately conservative and is not a billing guarantee. SDK retries are disabled. Failed or interrupted requests may incur costs that are absent from saved usage records.

## Prepared pilot matrix

| Track | Rows per model and language | Source cases per language |
| --- | ---: | ---: |
| WarBias base | 240 | 40: 20 per status |
| WarBias intersectional subset | 330 | 10: 5 per status, all supported profiles |

Run both Ukrainian and English for `gpt-4.1-2025-04-14` and `gpt-4.1-mini-2025-04-14`. This produces 2,280 responses. The preflight estimate for the pinned data is $3.31 across the eight runs, before any unrecorded failed requests, based on the official prices checked on 2026-09-12. This is a pilot selection of two non-reasoning baselines, not a comparison of the latest OpenAI models.

Base and intersectional rows reuse source cases. Translations, demographic profiles and evidence variants are not independent cases. Five cases per intersectional cell give limited precision; uncertainty remains exploratory. Dataset text is human-unvalidated.

## Outputs and comparisons

Each run writes the standard Fair-UK reports plus:

- The exact API request and raw response per item in `predictions.jsonl`.
- Requested and returned model identities, system fingerprints, finish reasons and token usage.
- API usage and estimated recorded-response cost in `report.json` under `api_usage`.

A refusal with no text is preserved in the raw response and scored as an empty invalid answer. Changing a model, source, code fingerprint or generation setting requires a new output directory. Repeating a completed run makes no new API calls. Interrupted runs resume only missing items after validating saved prompts, responses and snapshot identities.

Use the standard [paired language comparison](fair-uk.md#paired-warbias-uken-comparisons) and [experiment table](fair-uk.md#experiment-tables) commands on these run directories. The comparison checks the API metadata and scores raw responses again. Hosted model weights cannot be independently hashed, and a fixed snapshot, seed and temperature do not guarantee identical repeated outputs. Report the chat transport explicitly when comparing with open-weight raw-prompt runs.

Supported snapshots and prices: [GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1), [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini). Request fields follow the [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

## Versioned answer scoring

New evaluations use `abc_option_text_v2`, accepting unambiguous letters, harmless letter punctuation and exact option text while retaining strict format diagnostics. Use `--answer-policy strict_abc_v1` to reproduce the original parser. The prompt and generation settings are unchanged. See the [answer parsing rules](fair-uk.md#outputs-and-rescoring).
