# Gemini WarBias pilot

Use `run-gemini` for WarBias generation through Google's Gemini API. Supported pilot models are `gemini-3.6-flash` and `gemini-3.5-flash-lite`. Each item sends the unchanged WarBias prompt as one user turn with no system instruction, temperature 0, seed 42 by default, minimal thinking and a 16-token output limit.

## Run

```sh
pip install -e '.[fair-uk-gemini]'

fair-uk-eval run-gemini \
  --task warbias_uk \
  --snapshot gemini-3.6-flash \
  --limit-clusters-per-stratum 3 \
  --concurrency 1 \
  --output results/gemini/flash/warbias_uk \
  --dry-run
```

Configure `GEMINI_API_KEY` locally, then remove `--dry-run` to execute. `GOOGLE_API_KEY` is accepted as a fallback. Keys go only in the request header and never in result artifacts. Each completed response is saved individually; repeating a completed run makes no model calls. Use the same output directory to resume an interrupted run with unchanged settings and source/code fingerprints.

The preflight estimate uses UTF-8 byte counts plus a chat-envelope allowance, standard paid-tier prices and no cache discount. `--max-estimated-usd 2` rejects a pending-request estimate above $2. This is a conservative estimate rather than an account billing guarantee. Recorded token usage is reported separately; free-tier access, cache discounts and failed-call charges can change the actual bill. Only transient HTTP 429 responses are retried, at most twice, respecting bounded retry delays. Daily quota failures stop the run.

## Pilot design

Start with three source cases per status, preserving all six evidence/polarity variants. Run the same cases in Ukrainian and English for each model. This gives 36 rows per model and language, or 144 answers for a complete two-model pilot. The preflight estimate for this matrix was approximately $0.09 at standard paid-tier prices on 2026-09-12.

Check the account's quotas before expanding. The first live attempt encountered a five-request-per-minute and twenty-request-per-day free-tier limit for Gemini 3.6 Flash. These are observations for that account, not universal limits. `--concurrency 1` avoids parallel bursts but does not enforce a requests-per-minute ceiling; use pacing around the HTTP client for low-quota accounts. Interrupted checkpoints are retained, and incomplete runs must not be described as a completed bilingual evaluation.

A larger follow-up can cover all 240 base rows and an intersectional subset of 330 rows (five source cases per status) per language/model. That complete matrix requires 2,280 answers; it is deferred until quota capacity is available.

Source cases determine the effective sample size. Variants and translations are not independent observations. Three cases per status give exploratory uncertainty only; the base pilot cannot support age/gender or intersectional conclusions. Model-reviewed, human-unvalidated text also limits construct validity. Do not rank models using unmatched or quota-truncated samples.

## Scoring and provenance

The strict parser accepts only A/B/C after trimming and uppercasing. Explanations, refusals and empty responses are invalid, rather than unknown answers. Provider prompt blocks remain visible in the raw response. HTTP failures are run failures, not model predictions. Thought-marked response parts are retained in the raw API response but excluded from the answer being scored.

Outputs include the usual Fair-UK reports plus exact requests, raw responses, timestamps, returned `modelVersion`, finish reasons and token usage. `MAX_TOKENS`, prompt blocks and any thought-token usage are explicit diagnostics in `report.json`. Minimal thinking does not guarantee that thinking is disabled; truncations must be considered when interpreting the pilot.

Gemini model names in this runner are stable provider identifiers, **not immutable weight snapshots**. Hosted weights cannot be independently hashed, and the provider's returned version may not uniquely identify them. Preserve model versions, code hashes and evaluation dates. Raw-prompt open-weight runs and hosted chat runs also have different input transports.

Use the standard [paired UK–EN comparison](fair-uk.md#paired-warbias-uken-comparisons) and [experiment summary](fair-uk.md#experiment-tables) commands. The comparison validates each saved API request and response before re-scoring.

Official references: [GenerateContent API](https://ai.google.dev/api/generate-content), [thinking settings](https://ai.google.dev/gemini-api/docs/generate-content/thinking), [pricing](https://ai.google.dev/gemini-api/docs/pricing).
