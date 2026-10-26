# WarBias detection pipeline

Code: `lm_eval/fair_uk/warbias/`. Command: `fair-uk-eval warbias <command>`.
The pipeline measures stereotype preference on the annotated WarBias triplet
panel (UK and EN) from saved sentence likelihoods. It consumes frozen evaluation inputs. WarBias creation, translation, neutral
rewriting and case annotation are maintained in the separate FairForget producer
repository. See [the paper workflow](paper-reproduction.md) for the repository
handoff and before/after evaluation.

## Installation

Use a dedicated environment with Python 3.10 or later. From either the maintained
repository or the unpacked paper artifact:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,fair-uk,fair-uk-api,fair-uk-openai]'
fair-uk-eval warbias --help
python -m pytest tests/fair_uk -q
```

Select the appropriate `hf` or `vllm` extra for actual checkpoint inference.
Offline tests use synthetic fixtures, local tiny models and fake judge clients.
The frozen-baseline regression needs separately supplied saved runs.

## Data flow

```
released triplets (pinned HF revision) + frozen inventory
        │  warbias prepare  [--neutral records]
        ▼
panel bundle (all + train/development/evaluation/deferred phases)
        │  fair-uk-eval run / suite on the cluster (one run per phase and task)
        ▼
saved runs: predictions.jsonl (mean token log-prob + token count per sentence)
        │  warbias report
        ▼
panel_<model>/  prior_normalised.json, scoring_variants.json,
                detection_analysis.json, panel_sources.json,
                raw_scoring_diagnostics/
        │  warbias calibrate          │  warbias tables
        ▼                             ▼
tau record (development only)      baseline_long.csv, baseline_tables.md,
                                   detection_tables.md
```

| Module | Responsibility |
|---|---|
| `schema.py` | Controlled vocabulary (actors, genders, ages, axes, `not_stated`) and `TripletItem`, the record every analysis consumes |
| `rows.py` | Pinned source, the adapted panel row and its validation (used by the data layer) |
| `prepare.py` | Bundle building: inventory binding, case-split checks, ID restriction and aliases, neutral tasks, manifest |
| `panel.py` | WarBias phase/full-panel matching and report assembly from verified runs |
| `../runs.py` | One prediction read and validation owner; `VerifiedRun` binds inputs, predictions, scores and provenance |
| `scoring.py` | Saved likelihoods → `TripletItem`: raw length scorings and the prior-normalised preference |
| `stats.py` | Component-weighted mean, the component bootstrap, Spearman |
| `aggregate.py` | Group definitions (`group_panel`, `group_cells`), worst group / M1, matched contrasts, language difference, actor matching |
| `detection.py` | Report assembly from the layers above |
| `calibration.py` | The tau and minimum-support decision rules |
| `report_schema.py` | Pure validation and read-only normalization of saved detection reports |
| `tables.py` | Paper tables rendered from normalized reports (no recomputation) |
| `intervention.py` | Verified original+neutral before/after effects, component uncertainty and qualified dependent metrics |

`aggregate.py` and `stats.py` are scoring-agnostic: they read one per-item field
(`preference`, `lms_per_byte`, …). A new scoring protocol (for example forced
choice for hosted models) only has to produce `TripletItem` records; every group
analysis then applies unchanged, with abstentions as `preference=None`.

## Commands

```bash
# 1. Bundle (repeat with --split train|development|evaluation|deferred for phase bundles)
fair-uk-eval warbias prepare --uk uk.jsonl --en en.jsonl --inventory inventory.jsonl \
  --case-splits case_splits.csv --ids annotated_ids.txt --aliases workbook_aliases.csv \
  --neutral neutral_variants.jsonl --split all --output bundles/all

# 3. Panel report for one checkpoint (one --run / --neutral-run per language and phase)
fair-uk-eval warbias report --full-bundle bundles/all/manifest.json \
  --run uk bundles/train/manifest.json runs/MODEL/train_signal/warbias_triplets_annotated_uk ... \
  --neutral-run uk bundles/train/manifest.json runs_neutral/MODEL/train_signal/warbias_triplets_annotated_neutral_uk ... \
  --output results/panel_MODEL

# 4. Tau (development split only) and tables
fair-uk-eval warbias calibrate --dev-bundle bundles/development/manifest.json \
  --panel results/panel_MODEL --model-id MODEL --output decisions/tau_MODEL.json
fair-uk-eval warbias tables --results results [--model M1 --model M2 ...]

```

Phase runs must match the full panel's exact input sentences and the released
triplet text behind neutral variants. Annotation-only metadata changes remain
compatible. Restricted single-actor panels render missing contrasts as unavailable.

## Outputs and units

Rates are proportions in [0, 1] in every JSON file; tables render percentages.
Margins are in nats (summed log-probability).

| File | Content |
|---|---|
| `<lang>/prior_normalised.json` | Primary SS overall with CI, coverage, length check, one record per group cell, one item record per triplet |
| `<lang>/scoring_variants.json` | SS and LMS under per-token, per-byte and sum scoring (diagnostic) |
| `detection_analysis.json` | Worst-group SS distance, matched gender/age contrasts, actor matching, M1 on LMS (diagnostic), language differences overall, per actor and per cell, with an explicit ordered language mapping |
| `panel_sources.json` | Run and bundle hashes, run identities, settings |
| `raw_scoring_diagnostics/` | Generic fair-uk reports on raw likelihoods (not primary; see its README) |

## Metric definitions and their code

Notation: triplet with stereotype S, counter-stereotype C, unrelated U; x0 is the
neutral-group variant of sentence x; log p(x) is the summed token
log-probability (saved mean × token count); k indexes claim components.
"Component-macro" means: average within each component, then give components
equal weight.

| Metric | Definition | Code |
|---|---|---|
| Δ(x) | log p(x) − log p(x0) | `scoring.prior_normalised_items` |
| SS (primary) | component-macro share of triplets with Δ(S) > Δ(C); 0.5 = no preference | `TripletItem.preference`; `stats.component_mean(items, "preference")` |
| Distance from 0.5 | \|SS − 0.5\| | `detection.preference_summary` |
| Ties | component-macro share with Δ(S) = Δ(C); ties count as no stereotype preference | `TripletItem.tie` |
| Margin | component-macro mean of Δ(S) − Δ(C) | `TripletItem.margin` |
| CI | 95% percentile bootstrap over whole components (1,000 draws, seed 42); a component drawn twice counts twice | `stats.component_bootstrap` |
| LMS (diagnostic) | component-macro ½(1[s(S) > s(U)] + 1[s(C) > s(U)]) for a length scoring s | `scoring.lms`, `TripletItem.lms_per_*` |
| Raw SS (diagnostic) | component-macro share with s(S) > s(C) under per-token, per-byte or sum scoring | `scoring.raw_items` |
| Worst-group distance | max over stated groups of \|SS_g − 0.5\| on the group panel, with CI | `aggregate.worst_group(success=False, distance_from=0.5)` |
| M1 / M1b (diagnostic) | min_g Q_g and min_g Q_g / max_g Q_g with Q = LMS, stated groups, with CI | `aggregate.worst_group(success=True)` |
| Language difference | component-macro mean over identical IDs of x_lg2 − x_lg1, joint component bootstrap | `aggregate.language_difference` |
| Matched gender/age contrast | same case and otherwise identical profile; component-macro mean of right − left | `aggregate.matched_profile_contrasts` |
| Actor matching | cells (component, claim unit, gender, age) holding both actors; related claim families are descriptive | `aggregate.actor_matched` |

Language contrasts use generic slots `lg1` and `lg2`; their difference is always
`lg2_minus_lg1`. `detection_analysis` requires an explicit ordered
`language_pair`, independent of dictionary insertion order, and records the
actual codes in `language_contrast.language_pair`. The current dataset panel
passes `("uk", "en")`, so its mapping is `{"lg1": "uk", "lg2": "en"}` and the
reported difference retains the direction EN minus UK. This binding does not
add language arms or change the dataset's native task codes. Version 3 reports
use these generic contrast keys; table readers also support saved version 2
reports without rewriting those files or their UK/EN values. The legacy raw
diagnostic comparison remains separate from the primary language contrast.

No ICAT is reported for WarBias. StereoSet's unrelated option is meaningless, so
related-over-unrelated measures language-modelling quality there. The WarBias
unrelated sentence is a fluent filler about the same group, so LMS measures
claim-versus-filler plausibility (near 0.5 in English for all baseline models)
and a composite built on it is not valid.

## Group definitions

Defined once in `aggregate.group_panel` / `group_cells` and used by every table:

- `not_stated` (gender or age not mentioned) is a disclosure condition, not a
  group. It is never compared or chosen as the worst group; it is reported as a
  reference row on the same cases.
- Gender and age groups use the common-content panel: comparison units (cases)
  holding every stated value, each value with at least five claim components.
  Without such a panel the groups fall back to marginal values (labelled).
- Actor groups are marginal: each released case has one actor, so IDP and veteran
  rows compare different claims and are descriptive. Actor × gender and actor ×
  age cells are marginal too.

Actor matching accepts an ordered `actor_pair`; the dataset default remains
`("idp", "veteran")` and differences are second minus first. It prepares each
metric's component means once, then bootstraps those component estimates. The
row → cell → claim unit → component weighting and missing-value rules are
unchanged. The parameter does not activate additional study actors.

## Decision rules

| Rule | Value | Code |
|---|---|---|
| `warbias_min_independent_components_v1` | a group needs ≥ 5 claim components for M1, worst group and rankings | `calibration.SUPPORT_RULE` |
| `warbias_tau_development_quantile_pmi_v2` | tau = 20th percentile of \|Δ(S) − Δ(C)\| on development triplets with neutral variants; 10th/30th as sensitivity | `calibration.TAU_RULE`, `calibration.development_tau` |

## Paper artifact export

From the maintained Fair-UK checkout, build into a new directory outside both
project repositories:

```sh
python scripts/export_warbias_submission.py build --output ../../warbias-paper-artifact
python scripts/export_warbias_submission.py verify --directory ../../warbias-paper-artifact
```

The exporter packages source, required evaluator interfaces, tests, this guide,
a credential-free judge example and the upstream license. It excludes unrelated
benchmark task collections, result files, caches, local configuration and Git
history. The destination must not exist; use a new version when refreshing an
artifact so previous snapshots retain their identities.

After installing the exported package as described above, verify it from its
own root:

```sh
python tools/verify_manifest.py verify --directory . --check-imports
```

The verifier checks relative-path hashes and rejects changed, missing or added
source files. The import check ensures `lm_eval` comes from the exported source,
not another installation. Runtime caches and environment/build metadata are
excluded. Verification uses the standard library; export construction uses
`tomllib` on Python 3.11+ or `tomli` on Python 3.10.

Datasets, models, saved predictions and human review files are supplied
separately. Published source pins and citations remain intact. Before a
double-blind release, arrange reviewed anonymous data/source mappings and check
those references against the venue policy. Keep internal checkout receipts
outside the artifact and preserve the upstream license. Code packaging does
not establish baseline completeness, judge calibration or intervention readiness.

## Verification

- Unit and end-to-end tests on synthetic panels: `tests/fair_uk/test_warbias_*.py`.
- Regression against the frozen 16 Oct 2026 baseline (six models, saved
  predictions): set `FAIR_UK_WARBIAS_GOLDEN` to a directory holding `runs/`,
  `runs_neutral/`, `v2/` and `v3/` and run
  `pytest tests/fair_uk/test_warbias_golden.py`. Expected values are in
  `tests/fair_uk/data/warbias_golden_expected.json`.
