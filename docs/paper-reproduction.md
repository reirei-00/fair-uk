# Paper reproduction

Fair-UK owns evaluation inputs, model inference, response judging, scoring and
paper tables. FairForget is the broader project repository; its `warbias/`
subproject owns WarBias construction, translation, linguistic review and
unlearning pipelines. The subproject has reusable Python modules, command entry
points, tests and thin notebooks. Install the paired Fair-UK checkout before
installing `fairforget-warbias` from that directory; the upstream `lm_eval`
package alone does not provide these study interfaces. Other FairForget
benchmark workflows and their root dependencies remain separate.

## Frozen evaluation inputs

`data/paper/warbias-evaluation-v3/` contains the exact frozen adapted evaluation
bundle: 186 original triplets and 175 neutral variants per language, matched
language pairs, semantic components and original preparation provenance. Its
`manifest.json` is byte-identical to the collected baseline release. The outer
`release.json` inventories all input files with SHA256, exact IDs and families,
source revision, coverage and qualification. Human validation is pending; these
files are software-verifiable inputs, not final reviewed annotations.

```sh
fair-uk-eval paper validate-release --release data/paper/warbias-evaluation-v3/release.json
```

WarBias source texts are attributed to the FairForget/WarBias release identified
in the manifest. Its source license is CC BY-SA 4.0; retain the included data
license and source card when redistributing derived inputs. The evaluator code
retains its upstream MIT license. Other benchmarks remain pinned in
`lm_eval/fair_uk/datasets.json`; use `fair-uk-eval list` and the maintained
`validate`/`run` commands to fetch/verify each selected immutable source. The larger historical native evaluation bundle is distributed as the separate
`native-evaluation-portable-v1.tar.gz` asset, rather than a large Git payload.
`data/paper/native-evaluation-20261008/catalog.json` records its compressed hash,
44 exact files, task roles, coverage, protocols and preparation-only sources.
It preserves 18 task views plus inventories/pairs. An absolute historical source location is rebased in this new manifest, with both manifest hashes recorded; original saved-run identities remain untouched and are not silently made compatible. The historical StereoSet EN
staging view is explicitly preparation only; including its frozen evidence does
not activate it. This historical bundle is distinct from the primary annotated
WarBias v3 evaluation inputs.

```sh
fair-uk-eval paper materialize-inputs --catalog data/paper/native-evaluation-20261008/catalog.json --archive native-evaluation-portable-v1.tar.gz --output inputs/native-evaluation
fair-uk-eval paper compare-native --before-run runs/base/TASK --after-run runs/after/TASK --dataset-bundle inputs/native-evaluation/manifest.json --output results/native-TASK
```

The native comparator preserves benchmark denominators, matched component
resampling and missing judgment coverage. WinoPron remains UK-only role/order
accuracy and consistency diagnostics. Invalid answer tokens do not become
refusal labels. Training exports cannot reuse evaluation IDs or linked families. Capability batteries
outside the approved study are excluded.

## Artifact handoff

The public standard-library validators in `lm_eval.fair_uk.contracts` define
version 1 of three JSON contracts. `data_release` declares item/language hashes,
semantic-family splits, immutable sources, file hashes and explicit human-review
status. `intervention` declares immutable before/after checkpoints, the exact
training release digest, method, seed, language and objective streams.
`exposure` contains actual optimizer events with stream, source item/hash and
valid loss-token counts. Checkpoints use an immutable remote revision or a
relative file hash inventory. Exposure validation binds the receipt and release,
and rejects development/evaluation/deferred gradients. A valid declaration alone
does not establish that the trainer used it; preserve the emitted event evidence.

FairForget's `fairforget doctor`, release command and training runner use these
public interfaces. Fair-UK never imports private producer code. Dataset creation
and model training cannot happen while scoring saved predictions.

## Evaluate both checkpoints

`configs/paper/models.json` preserves the six registered base-model pins and
their source-configuration digest. Actual intervention checkpoints must be
identified by their own trainer receipts; these base pins do not qualify a
new runtime or method-specific integration.

Use the same frozen bundle, tokenizer, scoring protocol and inference settings
for the base and intervention checkpoint. Collect original **and** neutral runs
for each checkpoint and language. Follow the [WarBias guide](warbias-detection.md)
for the existing `run`, panel report, development calibration and tables commands.

Copy `configs/paper/warbias_before_after.json`, select the pre-registered target
and training language, and supply the four saved-run mappings. Each mapping is
`{"uk": [{"bundle": "relative/manifest.json", "run": "relative/run_dir"}],
"en": [...]}`. Paths resolve relative to that configuration file. The sample
leaves real runs and unresolved study choices null deliberately.

```sh
fair-uk-eval paper compare --config study.json --output results/intervention
```

This primary adapter uses summed prior-normalised margins, in nats, and
component-macro preference. It preserves the raw native diagnostic as a separate
track. The historical per-token NNR draft is not silently substituted. NNR needs
an explicit calibrated tau record, calibration identity, exact frozen eligible
baseline IDs and the primary baseline-item SHA256. Missing calibration yields
an unavailable value. RI additionally needs a practical target reduction and
independent-support/positive-confidence checks. M3 requires the declared
training language and matched paired content. Language contrasts use `lg1`,
`lg2`, `lg2_minus_lg1` with their explicit language mapping. Bias redistribution
is distinct from task utility, refusal and helpfulness.

A report verifies saved input/prediction/run identities and paired inference
settings. It is an observational checkpoint comparison until matching trainer
receipt, release and exposure evidence establish the method lineage. Bind those
artifacts after comparison using `fair-uk-eval paper bind-training --report
results/intervention/intervention.json --intervention trainer/intervention.json
--exposure trainer/exposure.json --release training-data/release.json --output
results/attributed.json`. This verifies both exposure hashes and evaluated
checkpoint identities; an altered checkpoint or release cannot inherit a method
label.

Method attribution additionally requires the trainer's starting-checkpoint
verification: the declared checkpoint reference, verified file inventory and
actual loaded tensor-state fingerprint. Historical declaration-only receipts
remain unverified; the binder does not upgrade them. For primary language
transfer, the declared training language and language slots must agree with
the trainer receipt. Missing language evidence leaves transfer unavailable.

Older runs
without effective tokenizer/input evidence cannot qualify for this stricter
paired adapter; their original provenance is preserved.

## Software and paper evidence

`notebooks/paper/before_after_evaluation.ipynb` demonstrates the public scoring
API using explicitly synthetic records, including a no-update assertion. The
WarBias subproject notebooks (`warbias/notebooks/paper/` in FairForget) demonstrate
construction, translation and a tiny CPU
optimizer step. They are executable software checks, never paper measurements.

```sh
python -m pytest tests/fair_uk tests/submission -q
```

An optional saved-baseline regression takes `FAIR_UK_WARBIAS_GOLDEN` pointing to
the original `v2/`, `v3/`, `runs/` and `runs_neutral/` layout. This verifies the
supplied saved baseline without inference; saved run bundles stay outside Git.
Real post-intervention results, human calibration and model-specific production
training qualification are not supplied by this reorganisation. Add these as
immutable, separately hashed artifacts before claiming paper-result reproduction.

The [export command](warbias-detection.md#paper-artifact-export) includes source,
tests, the compact frozen evaluation inputs, configuration and notebook. Its
complete hash inventory detects changed, missing or unexpected files. Exports
exclude credentials, working outputs and machine-specific orchestration.
