# Bilingual source and alignment audit

The local preparation audit completed on 22 September 2026 without inference,
translation, annotation changes, or publication. It adds English counterparts to
the four benchmark tasks whose public registry previously contained only
Ukrainian. WarBias's existing registered English and Ukrainian tasks remain
available separately.

## Coverage

| Family | Ukrainian rows | Matched English rows | Source clusters | Pairing exclusions |
| --- | ---: | ---: | ---: | --- |
| BBQ | 58,492 | 58,492 | 343 question templates | None |
| StereoSet intrasentence | 949 | 949 | 949 items | 1,157 other original English items are outside the Ukrainian subset |
| WinoBias Natural | 1,674 | 558 | 279 pro/anti panels | 1,116 Ukrainian agreement/cross-control rows have no original English counterpart |
| WinoBias Controlled | 1,580 | 1,580 | 790 pro/anti panels | None within the pinned Ukrainian test split |

English tasks represent the Ukrainian-matched source subset. They are not claims
to evaluate every row or split of the original English datasets. All existing
Ukrainian rows remain in full-language evaluation, including Natural controls;
paired comparisons exclude only rows explicitly marked in the bundle.

## What was checked

- Each Ukrainian input file matches the SHA-256 and row count in the immutable
  public registry before any English preparation occurs.
- BBQ matches `(category, example_id)`, then independently checks the answer key,
  question index, polarity, evidence condition and unknown-answer index.
- StereoSet matches original item IDs and orders English candidates by their
  original semantic labels: stereotype, anti-stereotype, unrelated. English
  sentences are read directly from the original source. Both languages use the
  same 79 target groups for comparable target-macro metrics; English target names
  remain in source metadata.
- WinoBias matches source split, sentence type and original line number. Existing
  construction records independently verify the exported Ukrainian sentences,
  marked spans, candidate order, answer keys and English occupations. Pronoun
  gender and pro/anti condition must match in both languages.
- English WinoBias prompts remove the gold antecedent's source annotation.
  Natural prompts show an unmarked sentence and name the queried pronoun;
  Controlled prompts retain brackets only around the first pronoun in the
  original annotated coreference chain. Additional pronoun brackets are removed.
  Both answer candidates use canonical occupation labels, so determiner or
  capitalization style cannot identify the gold option.
- Eligible UK and English rows must have identical IDs and scoring/grouping
  metadata. Duplicate IDs, missing partners or conflicting labels stop preparation.

These are mechanical integrity checks. They do **not** establish human-validated
translation quality or meaning preservation.

## Interpretation limits

WinoBias Natural's primary `mm`/`ff` variants expose Ukrainian grammatical gender
agreement. Its agreement and cross controls cannot be given fabricated English
translations and called original WinoBias. They remain Ukrainian-only diagnostics.

WinoBias Controlled compares Ukrainian neutral occupation paraphrases with the
original English occupation wording. Its English sentences have not been rewritten
to imitate the Ukrainian construction. A measured difference therefore combines
language and adaptation; it is not a controlled estimate of a language-only
effect. This limitation is recorded in each paired WinoBias row and the bundle
metadata. The English candidate-selection tasks use Fair-UK's documented adapted
protocols, not the original coreference-system F1 protocol.

StereoSet's 949 English items are selected using the pinned Ukrainian item IDs
from 2,106 original intrasentence examples. A score from this subset must not be
presented as an original full-English benchmark score.

## Reproduce the local preparation

```bash
python -m lm_eval.fair_uk.bilingual \
  --source-root /path/to/fairForget \
  --output /path/to/new-empty-bundle
```

This is an explicit recipe for the existing project source package. It requires
the local original BBQ and StereoSet sources, pinned Ukrainian exports and the
existing WinoBias construction/source records in the paths listed in
`bilingual.py`. An external user cannot reconstruct the bundle from the Fair-UK
repository alone until those source dependencies or a reviewed bundle are
distributed. No source is downloaded and no provider credentials are read.

The prepared artifact for this audit is:

```text
outputs/fair_uk_bilingual_prepared_20260922_v4/manifest.json
```

The manifest records source file hashes, original available source commits,
preparation code hash, inherited Ukrainian registry entries and each output file's
hash. Separate pair maps record explicit UK/EN IDs and pairing counts. The loader
checks both dataset and map hashes, complete eligible-ID agreement and documented
exclusions. Local revisions use `local-sha256:` identities, never fictitious
Hugging Face revisions.

Pass this manifest explicitly with `--dataset-bundle`. The public registry is
unchanged. Local English tasks are `bbq_en`, `stereoset_en`,
`winobias_en_natural`, and `winobias_en_controlled`. The bundle also contains
the matching Ukrainian inputs. WarBias continues to use its registered releases.

## Remaining validation and distribution work

- Preserve the pilot/human-unvalidated status in every run and report.
- Review translation equivalence and the WinoBias adaptation limitations before
  making cross-language claims; no additional annotator tasks were created.
- Decide whether to publish the audited English subsets and mapping artifacts,
  with license/provenance documentation, or distribute a source package for local
  reconstruction. This audit did not upload or register a new public dataset.
- Run model experiments only after the implementation is approved as ready.
