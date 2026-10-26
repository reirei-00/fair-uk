---
pretty_name: "WarBias"
language:
  - uk
license: cc-by-sa-4.0
tags:
  - ukrainian
  - machine-unlearning
  - bias
  - war-context
  - internally-displaced-people
  - veterans
configs:
  - config_name: uk_unlearning
    data_files:
      - split: train
        path: data/train.csv
---

# WarBias

WarBias is an independently constructed dataset for studying bias connected with war and its
social consequences. The first release provides Ukrainian cases about internally displaced people
(ВПО) and veterans. Each complete item contains a stereotype, counter-stereotype, and unrelated
full sentence for exploratory machine-unlearning research.

## Preliminary release

- 151 conceptual anchors were planned.
- 150 complete triplets were generated with MamayLM-Gemma-3-27B-IT-v2.0.
- One anchor is retained in the review queue after an API connection failure.
- Human review has not been completed.
- `review_required` identifies items selected by automatic checks for focused review.
- `review_required: false` means that automatic checks passed; it does not indicate human approval.
- `dataset_status` is `preliminary_unreviewed` for every published triplet.

The current `uk_unlearning` configuration contains conceptual anchors. The prepared age, gender,
and target-track expansion is intentionally deferred until the anchors have been reviewed. Future
languages and evaluation formats can be added as separate configurations.

## Files

- `data/train.csv`: 150 complete preliminary triplets.
- `review/human_review_required.csv`: 93 flagged triplets and one generation error, with empty
  fields for the human decision, corrections, and notes.
- `metadata/sources.csv`: sources used to define the bias taxonomy and generation constraints.

The review file is an annotation artifact and is excluded from the dataset configuration.

## Main fields

| Field | Description |
| --- | --- |
| `item_id` | Stable item identifier |
| `bias_type_uk` | Ukrainian high-level bias type |
| `category_uk` | Ukrainian topic label |
| `target_uk` | Canonical Ukrainian target label |
| `gender_track` | Gender track of the conceptual anchor |
| `age_track` | Age track of the conceptual anchor |
| `stereotype_uk` | Full stereotype sentence |
| `anti_stereotype_uk` | Full counter-stereotype sentence |
| `unrelated_uk` | Full sentence unrelated to the tested association |
| `review_required` | Automatic flag for focused human review |
| `review_flags` | Semicolon-separated quality-check codes |
| `review_reason_uk` | Ukrainian explanation of the review flags |
| `dataset_status` | Release status; currently `preliminary_unreviewed` |

## Intended use

The triplets are intended for exploratory machine-unlearning experiments. A possible role mapping
is stereotype as forget material, counter-stereotype as matched corrective material, and unrelated
sentence as a control. Experiments also require a broader retain corpus and independent evaluation
data.

This preliminary release must not be treated as a validated benchmark or as evidence about any
person or social group. Generated statements may contain factual, linguistic, or framing errors.

## Method

The taxonomy was created from Ukrainian legal, research, civil-society, and guidance sources. Each
item uses a stereotype, counter-stereotype, and unrelated-sentence triplet. All cases and sentences
in this release were constructed for WarBias. MamayLM-Gemma-3-27B-IT-v2.0 generated the Ukrainian
triplets through the Lapatonia API. A second model pass and deterministic checks assessed sentence
roles, Ukrainian naturalness, target agreement, unsupported specificity, sensitive factual claims,
and cross-item duplication. The resulting flags support human review and do not replace it.

## Creators

The Ukrainian extension, taxonomy, generation pipeline, and cleaning workflow were prepared by
[Viktoriia Makovska](https://orcid.org/0009-0003-7622-9143),
[Oleksandra Ostafiichuk](https://www.linkedin.com/in/oleksandra-ostafiichuk-3837a42aa/),
and [Tetiana Zakharchenko](https://cis.mit.edu/our-people/tetiana-zakharchenko).

## License

This dataset is released under the
[Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/).
Individual publications listed in `metadata/sources.csv` remain subject to their own terms. The
source table documents evidence used for taxonomy design; it does not redistribute those works.

## Citation

Please cite WarBias.

```bibtex
@dataset{makovska_etal_2026_warbias,
  title = {{WarBias}: War-Context Bias Triplets},
  author = {Makovska, Viktoriia and Ostafiichuk, Oleksandra and Zakharchenko, Tetiana},
  year = {2026},
  publisher = {FairForget},
  url = {https://huggingface.co/datasets/FairForget/WarBias}
}

```
