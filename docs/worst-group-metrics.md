# Worst-group comparisons for Fair-UK

Intersectional dataset release status: [WarBias PR #21](https://huggingface.co/datasets/FairForget/WarBias/discussions/21) is merged and verified on HF `main`. The registry retains the same immutable content revision.

Design adaptation updated 2026-09-12. The first implementation now provides
rate comparisons, explicit group coverage and exploratory source-case bootstrap
intervals across WarBias QA, BBQ-UK, StereoSet-UK and WinoBias-UK.
See [implemented scope and validation](fair-uk.md). Native-score reference
checks and tiny-model smoke tests pass. Human validation, broader interval-coverage
simulations and paired between-model inference remain outstanding; this document
also describes those planned extensions.

## Source and interpretation

Ghosh, Genuit and Reagan, *Characterizing Intersectional Group Fairness with
Worst-Case Comparisons*, PMLR 142:22–34 (2021), arXiv:2101.01673v5.

- [Paper](https://arxiv.org/html/2101.01673v5)
- [Published version](https://proceedings.mlr.press/v142/ghosh21a.html)

Sections 3.1–3.2 define intersectional subgroups and compare their metric values
through a minimum/maximum ratio. Section 3.3 applies this to outcome and error
rates. Section 3.5 considers maximum pairwise distribution divergence. Section
5.2 identifies sparse intersections and statistical inference as unresolved
limitations.

We adopt the min/max comparison as a disparity diagnostic. Absolute worst-group
performance, gap reporting, task-specific outcome definitions, uncertainty and
degenerate-case policies below are our adaptation. This replaces the proposed
weighted mean–worst composite as the initial design direction.

## Three quantities, with different meanings

For a fixed dataset version, language, protocol and evidence condition, let
`r_g` be the same nonnegative rate in each prespecified group `g`.

1. `minmax_ratio = min_g(r_g) / max_g(r_g)` when the denominator is positive.
   One means equal measured rates. Smaller values mean greater relative
   disparity; this is not an absolute quality or harm score.
2. `group_gap = max_g(r_g) - min_g(r_g)`. The JSON/CSV rate gaps use [0,1] units; multiply by 100 to display percentage points.
3. Absolute worst-group performance: `max_g(r_g)` for an undesirable event,
   or `min_g(r_g)` for a desirable outcome such as correct answers.

Always publish constituent rates, numerator/denominator counts, independent
cluster counts, the identities attaining extrema (including ties), and coverage
status. Retain pooled and group-balanced averages as separate summaries.

For example, hypothetical harmful-choice rates of 0.10 and 0.30 yield a ratio
of 1/3, a gap of 0.20 and worst-group harm of 0.30. Rates of 0.80 and 0.80 yield
ratio 1 and gap 0, but worst-group harm remains 0.80. Equality does not establish
acceptable behavior. These examples are not model results.

Do not silently replace an error rate by its complementary success rate: the
absolute gap is unchanged but the min/max ratio generally changes. Register
the chosen outcome and direction before model evaluation.

## Rate definitions for the first implementation

The table specifies separate outputs, not quantities to combine into one score.

| Task | Rate or value to compare | Required companions and interpretation |
| --- | --- | --- |
| WarBias QA, ambiguous | Stereotype-consistent answers / all ambiguous items in each group | Maximum rate, min/max ratio, gap; also correct-unknown, counter-stereotypical and invalid rates. Correct-unknown + stereotype + counter + invalid must sum to one. |
| WarBias QA, conflicting evidence | Correct answers / all conflicting-evidence items in each group | Minimum accuracy, accuracy ratio and gap; also stereotypical wrong answers, unknowns and invalids. Not all errors are stereotype-driven. |
| WarBias QA, aligned evidence | Correct answers / all aligned-evidence items in each group | Same accuracy summaries, plus the signed aligned-minus-conflicting accuracy difference within each group. That signed difference uses range comparisons, not min/max ratios. |
| BBQ-UK | Unsupported stereotypical-choice rate under ambiguity; disambiguated accuracy, further split by evidence alignment when metadata permits | Derive semantic answer roles using polarity and source metadata. Keep official-style BBQ scores and their eligible denominators separately; do not ratio signed BBQ scores. |
| WinoBias-UK | Coreference accuracy within a fixed construction/type/condition across supported explicit gender groups | Minimum accuracy, accuracy ratio and gap. Keep pro/anti signed and absolute gaps, paired correctness and linguistic controls separately. Natural and Controlled must remain distinct. |
| StereoSet-UK | Native stereotype preference and language-modeling competence per supported target/group | Preserve SS, LMS and ICAT and their reference aggregation. A min/max LMS comparison describes competence disparity. Report signed stereotype preference and the largest absolute distance from the 50% benchmark point; do not call a raw SS ratio a fairness score. |

StereoSet's balance reference is a benchmark convention, not proof that every
individual stereotype/counter pair ought to receive equal probability. For
SS represented on [0,1], `abs(2*SS_g - 1)` is a normalized group-level imbalance
diagnostic. Report its maximum and the original signed preference. It is not
the same construct as an observed QA error rate.

Raw token log-likelihoods and signed score margins cannot use the min/max
parity ratio: they may be negative and are not nonnegative outcome rates.
Keep continuous margins descriptive in the first release. The paper's
distribution-divergence route is a possible later extension; it requires
additional choices about distribution estimation and zero support.

## Semantic classes, not answer letters

Never compare the frequency of literal A/B/C across demographic groups.
Their meanings vary across questions and permutations. Map each response to
roles such as stereotype-consistent, counter-stereotype, unknown and invalid,
using item-specific metadata. Preserve question polarity.

QA answer correctness is not a person's positive social outcome. Name our
metric an accuracy ratio, not demographic parity or equal opportunity unless
we have explicitly defined the corresponding social prediction and gold-label
conditioning. In particular, the paper's Eq. 12 uses output probabilities
conditioned on group without conditioning on the true label in the displayed
formula; copying that expression does not establish conventional equalized
odds for a multiclass QA task.

## Groups and comparability

Define a registry for each dataset rather than a universal cross-benchmark
group list. Distinguish demographic attributes from bias categories, topics,
test conditions and grammatical controls.

- Base WarBias QA (`eval_uk`, `eval_en`) supports IDP and veteran scenario
  groups. The separate `eval_intersectional_uk` and `eval_intersectional_en`
  configurations add 1,320 rows per language across 11 status–gender–age cells:
  men/women at age anchors 25/45/65, retaining the older-women-veteran exclusion.
  Each cell has 120 rows from 20 source cases. Both central people share the
  profile, so this measures status associations across demographic profiles,
  not direct gender or age preference within an item. Generated variants are
  human-unvalidated and have no annotation assignments. Use
  `intersectional_group` for cells, `profile_scenario_id` for six-variant panels
  and `source_case_id` for clustering across every profile and language.
  Compare status summaries over the five shared profiles; separately show
  the extra IDP older-women cell. Different case composition by status remains
  a comparability limit. See the dataset's `evaluation/intersectional/` records.
- BBQ intersections require actual, validated joint target attributes. A
  category called race-by-gender alone does not identify every demographic
  subgroup needed for a demographic parity claim.
- WinoBias exposes explicit gender/occupation-related structure, but occupation
  stereotypes and grammatical controls require separate interpretations.
- StereoSet category labels alone are not demographic intersections. Join
  validated target metadata only where it supplies the required identities.

Compute marginal and supported intersectional reports separately. Do not
infer identity from generic grammatical gender, a speaker's job, or a topic.
Unspecified attributes are not an invented demographic group. Record missing
attributes and excluded comparisons visibly. Do not create unobserved data by
taking a Cartesian product of labels.

Compare rates within the same evidence condition and protocol. When groups
have different scenario/topic composition, their difference is a descriptive
benchmark disparity, not proof of an identity effect. Use matched scenarios
where they exist; otherwise optionally standardize to prespecified common
strata and shared weights, reporting the resulting coverage reduction. Keep
raw and standardized estimates distinctly labeled.

Within a group, prevent repeated wording variants from silently determining
the estimand. For the adapted rates, give independent source cases equal weight
and average their eligible variants within case, unless a different weighting
is prespecified. Report reference item-weighted metrics separately. Record
both raw event counts and the weights used to produce the adapted rate.

## Sparse and degenerate cases

- No eligible items: return an undefined rate and `no_eligible_items`; never 0.
- Fewer than two supported groups: the ratio/gap comparison is undefined.
- All group rates zero: ratio is undefined (`all_rates_zero`), gap is 0.
  Absolute performance remains interpretable: all-zero harm differs from
  all-zero accuracy. Do not add an arbitrary epsilon or silently return 1.
- One zero and one positive rate: ratio is 0. Show sample support and uncertainty.
- Missing registered groups: report coverage as incomplete. Any available-group
  extrema must be labeled as such, not as full-registry intersectional fairness.
- All-invalid answers: task accuracy/validity reveals failure even if the
  stereotypical-answer numerator is zero. Do not call the run fair.
- Prespecify support rules using independent case counts and desired precision.
  Keep sparse groups visible, with inconclusive status; support filtering must
  not silently erase a vulnerable group from the report.

Do not assign a universal pass/fail fairness threshold to these ratios.

## Uncertainty and validation plan

The paper does not supply a completed small-subgroup inference solution. We
must validate ours before making confirmatory rankings.

Use the independent source case/family as the resampling unit, preserving its
variants, paired contexts and model predictions. Report the number of unique
scenarios and source cases. Recompute extrema in each resample rather than
fixing the originally worst group. Never condition the interval on a group
selected after seeing the same results.

For confirmatory reporting, investigate simultaneous intervals for all
registered group rates and derive conservative ratio/gap/extreme bounds from
them. Validate coverage through simulations with equal rates, rare events,
imbalanced group sizes and tied extrema. Ordinary percentile bootstrap output
alone is not a guarantee of valid extreme-value inference. Track resamples
with undefined ratios; do not silently drop them and present a full-confidence
interval. Repeated seeds do not increase the count of independent scenarios.

Prespecify multiplicity handling across reported conditions/group families.
Keep statistical intervals distinct from uncertainty due to translation quality,
construct validity and benchmark representativeness.

## Paired language inference

The `compare` command compares matching WarBias UK and EN runs after validating their model identity, settings, source IDs and semantic metadata. For each condition and group it reports `rate_EN - rate_UK`. It also reports `worst_EN - worst_UK`, where the worst group can differ between languages. All rates use equal source-case weights within each group.

Bootstrap draws share one source-case weight across both languages and every demographic profile, stratified by status. Recomputing each language's extrema within every draw preserves group selection uncertainty. There is no min/max ratio of signed language differences. Intervals remain exploratory and describe these matched benchmark cases; they do not establish a causal language effect or generalization to unseen demographic groups. Paired between-model inference and confirmatory coverage validation remain future work.

## Required metric stress tests

1. Hidden intersections: four equal-sized gender-by-status cells with error
   rates 0.10, 0.90, 0.90, 0.10 have identical marginal rates (0.50). Marginal
   ratios are 1; the intersectional ratio is 1/9 and worst error is 0.90.
2. Equal poor performance: error rates 0.80/0.80 give parity 1, not low harm.
3. Zero events, empty conditions, missing groups and all-invalid predictions
   produce distinct, explicit statuses.
4. Complement sensitivity: error rates 0.10/0.20 produce ratio 0.5; success
   rates 0.90/0.80 produce ratio 8/9. Neither may silently replace the other.
5. Whole-dataset duplication leaves point estimates unchanged and cannot
   create additional independent case IDs or shrink cluster-based uncertainty.
6. Reordering A/B/C together with their semantic indices preserves metrics.
7. Signed margins/BBQ bias scores are rejected by the rate-ratio interface.
8. Adapters reproduce their reference metrics before adapted aggregation.

## Delivery sequence

1. Validate dataset metadata and register explicit groups, outcomes, weighting,
   evidence conditions and cluster identifiers for all four benchmark families.
2. Implement one shared nonnegative-rate comparison helper with the statuses
   above, plus dataset-specific semantic event extraction.
3. Add reference-parity checks and synthetic metric stress tests.
4. Add uncertainty with simulation checks and the group-aware report.
5. Run reviewed datasets and freeze a release with reproducible manifests.

The implementation should expose the adopted min/max method with attribution.
Any research claim concerns its validated application to these bias tasks, including bilingual WarBias, rather than invention of the min/max ratio itself.
