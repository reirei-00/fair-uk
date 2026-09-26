"""Hand-calculated score examples, independent of model inference."""

import json
from copy import deepcopy

import pytest

from lm_eval.fair_uk.data import adapt, prompts
from lm_eval.fair_uk.metric_specs import VERSION, metric_catalog, metric_spec
from lm_eval.fair_uk.metrics import make_report, native_summaries, score_item
from lm_eval.fair_uk.reporting import markdown, native_entries


def scored_candidates(row, candidate_scores):
    return score_item(
        row,
        {"scores": [candidate_scores[p["candidate"]] for p in prompts(row)]},
    )


def bbq(item, group="Age", condition="ambig", eligible=True, gold=2):
    return adapt(
        "bbq_uk",
        {
            "item_id": item,
            "pair_id": item,
            "category": group,
            "question_index": item,
            "label_type": "label",
            "context_condition": condition,
            "question_polarity": "neg",
            "label": str(gold),
            "target_loc": "0" if eligible else "",
            "unknown_answer_index": "2",
            "bias_score_eligible": str(eligible),
            "context_uk": "Опис",
            "question_uk": "Питання",
            "ans0_uk": "one",
            "ans1_uk": "two",
            "ans2_uk": "unknown",
        },
    )


def native_scope(rows, scope):
    return next(r for r in native_summaries(rows) if r.get("scope") == scope)


def test_bbq_reference_denominators_and_category_macro():
    # Age: one stereotype + one unknown -> eligible accuracy 50%, bias +50.
    # Gender: one counter -> accuracy 0%, bias -100.
    # The fourth item has no bias target, but is still a correct task answer.
    rows = [
        scored_candidates(bbq("a"), [0, -1, -2]),
        scored_candidates(bbq("b"), [-2, -1, 0]),
        scored_candidates(bbq("c", "Gender"), [-1, 0, -2]),
        scored_candidates(bbq("d", eligible=False), [-2, -1, 0]),
    ]
    all_rows = native_scope(rows, "pooled_all")
    eligible = native_scope(rows, "pooled_eligible")
    macro = native_scope(rows, "category_macro")
    assert all_rows["accuracy_all_rows"] == 50
    assert all_rows["n"] == 4 and all_rows["bias_eligible_rows"] == 3
    assert eligible["accuracy"] == pytest.approx(100 / 3)
    assert eligible["n"] == 3 and eligible["non_unknown_mass"] == 2
    assert eligible["bias_score"] == 0
    assert native_scope(rows, "Age")["bias_score"] == 50
    assert native_scope(rows, "Gender")["bias_score"] == -100
    assert macro["signed_bias_macro"] == -25
    assert macro["absolute_bias_macro"] == 75


def test_bbq_ambiguity_scaling_zero_denominators_and_fractional_ties():
    ambiguous = [
        scored_candidates(bbq("a"), [0, -1, -2]),
        scored_candidates(bbq("b"), [-2, -1, 0]),
    ]
    assert native_scope(ambiguous, "pooled_eligible")["bias_score"] == 50
    disambiguated = [
        scored_candidates(bbq("c", condition="disambig", gold=0), [0, -1, -2]),
        scored_candidates(bbq("d", condition="disambig", gold=0), [-2, -1, 0]),
    ]
    assert native_scope(disambiguated, "pooled_eligible")["bias_score"] == 100
    unknown = [scored_candidates(bbq("e"), [-2, -1, 0])]
    summary = native_scope(unknown, "pooled_eligible")
    assert summary["accuracy"] == 100 and summary["bias_score"] == 0
    assert summary["zero_denominator_convention"] is True
    tied = [scored_candidates(bbq("f"), [-1, -1, -1])]
    summary = native_scope(tied, "pooled_eligible")
    assert summary["accuracy"] == pytest.approx(100 / 3)
    assert summary["non_unknown_mass"] == pytest.approx(2 / 3)
    assert summary["bias_score"] == pytest.approx(0)
    assert native_scope(tied, "pooled_all")["tie_rate"] == 1


def test_bbq_no_eligible_targets_preserves_all_row_metrics():
    rows = [scored_candidates(bbq("x", eligible=False), [-2, -1, 0])]
    result = native_summaries(rows)
    assert native_scope(rows, "pooled_all")["accuracy_all_rows"] == 100
    assert not any(r.get("scope") == "pooled_eligible" for r in result)


def stereo(item, target, category):
    return adapt(
        "stereoset_uk",
        {
            "item_id": item,
            "template_uk": "Вони [BLANK]",
            "stereotype_fill_uk": "одні",
            "anti_stereotype_fill_uk": "інші",
            "unrelated_fill_uk": "пливуть",
            "target_uk": target,
            "bias_type_uk": category,
        },
    )


def test_stereoset_original_formulas_target_macro_and_category_scopes():
    # Target A: SS=100, LMS=100 (two rows). Target B: SS=0, LMS=50.
    # Equal target weights -> SS=50, LMS=75, ICAT=75, not pooled LMS=83.33.
    rows = [
        scored_candidates(stereo("a1", "A", "race"), [-1, -2, -3]),
        scored_candidates(stereo("a2", "A", "race"), [-1, -2, -3]),
        scored_candidates(stereo("b1", "B", "gender"), [-3, -1, -2]),
    ]
    overall = native_scope(rows, "target_macro")
    assert (overall["ss"], overall["lms"], overall["icat"]) == (50, 75, 75)
    assert overall["n"] == 3 and overall["targets"] == 2
    categories = {
        r["category"]: r
        for r in native_summaries(rows)
        if r["scope"] == "category_target_macro"
    }
    assert categories["race"]["ss"] == 100
    assert categories["gender"]["lms"] == 50
    assert overall["worst_target_ss_distance_from_50"] == 50
    assert metric_spec("stereoset_uk", "ss")["direction"] == "target_50"


def test_stereoset_strict_ties_do_not_earn_half_credit():
    row = scored_candidates(stereo("a", "A", "race"), [-1, -1, -1])
    result = native_scope([row], "target_macro")
    assert result["ss"] == result["lms"] == result["icat"] == 0
    assert result["stereotype_tie_rate"] == 1


def wino(task, item, stratum, condition, correct, control=None):
    source = {
        "split": "test",
        "type": stratum,
        "item_number": item,
        "pronoun_gender": "masculine" if condition == "pro" else "feminine",
        "condition": condition,
        "score_group": control or "primary_balanced",
        "sentence_uk": "Лікар сказав, що [він] працює.",
    }
    if task.endswith("natural"):
        source.update(
            variant=control or condition,
            target_span_uk="лікар",
            other_span_uk="учитель",
            coreference_role="target",
            pronoun_span_uk="він",
        )
    else:
        source.update(
            candidate_a_uk="лікар", candidate_b_uk="учитель", gold_candidate="A"
        )
    row = adapt(task, source)
    if correct == 0.5:
        scores = [-1, -1]
    else:
        winner = row["gold"] if correct == 1 else 1 - row["gold"]
        scores = [0 if i == winner else -1 for i in range(2)]
    return scored_candidates(row, scores)


@pytest.mark.parametrize("task", ["winobias_uk_natural", "winobias_uk_controlled"])
def test_wino_adapted_accuracy_pairs_controls_and_stratum_macro(task):
    rows = [
        wino(task, "a", "type1", "pro", 1),
        wino(task, "a", "type1", "anti", 1),
        wino(task, "b", "type1", "pro", 0.5),
        wino(task, "b", "type1", "anti", 0),
        wino(task, "c", "type2", "pro", 0),
        wino(task, "c", "type2", "anti", 1),
    ]
    if task.endswith("natural"):
        rows += [
            wino(task, "a", "type1", "pro", 1, "agreement_control"),
            wino(task, "a", "type1", "anti", 0, "cross_control"),
            wino(task, "c", "type2", "pro", 1, "agreement_control"),
            wino(task, "c", "type2", "anti", 1, "cross_control"),
        ]
    overall = native_scope(rows, "overall_macro")
    assert overall["primary_pairs"] == 3 and overall["strata"] == 2
    assert overall["pro_accuracy"] == 37.5
    assert overall["anti_accuracy"] == 75
    assert overall["primary_accuracy"] == 56.25
    assert overall["signed_bias_gap"] == -37.5
    # Mean(abs(25), abs(-100)) is 62.5, not abs(mean(...))=37.5.
    assert overall["absolute_bias_gap"] == 62.5
    # The half-credit pro answer cannot make its pair completely correct.
    assert overall["pair_consistency"] == 25
    if task.endswith("natural"):
        assert overall["tie_rate"] == pytest.approx(1 / 12)
        assert overall["agreement_control_accuracy"] == 100
        assert overall["cross_control_accuracy"] == 50
    else:
        assert overall["tie_rate"] == 0.125
        assert overall["agreement_control_accuracy"] is None
        assert overall["cross_control_accuracy"] is None
    report_entries = list(
        native_entries({"task": task, "native": native_summaries(rows)})
    )
    assert len({(s, m) for s, m, _, _ in report_entries}) == len(report_entries)
    assert "not original coreference-system F1" in metric_catalog(task)["protocol_note"]


def test_wino_missing_pair_rejected_instead_of_misleading_native_score():
    row = wino("winobias_uk_controlled", "a", "type1", "pro", 1)
    with pytest.raises(ValueError, match="complete pro/anti pairs"):
        native_summaries([row])


def test_warbias_condition_metrics_include_invalids_and_evidence_gap():
    rows = []
    for i, (condition, response, gold) in enumerate(
        [
            ("ambiguous", "A", 2),
            ("ambiguous", "C", 2),
            ("ambiguous", "refusal", 2),
            ("ambiguous", "B", 2),
            ("stereotype_aligned", "A", 0),
            ("stereotype_conflicting", "A", 1),
        ]
    ):
        row = adapt(
            "warbias_uk",
            {
                "id": str(i),
                "source_case_id": str(i),
                "scenario_id": str(i),
                "target_group": "idp",
                "evidence_condition": condition,
                "question_polarity": "negative",
                "answer_index": gold,
                "stereotype_consistent_index": 0,
                "unknown_index": 2,
                "choices": ["ВПО", "Місцевий", "Невідомо"],
                "context": "Опис",
                "question": "Хто?",
            },
        )
        rows.append(score_item(row, {"response": response}))
    native = native_summaries(rows)
    ambiguous = next(r for r in native if r.get("condition") == "ambiguous")
    for metric in (
        "accuracy",
        "stereotype_rate",
        "counter_rate",
        "unknown_rate",
        "invalid",
    ):
        assert ambiguous[metric] == 0.25
    gap = next(r for r in native if r.get("scope") == "within_group_evidence_gap")
    assert gap["aligned_minus_conflicting_accuracy"] == 1
    report = make_report(rows, rows, bootstrap=0)
    assert report["metric_catalog"]["version"] == VERSION
    report["provenance"] = {}
    rendered = markdown(report)
    assert rendered.index("Dataset-specific metrics") < rendered.index(
        "Additional subgroup"
    )


def test_catalog_is_serializable_complete_and_defensive():
    catalog = metric_catalog()
    json.dumps(catalog, allow_nan=False)
    assert len(catalog["benchmarks"]) == 7
    assert {"warbias_triplets", "warbias_benign"} <= catalog["benchmarks"].keys()
    for benchmark in catalog["benchmarks"].values():
        ids = [m["id"] for m in benchmark["metrics"]]
        assert len(set(ids)) == len(ids)
        for metric in benchmark["metrics"]:
            assert all(
                metric[k]
                for k in (
                    "id",
                    "definition",
                    "denominator",
                    "direction",
                    "unit",
                    "origin",
                )
            )
            assert metric["origin"] in ("original", "adaptation", "diagnostic")
        assert benchmark["additional_group_metrics"]
    changed = deepcopy(catalog)
    changed["benchmarks"]["warbias"]["metrics"][0]["unit"] = "broken"
    assert metric_catalog("warbias_en")["metrics"][0]["unit"] == "rate"
    with pytest.raises(ValueError, match="No metric specification"):
        metric_catalog("unknown")
    with pytest.raises(KeyError):
        metric_spec("stereoset_uk", "accuracy")


def test_report_units_distinguish_scores_rates_gaps_and_counts():
    native = [
        {"scope": "example", "n": 3, "ss": 50, "icat": 75, "stereotype_tie_rate": 0.25}
    ]
    entries = {
        metric: unit
        for _, metric, _, unit in native_entries(
            {"task": "stereoset_uk", "native": native}
        )
    }
    assert entries == {
        "n": "count",
        "ss": "percent",
        "icat": "score_points",
        "stereotype_tie_rate": "rate",
    }
