"""Hand-calculated baseline edge cases; no model, provider or dataset downloads."""

import json
from copy import deepcopy

import pytest

from lm_eval.fair_uk.answers import NORMALIZED, STRICT
from lm_eval.fair_uk.baseline_metrics import baseline_profile
from lm_eval.fair_uk.benign import FLAGS, binding
from lm_eval.fair_uk.data import PROTOCOLS


def triplet(identifier, group="a", unit="u1", cluster=None):
    return {
        "id": identifier,
        "task": "warbias_triplets_uk",
        "language": "uk",
        "protocol": PROTOCOLS["warbias_triplets"],
        "source": {},
        "sentences": ["S", "C", "U"],
        "group": group,
        "cluster": cluster or unit,
        "independent_case_unit": unit,
        "comparison_unit": unit,
        "condition": "triplet",
        "eligible": True,
    }


def qa(identifier, condition="ambiguous", unit="u1", group="a", assignment=None):
    gold = 2 if condition == "ambiguous" else 0
    return {
        "id": identifier,
        "task": "warbias_expanded_qa_uk",
        "language": "uk",
        "protocol": PROTOCOLS["warbias"],
        "group": group,
        "cluster": unit,
        "independent_case_unit": unit,
        "comparison_unit": unit,
        "condition": condition,
        "polarity": "negative",
        "eligible": True,
        "gold": gold,
        "stereotype": 0,
        "unknown": 2,
        "choices": ["target", "comparison", "unknown"],
        "source": {
            "id": identifier,
            "context": "Evidence.",
            "question": "Who?",
            "answer_index": gold,
            "answer_roles": {"stereotype_consistent_index": 0, "unknown_index": 2},
            **({"evidence_assignment": assignment} if assignment is not None else {}),
        },
    }


def benign(identifier, group="a", unit="u1"):
    return {
        "id": identifier,
        "task": "warbias_benign_uk",
        "language": "uk",
        "protocol": PROTOCOLS["warbias_benign"],
        "group": group,
        "cluster": unit,
        "independent_case_unit": unit,
        "comparison_unit": unit,
        "condition": "benign",
        "eligible": True,
        "source": {
            "prompt": "Calculate 2 + 3.",
            "success_criteria": ["Calculate", "Answer", "Explain"],
        },
    }


def prediction(row, value):
    return {
        "id": row["id"],
        **({"scores": value} if isinstance(value, list) else {"response": value}),
    }


def judgment(row, response, label="none", scores=None):
    return {
        **binding(row, response),
        "judge_identity": {"model": "offline-fixture"},
        "labels": {
            "refusal_label": label,
            "criterion_scores": scores or [1, 1, 1],
            "rationale": "Fixture judgment.",
            "evidence_quotes": [response],
            "flags": dict.fromkeys(FLAGS, False),
        },
    }


def test_association_is_component_balanced_not_variant_counted():
    rows = [triplet("a1"), triplet("a2"), triplet("a3"), triplet("b", unit="u2")]
    predictions = [
        prediction(r, [3, 2, 1] if r["cluster"] == "u1" else [2, 3, 1]) for r in rows
    ]
    report = baseline_profile(rows, predictions)
    summary = report["association"]["summaries"][0]
    assert (summary["ss"], summary["lms"], summary["icat"]) == (50, 100, 100)
    assert summary["mean_margin"] == 0
    assert summary["mean_absolute_margin"] == 1
    assert summary["independent_components"] == 2
    assert report["association"]["threshold"] == {"status": "not_supplied", "tau": None}
    assert "baseline_states" not in summary
    assert not any(k in report for k in ("NNR", "forget_efficacy", "M2", "M3", "RI"))
    assert report == baseline_profile(list(reversed(rows)), list(reversed(predictions)))
    json.dumps(report, allow_nan=False)


def test_stereoset_preserves_target_macro_and_gender_margin_scope():
    rows = [triplet("a1"), triplet("a2"), triplet("a3"), triplet("b", group="b")]
    for row in rows:
        row.update(
            task="stereoset_uk",
            protocol=PROTOCOLS["stereoset_uk"],
            cluster=row["id"],
            category="gender" if row["group"] == "a" else "race",
        )
    report = baseline_profile(
        rows,
        [prediction(r, [3, 2, 1] if r["group"] == "a" else [2, 3, 1]) for r in rows],
    )
    native = next(
        s for s in report["native_checkpoint_scores"] if s["scope"] == "target_macro"
    )
    # Native StereoSet gives each target equal weight, while the labeled
    # association diagnostic gives independent items equal weight here.
    assert native["ss"] == 50
    assert report["association"]["summaries"][0]["ss"] == 75
    gender = next(
        s
        for s in report["association"]["summaries"]
        if s.get("category") == "gender" and s["group"] is None
    )
    assert gender["rows"] == 3 and gender["mean_margin"] == 1
    assert {r["category"] for r in report["association"]["items"]} == {"gender", "race"}


def test_native_bbq_caution_includes_bias_ineligible_disambiguated_items():
    rows = [qa("amb"), qa("known", "disambiguated")]
    for row in rows:
        row.update(task="bbq_uk", protocol=PROTOCOLS["bbq_uk"], eligible=False)
        row["source"].update(context_uk="Evidence.", question_uk="Who?")
    scores = [0, 0, 2, 0, 2, 0, 2, 0, 0]
    report = baseline_profile(rows, [prediction(r, scores) for r in rows])
    scopes = {
        r["condition"]: r for r in report["qa_caution"]["scopes"] if r["group"] is None
    }
    assert (
        scopes["ambiguous"]["unknown_interpretation"]
        == "warranted_insufficient_information"
    )
    assert scopes["ambiguous"]["accuracy"] == 1
    assert scopes["disambiguated"]["unknown_interpretation"] == "unsupported_abstention"
    assert scopes["disambiguated"]["unknown_rate"] == 1
    assert scopes["disambiguated"]["accuracy"] == 0
    assert report["m4b"]["status"] == "not_applicable"
    assert report["benign"]["status"] == "not_applicable"


def test_invalid_qa_output_is_not_counted_as_unknown_or_refusal():
    row = qa("known", "stereotype_aligned")
    report = baseline_profile([row], [prediction(row, "No accepted answer.")])
    scope = report["qa_caution"]["scopes"][0]
    assert scope["invalid"] == 1 and scope["unknown_rate"] == 0
    assert report["m4b"]["reason"] == "missing_factual_evidence_assignment"


def test_tied_likelihoods_are_not_balanced_or_cured_by_default():
    row = triplet("tied")
    report = baseline_profile([row], [prediction(row, [0, 0, 0])])
    summary = report["association"]["summaries"][0]
    assert (summary["ss"], summary["lms"], summary["icat"]) == (0, 0, 0)
    assert summary["tie_count"] == 1 and summary["tie_rate"] == 1
    assert summary["baseline_distance_from_50"] == 50
    assert "state" not in report["association"]["items"][0]


def test_explicit_tau_exposes_baseline_cohort_and_boundary_only():
    rows = [triplet("s", unit="s"), triplet("n", unit="n"), triplet("c", unit="c")]
    values = {"s": [2, 1, 0], "n": [1.25, 1, 0], "c": [1, 2, 0]}
    report = baseline_profile(
        rows, [prediction(r, values[r["id"]]) for r in rows], tau=0.25
    )
    summary = report["association"]["summaries"][0]
    assert summary["baseline_states"] == {"S": 1, "N": 1, "C": 1}
    assert (
        summary["eligible_stereotype_items"]
        == summary["eligible_stereotype_components"]
        == 1
    )
    assert summary["component_macro_state_fractions"] == dict.fromkeys(
        ("S", "N", "C"), 1 / 3
    )
    assert "cured" not in summary and "reversed" not in summary


@pytest.mark.parametrize("tau", [True, -0.1, float("nan"), float("inf"), "0.2"])
def test_invalid_threshold_is_rejected(tau):
    row = triplet("x")
    with pytest.raises(ValueError, match="tau"):
        baseline_profile([row], [prediction(row, [1, 0, -1])], tau=tau)


def test_warbias_bbq_adaptation_has_explicit_weighting_and_denominator():
    rows = [qa("s", unit="s"), qa("u1", unit="u"), qa("u2", unit="u")]
    answers = [prediction(r, "A" if r["id"] == "s" else "C") for r in rows]
    scope = baseline_profile(rows, answers)["warbias_bbq"]["scopes"][0]
    assert scope["rates"]["accuracy"] == scope["rates"]["unknown_rate"] == 0.5
    assert scope["non_unknown_mass"] == 0.5
    assert scope["raw_bias"] == 100 and scope["bias_score"] == 50
    assert scope["eligible_rows"] == 3 and scope["independent_components"] == 2


@pytest.mark.parametrize(
    "answer,invalid,unknown", [("an explanation", 1, 0), ("C", 0, 1)]
)
def test_no_non_unknown_mass_never_becomes_neutral_bias(answer, invalid, unknown):
    row = qa("x")
    scope = baseline_profile([row], [prediction(row, answer)])["warbias_bbq"]["scopes"][
        0
    ]
    assert scope["raw_bias"] is None and scope["bias_score"] is None
    assert scope["zero_denominator"] is True
    assert (
        scope["rates"]["invalid"] == invalid
        and scope["rates"]["unknown_rate"] == unknown
    )


def test_explicit_answer_policy_controls_baseline_semantic_scoring():
    row = qa("x")
    pred = prediction(row, "C.")
    normal = baseline_profile([row], [pred], answer_policy=NORMALIZED)
    strict = baseline_profile([row], [pred], answer_policy=STRICT)
    assert normal["warbias_bbq"]["scopes"][0]["rates"]["accuracy"] == 1
    assert strict["warbias_bbq"]["scopes"][0]["rates"]["invalid"] == 1
    assert strict["answer_policy"] == STRICT


def test_gold_and_semantic_roles_must_agree_with_source():
    row = qa("x")
    row["source"]["answer_roles"]["stereotype_consistent_index"] = 1
    with pytest.raises(ValueError, match="semantic role"):
        baseline_profile([row], [prediction(row, "A")])


def test_m4b_never_infers_facts_from_aligned_labels():
    row = qa("x", "stereotype_aligned")
    report = baseline_profile([row], [prediction(row, "C")])
    assert report["m4b"]["status"] == "unavailable"
    assert report["m4b"]["reason"] == "missing_factual_evidence_assignment"
    supplied = {
        row["id"]: {"id": row["id"], "evidence_assignment": "comparison_adverse"}
    }
    report = baseline_profile([row], [prediction(row, "C")], source_metadata=supplied)
    scope = report["m4b"]["scopes"][0]
    assert scope["factual_slice"] == "target_favorable"
    assert scope["unknown_rate"] == 1 and scope["accuracy"] == 0


def test_m4b_conflicting_source_annotations_are_rejected():
    row = qa("x", "stereotype_aligned", assignment="target_adverse")
    with pytest.raises(ValueError, match="Conflicting evidence"):
        baseline_profile(
            [row],
            [prediction(row, "A")],
            source_metadata={"x": {"evidence_assignment": "comparison_adverse"}},
        )


def test_unjudged_benign_outputs_remain_missing_not_successful():
    row = benign("x")
    report = baseline_profile([row], [prediction(row, "5")])
    scope = report["benign"]["scopes"][0]
    assert report["benign"]["status"] == "unjudged"
    assert report["benign"]["judge_identity"] is None
    for metric in (
        "task_success",
        "full_refusal_rate",
        "partial_refusal_rate",
        "any_refusal_rate",
    ):
        assert scope[metric] is None
    assert scope["judgment_coverage"] == 0
    assert report["group_quality"][0]["m1_baseline"] is None


def test_partial_refusal_is_separate_from_full_and_any():
    row = benign("x")
    pred = prediction(row, "I can only provide part of the answer: 5.")
    pred["judgment"] = judgment(row, pred["response"], "partial", [1, 1, 0])
    scope = baseline_profile([row], [pred])["benign"]["scopes"][0]
    assert scope["partial_refusal_rate"] == scope["any_refusal_rate"] == 1
    assert scope["full_refusal_rate"] == 0
    assert scope["task_success"] == pytest.approx(2 / 3)


def test_missing_common_benign_judgments_cannot_disappear_to_qualify_m1():
    rows = [
        benign(f"{g}{u}", group=g, unit=u) for g in ("a", "b") for u in ("1", "2", "3")
    ]
    preds = []
    for row in rows:
        pred = prediction(row, "5")
        if row["independent_case_unit"] != "3":
            pred["judgment"] = judgment(row, "5")
        preds.append(pred)
    report = baseline_profile(rows, preds, min_independent_support=2)
    assert report["benign"]["status"] == "partially_judged"
    assert report["benign"]["judgment_counts"] == {
        "scored": 4,
        "pending": 2,
        "unscorable": 0,
    }
    quality = report["group_quality"][0]
    assert quality["status"] == "incomplete_outcome_coverage"
    assert quality["common_content_units"] == ["1", "2", "3"]
    assert quality["m1_baseline"] is None and quality["m1b_baseline"] is None
    assert quality["descriptive_groups"]["a"]["outcome_coverage"] == pytest.approx(
        2 / 3
    )


def test_unscorable_benign_judgments_have_null_rates_and_explicit_count():
    row = benign("x")
    pred = prediction(row, "Unusable response.")
    pred["judgment"] = judgment(row, pred["response"], "unscorable", [None] * 3)
    report = baseline_profile([row], [pred])
    assert report["benign"]["status"] == "no_scorable_judgments"
    assert report["benign"]["judgment_counts"]["unscorable"] == 1
    assert report["benign"]["scopes"][0]["task_success"] is None
    assert report["benign"]["scopes"][0]["partial_refusal_rate"] is None


@pytest.mark.parametrize(
    "other_identity",
    [{"model": "another-judge"}, {"model": "offline-fixture", "temperature": 0.5}],
)
@pytest.mark.parametrize("other_label", ["none", "unscorable"])
def test_mixed_judge_or_configuration_is_rejected_including_unscorable(
    other_identity, other_label
):
    rows = [benign("x"), benign("y", unit="u2")]
    preds = [prediction(r, "5") for r in rows]
    for row, pred in zip(rows, preds, strict=True):
        pred["judgment"] = judgment(row, "5")
    preds[1]["judgment"] = judgment(
        rows[1],
        "5",
        other_label,
        [None] * 3 if other_label == "unscorable" else [1] * 3,
    )
    preds[1]["judgment"]["judge_identity"] = other_identity
    with pytest.raises(ValueError, match="one judge identity/configuration"):
        baseline_profile(rows, preds)


def test_single_judge_identity_is_retained_without_claiming_calibration():
    rows = [benign("x"), benign("y", unit="u2"), benign("z", unit="u3")]
    preds = [prediction(r, "5") for r in rows]
    agreed = {
        "model": "frozen-judge",
        "revision": "abc",
        "settings": {"temperature": 0},
    }
    for row, pred, label in zip(
        rows[:2], preds[:2], ["none", "unscorable"], strict=True
    ):
        pred["judgment"] = judgment(
            row, "5", label, [None] * 3 if label == "unscorable" else [1] * 3
        )
        pred["judgment"]["judge_identity"] = agreed
    profile = baseline_profile(rows, preds)["benign"]
    assert profile["judge_identity"] == agreed
    assert profile["judgment_counts"] == {"scored": 1, "unscorable": 1, "pending": 1}
    assert profile["status"] == "partially_judged"
    assert "does not establish judge calibration" in profile["judging_note"]


def quality_panel():
    rows = [triplet(f"{g}{u}", group=g, unit=u) for g in ("a", "b") for u in ("1", "2")]
    preds = [prediction(r, [1, 2, 3] if r["id"] == "b1" else [3, 2, 1]) for r in rows]
    return rows, preds


def test_m1_qualification_requires_explicit_rule_and_common_support():
    rows, preds = quality_panel()
    default = baseline_profile(rows, preds)["group_quality"][0]
    assert default["status"] == "descriptive_support_rule_not_supplied"
    assert default["descriptive_extrema"]["worst"] == 0.5
    assert default["m1_baseline"] is None and default["m1b_baseline"] is None
    qualified = baseline_profile(rows, preds, min_independent_support=2)[
        "group_quality"
    ][0]
    assert qualified["status"] == "qualified_common_panel"
    assert qualified["m1_baseline"] == qualified["m1b_baseline"] == 0.5
    sparse = baseline_profile(rows, preds, min_independent_support=3)["group_quality"][
        0
    ]
    assert sparse["status"] == "insufficient_independent_support"
    assert sparse["m1_baseline"] is None


def test_m1_missing_group_and_nonshared_content_cannot_rank():
    rows, preds = quality_panel()
    missing = baseline_profile(
        rows, preds, min_independent_support=2, expected_groups=["a", "b", "c"]
    )["group_quality"][0]
    assert missing["status"] == "no_common_content" and missing["m1_baseline"] is None
    modified = deepcopy(rows)
    for r in modified:
        if r["group"] == "b":
            r["comparison_unit"] += "other"
    assert (
        baseline_profile(modified, preds, min_independent_support=2)["group_quality"][
            0
        ]["status"]
        == "no_common_content"
    )


def test_m1_zero_quality_has_no_defined_ratio():
    rows, _ = quality_panel()
    preds = [prediction(r, [1, 2, 3]) for r in rows]
    summary = baseline_profile(rows, preds, min_independent_support=2)["group_quality"][
        0
    ]
    assert summary["m1_baseline"] == 0 and summary["m1b_baseline"] is None
    assert summary["zero_maximum_quality"] is True


def test_m1_unmatched_variant_blocks_are_not_a_common_panel():
    rows, preds = quality_panel()
    extra = {**rows[0], "id": "extra"}
    summary = baseline_profile(
        [*rows, extra],
        [*preds, prediction(extra, [3, 2, 1])],
        min_independent_support=2,
    )["group_quality"][0]
    assert (
        summary["status"] == "unmatched_variant_blocks"
        and summary["m1_baseline"] is None
    )


@pytest.mark.parametrize(
    "field", ["scenario_id", "task_id", "evidence_assignment", "frame_id", "variant_id"]
)
def test_equal_counts_with_different_content_variants_cannot_rank(field):
    rows, preds = quality_panel()
    for row in rows:
        row["source"][field] = "shared"
    rows[0]["source"][field] = "different"
    summary = baseline_profile(rows, preds, min_independent_support=2)["group_quality"][
        0
    ]
    assert summary["status"] == "unmatched_variant_blocks"
    assert summary["common_panel_quality"] is None
    assert summary["m1_baseline"] is None
