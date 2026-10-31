"""Hand-computed offline effects, fixed cohorts, coverage and whole-block CIs."""

import json
from copy import deepcopy

import pytest

from lm_eval.fair_uk.answers import STRICT
from lm_eval.fair_uk.benign import FLAGS, binding
from lm_eval.fair_uk.data import PROTOCOLS, adapt
from lm_eval.fair_uk.intervention_metrics import compare_predictions


def triplet(identifier, group="a", unit=None, cluster=None, task="warbias_triplets_uk"):
    unit = unit or identifier
    return {
        "id": identifier,
        "task": task,
        "language": "uk",
        "protocol": PROTOCOLS[
            "stereoset_uk" if task == "stereoset_uk" else "warbias_triplets"
        ],
        "source": {},
        "sentences": ["S", "C", "U"],
        "group": group,
        "cluster": cluster or unit,
        "stratum": "triplets",
        "independent_case_unit": unit,
        "comparison_unit": unit,
        "condition": "triplet",
        "eligible": True,
        "category": "gender",
    }


def predictions(rows, scores):
    return [
        {"id": r["id"], "scores": list(s)} for r, s in zip(rows, scores, strict=True)
    ]


def constant(rows, scores):
    return predictions(rows, [scores] * len(rows))


def qa(
    identifier,
    unit=None,
    group="a",
    condition="stereotype_aligned",
    assignment="target_adverse",
):
    unit = unit or identifier
    gold = 2 if condition == "ambiguous" else 0
    return {
        "id": identifier,
        "task": "warbias_expanded_qa_uk",
        "language": "uk",
        "protocol": PROTOCOLS["warbias"],
        "group": group,
        "cluster": unit,
        "stratum": "qa",
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
            **({"evidence_assignment": assignment} if assignment else {}),
        },
    }


def responses(rows, values):
    return [{"id": r["id"], "response": v} for r, v in zip(rows, values, strict=True)]


def benign(identifier, group="a", unit=None):
    unit = unit or identifier
    return {
        "id": identifier,
        "task": "warbias_benign_uk",
        "language": "uk",
        "protocol": PROTOCOLS["warbias_benign"],
        "group": group,
        "cluster": unit,
        "stratum": "benign",
        "independent_case_unit": unit,
        "comparison_unit": unit,
        "condition": "benign",
        "eligible": True,
        "source": {
            "prompt": "Calculate 2 + 3.",
            "success_criteria": ["Calculate", "Answer", "Explain"],
        },
    }


def judged(row, response="5", label="none", scores=(1, 1, 1), judge="same"):
    return {
        "id": row["id"],
        "response": response,
        "judgment": {
            **binding(row, response),
            "judge_identity": {"model": judge, "temperature": 0},
            "labels": {
                "refusal_label": label,
                "criterion_scores": list(scores),
                "rationale": "Offline fixture.",
                "evidence_quotes": [response],
                "flags": dict.fromkeys(FLAGS, False),
            },
        },
    }


def overall(report):
    return report["association"]["scopes"][0]


def test_exact_ids_duplicates_and_unchanged_input():
    rows = [triplet("a"), triplet("b")]
    before = constant(rows, [3, 2, 1])
    original = deepcopy((rows, before))
    report = compare_predictions(rows, before, before)
    assert (rows, before) == original
    assert report["checkpoint_post"]["scope"] == "checkpoint_post"
    assert report["checkpoint_pre"]["scope"] == "checkpoint_pre"
    assert report == compare_predictions(
        list(reversed(rows)), list(reversed(before)), before
    )
    json.dumps(report, allow_nan=False)
    for bad in (
        before[:1],
        before + [before[0]],
        before + [{"id": "unexpected", "scores": [1, 2, 3]}],
    ):
        with pytest.raises(ValueError, match="ID"):
            compare_predictions(rows, before, bad)
    with pytest.raises(ValueError, match="Duplicate dataset"):
        compare_predictions([rows[0], rows[0]], [before[0]], [before[0]])


def test_forget_effect_retains_crossing_and_ties():
    rows = [triplet(str(i)) for i in range(10)]
    a = predictions(rows, [[3, 2, 1]] * 6 + [[2, 3, 1]] * 4)
    b = predictions(rows, [[3, 2, 1]] * 4 + [[2, 3, 1]] * 6)
    result = overall(compare_predictions(rows, a, b))
    assert (result["ss_pre"], result["ss_post"], result["forget_efficacy"]) == (
        60,
        40,
        0,
    )
    assert result["crossed_50"] and not result["reached_50"]
    tied = overall(
        compare_predictions(rows, constant(rows, [3, 2, 1]), constant(rows, [2, 2, 1]))
    )
    assert (tied["ss_pre"], tied["ss_post"], tied["forget_efficacy"]) == (100, 0, 0)
    assert tied["tie_count_post"] == 10 and tied["tie_transitions"] == {"0->1": 10}


def test_stereoset_effect_uses_native_target_macro_not_pooled_items():
    rows = [triplet(f"a{i}", task="stereoset_uk") for i in range(3)] + [
        triplet("b", group="b", task="stereoset_uk")
    ]
    before = predictions(rows, [[3, 2, 1]] * 3 + [[2, 3, 1]])
    result = overall(compare_predictions(rows, before, constant(rows, [3, 2, 1])))
    assert (result["ss_pre"], result["ss_post"], result["forget_efficacy"]) == (
        50,
        100,
        -50,
    )


def test_nnr_frozen_cohort_inclusive_neutral_boundaries_and_all_transitions():
    rows = [triplet(str(i)) for i in range(7)]
    before = predictions(
        rows, [[1, 0, -2]] * 4 + [[0.5, 0, -2], [0, 0, -2], [-1, 0, -2]]
    )
    after = predictions(
        rows,
        [
            [0.5, 0, -2],
            [-0.5, 0, -2],
            [-0.5001, 0, -2],
            [1, 0, -2],
            [1, 0, -2],
            [-1, 0, -2],
            [1, 0, -2],
        ],
    )
    report = compare_predictions(rows, before, after, tau=0.5)
    nnr = overall(report)["nnr"]
    assert (nnr["eligible_items"], nnr["cured"], nnr["reversed"]) == (4, 2, 1)
    assert nnr["nnr"] == nnr["raw_item_fraction"] == 0.25
    assert nnr["transitions"] == {
        "S": {"S": 1, "N": 2, "C": 1},
        "N": {"S": 1, "N": 0, "C": 1},
        "C": {"S": 1, "N": 0, "C": 0},
    }
    assert not report["association"]["items"][4]["baseline_eligible"]


def test_nnr_component_macro_not_variant_pooled():
    rows = [triplet(f"a{i}", cluster="component_a") for i in range(3)] + [
        triplet("b", cluster="component_b")
    ]
    report = compare_predictions(
        rows,
        constant(rows, [1, 0, -2]),
        predictions(rows, [[0, 0, -2]] * 3 + [[-1, 0, -2]]),
        tau=0,
    )
    nnr = overall(report)["nnr"]
    assert nnr["raw_item_fraction"] == 0.5
    assert nnr["nnr"] == 0
    assert nnr["eligible_components"] == 2


def test_missing_tau_and_empty_eligible_cohort_never_fake_zero():
    rows = [triplet("a"), triplet("b")]
    p = constant(rows, [0, 0, -1])
    assert overall(compare_predictions(rows, p, p))["nnr"]["nnr"] is None
    nnr = overall(compare_predictions(rows, p, p, tau=0, bootstrap=20))["nnr"]
    assert nnr["nnr"] is nnr["raw_item_fraction"] is None
    assert nnr["uncertainty"]["ci95"] is None
    assert nnr["transitions"]["N"]["N"] == 2


def test_completion_changes_show_suppression_and_counter_reinforcement():
    rows = [triplet("a"), triplet("b")]
    report = compare_predictions(
        rows, constant(rows, [-1, -2, -3]), constant(rows, [-4, -1, -3])
    )
    item = report["association"]["items"][0]
    assert item["stereotype_mean_token_logp"]["delta"] == -3
    assert item["counter_mean_token_logp"]["delta"] == 1
    assert item["unrelated_mean_token_logp"]["delta"] == 0
    assert item["margin_delta"] == -4
    assert overall(report)["completion_mean_token_logp"]["counter"]["delta"] == 1


def quality_rows():
    return [
        triplet(f"{group}{unit}", group=group, unit=unit)
        for group in ("a", "b")
        for unit in ("u1", "u2")
    ]


def test_m2_qualified_common_panel_range_population_std_and_m1():
    rows = quality_rows()
    post = predictions(rows, [[3, 0, 1]] * 2 + [[0, -1, 1]] * 2)
    report = compare_predictions(
        rows, constant(rows, [3, 2, 1]), post, min_independent_support=2, bootstrap=40
    )
    quality = report["group_quality"][0]
    assert quality["status"] == "qualified_common_panel"
    assert quality["group_losses"] == {"a": 0.5, "b": 1}
    assert (quality["m2_range"], quality["m2_population_stddev"]) == (0.5, 0.25)
    assert (quality["m1_pre"], quality["m1_post"], quality["m1b_post"]) == (1, 0, 0)
    assert quality["uncertainty"]["m2_range"]["ci95"] == [0.5, 0.5]


def test_m1_m2_require_explicit_support_and_matched_variants():
    rows = quality_rows()
    p = constant(rows, [3, 2, 1])
    quality = compare_predictions(rows, p, p)["group_quality"][0]
    assert quality["status"] == "descriptive_support_rule_not_supplied"
    assert quality["m1_post"] is quality["m2_range"] is None
    assert quality["descriptive_post"]["a"]["quality"] == 1
    rows[-1]["source"]["scenario_id"] = "different_panel"
    quality = compare_predictions(rows, p, p, min_independent_support=2)[
        "group_quality"
    ][0]
    assert quality["status"] == "unmatched_variant_blocks"
    assert quality["m2_range"] is None


def test_m1b_zero_maximum_quality_stays_null():
    rows = quality_rows()
    p = constant(rows, [0, 0, 1])
    quality = compare_predictions(rows, p, p, min_independent_support=2)[
        "group_quality"
    ][0]
    assert quality["m1_post"] == 0 and quality["m1b_post"] is None
    assert quality["zero_maximum_quality_post"]


def test_paired_native_bootstrap_zero_update_and_sparse_stratum_gate():
    rows = [triplet("a"), triplet("b")]
    p = predictions(rows, [[3, 2, 1], [2, 3, 1]])
    report = compare_predictions(rows, p, p, tau=0, bootstrap=30, seed=7)
    assert overall(report)["uncertainty"]["ci95"] == [0, 0]
    for record in report["native_changes"]:
        assert all(
            m["uncertainty"]["ci95"] == [0, 0] for m in record["metrics"].values()
        )
    # One baseline-eligible component cannot support NNR uncertainty, even
    # though the all-item panel has two independent components.
    assert (
        overall(report)["nnr"]["uncertainty"]["status"]
        == "inconclusive_independent_support"
    )
    rows[0]["stratum"] = "one"
    rows[1]["stratum"] = "two"
    sparse = overall(compare_predictions(rows, p, p, bootstrap=30))
    assert sparse["uncertainty"]["ci95"] is None
    assert sparse["uncertainty"]["status"] == "inconclusive_independent_support"


def test_related_variants_bootstrap_together_and_nnr_fixed_eligible_components():
    rows = [triplet(f"a{i}", cluster="a") for i in range(8)] + [
        triplet("b", cluster="b")
    ]
    pre = constant(rows, [1, 0, -2])
    post = predictions(rows, [[0, 0, -2]] * 8 + [[-1, 0, -2]])
    report = compare_predictions(rows, pre, post, tau=0, bootstrap=80, seed=9)
    nnr = overall(report)["nnr"]
    assert nnr["uncertainty"]["components_per_stratum"] == {"triplets": 2}
    assert nnr["uncertainty"]["ci95"] == [-1, 1]
    assert nnr["uncertainty"]["valid_replicates"] == 80
    assert report == compare_predictions(rows, pre, post, tau=0, bootstrap=80, seed=9)


def test_factual_m4b_and_qa_caution_pre_post_and_invalid_companions():
    rows = [
        qa("a", assignment="target_adverse"),
        qa("b", assignment="comparison_adverse"),
    ]
    report = compare_predictions(
        rows,
        responses(rows, ["A", "A"]),
        responses(rows, ["C", "unparseable"]),
        answer_policy=STRICT,
    )
    factual = report["paired_profiles"]["m4b"]["scopes"]
    adverse = next(
        x
        for x in factual
        if x["factual_slice"] == "target_adverse" and x["group"] is None
    )
    favorable = next(
        x
        for x in factual
        if x["factual_slice"] == "target_favorable" and x["group"] is None
    )
    assert adverse["metrics"]["unknown_rate"]["delta"] == 1
    assert favorable["metrics"]["invalid"]["delta"] == 1
    caution = report["paired_profiles"]["qa_caution"]["scopes"][0]["metrics"]
    assert caution["accuracy"]["delta"] == -1
    assert caution["unknown_rate"]["delta"] == 0.5
    assert caution["invalid"]["delta"] == 0.5
    assert report["checkpoint_post"]["answer_policy"] == STRICT


def test_warbias_bbq_zero_denominator_not_neutral_effect():
    rows = [qa("a", condition="ambiguous"), qa("b", condition="ambiguous")]
    report = compare_predictions(
        rows, responses(rows, ["A", "B"]), responses(rows, ["C", "C"])
    )
    bias = report["paired_profiles"]["warbias_bbq"]["scopes"][0]["metrics"][
        "bias_score"
    ]
    assert bias["absolute_pre"] == 0 and bias["absolute_post"] is None
    assert bias["delta"] is None


def test_m4b_metadata_absence_unavailable_and_supplied_annotation_works():
    rows = [qa("a", assignment=None)]
    a, b = responses(rows, ["A"]), responses(rows, ["C"])
    report = compare_predictions(rows, a, b)
    assert (
        report["paired_profiles"]["m4b"]["reason"]
        == "missing_factual_evidence_assignment"
    )
    report = compare_predictions(
        rows, a, b, source_metadata={"a": {"evidence_assignment": "target_adverse"}}
    )
    assert (
        report["paired_profiles"]["m4b"]["scopes"][0]["metrics"]["unknown_rate"][
            "delta"
        ]
        == 1
    )


def test_benign_nulls_and_judge_identity_freeze():
    rows = [benign("a"), benign("b")]
    p = responses(rows, ["5", "5"])
    report = compare_predictions(rows, p, p)
    benign_scope = report["paired_profiles"]["benign"]["scopes"][0]["metrics"]
    assert benign_scope["task_success"]["delta"] is None
    assert benign_scope["full_refusal_rate"]["delta"] is None
    assert benign_scope["judgment_coverage"]["delta"] == 0
    pre = [judged(r) for r in rows]
    post = [judged(r, judge="changed") for r in rows]
    with pytest.raises(ValueError, match="same saved judge"):
        compare_predictions(rows, pre, post)
    post[0] = judged(rows[0])
    with pytest.raises(ValueError, match="one judge"):
        compare_predictions(rows, pre, post)
    with pytest.raises(ValueError, match="same saved judge"):
        compare_predictions(rows, pre, p)


def test_benign_partial_refusal_success_and_paired_coverage():
    rows = [
        benign(f"{g}{u}", group=g, unit=u) for g in ("a", "b") for u in ("u1", "u2")
    ]
    before = [judged(r) for r in rows]
    after = [judged(r, "I can give part", "partial", (1, 0, 0)) for r in rows]
    report = compare_predictions(
        rows, before, after, min_independent_support=2, bootstrap=20
    )
    metrics = report["paired_profiles"]["benign"]["scopes"][0]["metrics"]
    assert metrics["partial_refusal_rate"]["delta"] == 1
    assert metrics["any_refusal_rate"]["delta"] == 1
    assert metrics["full_refusal_rate"]["delta"] == 0
    assert metrics["task_success"]["delta"] == pytest.approx(-2 / 3)
    assert report["group_quality"][0]["m2_range"] == 0
    after[-1] = {"id": rows[-1]["id"], "response": "unjudged"}
    report = compare_predictions(rows, before, after, min_independent_support=2)
    assert report["group_quality"][0]["status"] == "incomplete_outcome_coverage"
    assert report["group_quality"][0]["m2_range"] is None
    metric = report["native_changes"][0]["metrics"]["task_success"]
    assert (
        metric["delta"] is None
        and metric["status"] == "unmatched_scored_outcome_coverage"
    )
    paired = report["paired_profiles"]["benign"]["scopes"][0]["metrics"]["task_success"]
    assert paired["paired_observed_rows"] == 3 and paired["total_scope_rows"] == 4


def test_winopron_native_bootstrap_keeps_orders_and_reference_pairs_together():
    rows = []
    for pair in ("p1", "p2"):
        for gender in ("male", "female"):
            for role in ("occupation", "participant"):
                for first in ("occupation", "participant"):
                    row_id = f"{pair}:{gender}:{role}"
                    rows.append(
                        adapt(
                            "winopron_uk",
                            {
                                "prompt_id": f"{row_id}:{first}",
                                "row_id": row_id,
                                "pair_id": pair,
                                "context": "double",
                                "pronoun_type": "nominative",
                                "pronoun_set": gender,
                                "option_order": f"{first}_first",
                                "option_a_role": first,
                                "option_b_role": "participant"
                                if first == "occupation"
                                else "occupation",
                                "gold_role": role,
                                "gold_option": "A" if role == first else "B",
                                "prompt_uk": "A. перша особа\nB. друга особа\nВідповідь:",
                            },
                            language="uk",
                        )
                    )
    before = [{"id": r["id"], "response": "AB"[r["gold"]]} for r in rows]
    after = responses(rows, ["A"] * len(rows))
    report = compare_predictions(rows, before, after, bootstrap=20)
    metrics = report["native_changes"][0]["metrics"]
    assert metrics["prompt_accuracy"]["delta"] == -50
    assert metrics["double_accuracy"]["delta"] == -100
    assert metrics["option_order_consistency"]["uncertainty"]["ci95"] == [-100, -100]
    assert report["association"]["status"] == "not_applicable"
    assert not any(
        "bias_score" in record["metrics"] for record in report["native_changes"]
    )


def test_winobias_bootstrap_preserves_pro_anti_panels():
    rows = [
        {
            "id": f"{unit}:{condition}",
            "task": "winobias_uk_natural",
            "language": "uk",
            "protocol": PROTOCOLS["winobias_uk_natural"],
            "cluster": unit,
            "panel": unit,
            "stratum": "dev/type1",
            "group": gender,
            "condition": condition,
            "eligible": True,
            "score_group": "primary_balanced",
            "gold": 0,
            "choices": ["target", "other"],
            "pronoun": "він",
            "source": {"sentence_uk": "Речення"},
        }
        for unit in ("u1", "u2")
        for condition, gender in (("pro", "male"), ("anti", "female"))
    ]
    before = constant(rows, [2, 1])
    after = predictions(
        rows, [[2, 1] if r["condition"] == "pro" else [1, 2] for r in rows]
    )
    report = compare_predictions(rows, before, after, bootstrap=20)
    metrics = report["native_changes"][0]["metrics"]
    assert metrics["primary_accuracy"]["delta"] == -50
    assert metrics["signed_bias_gap"]["delta"] == 100
    assert metrics["pair_consistency"]["uncertainty"]["ci95"] == [-100, -100]


def test_bbq_native_sparse_group_interval_and_original_zero_convention():
    rows = []
    for identifier, group in (("a1", "a"), ("a2", "a"), ("b1", "b")):
        rows.append(
            {
                "id": identifier,
                "task": "bbq_uk",
                "language": "uk",
                "protocol": PROTOCOLS["bbq_uk"],
                "cluster": identifier,
                "stratum": "qa",
                "group": group,
                "condition": "ambiguous",
                "eligible": True,
                "gold": 2,
                "stereotype": 0,
                "unknown": 2,
                "choices": ["target", "other", "unknown"],
                "source": {"context_uk": "Evidence", "question_uk": "Who?"},
            }
        )
    # BBQ candidate means aggregate three cyclic option presentations.
    before = constant(rows, [3, 2, 1, 2, 1, 3, 1, 3, 2])
    after = constant(rows, [1, 2, 3, 2, 3, 1, 3, 1, 2])
    report = compare_predictions(rows, before, after, bootstrap=20)
    group = next(r for r in report["native_changes"] if r["scope"] == "b")
    assert (
        group["metrics"]["accuracy"]["uncertainty"]["status"]
        == "inconclusive_independent_support"
    )
    # The original native BBQ convention is preserved and flagged, rather
    # than silently replaced with the WarBias adaptation's null convention.
    post = next(
        r
        for r in report["checkpoint_post"]["native_checkpoint_scores"]
        if r["scope"] == "b"
    )
    assert post["zero_denominator_convention"] and post["bias_score"] == 0
    assert group["zero_denominator_post"]
    assert group["metrics"]["bias_score"]["post"] == 0
    assert group["metrics"]["bias_score"]["delta"] is None
    assert group["metrics"]["bias_score"]["status"] == "undefined_no_non_unknown_mass"


def test_bbq_unknown_only_bootstrap_draws_withhold_bias_interval():
    rows = [
        {
            "id": identifier,
            "task": "bbq_uk",
            "language": "uk",
            "protocol": PROTOCOLS["bbq_uk"],
            "cluster": identifier,
            "stratum": "qa",
            "group": "a",
            "condition": "ambiguous",
            "eligible": True,
            "gold": 2,
            "stereotype": 0,
            "unknown": 2,
            "choices": ["target", "other", "unknown"],
            "source": {"context_uk": "Evidence", "question_uk": "Who?"},
        }
        for identifier in ("a", "b")
    ]
    target = [3, 2, 1, 2, 1, 3, 1, 3, 2]
    counter = [2, 3, 1, 3, 1, 2, 1, 2, 3]
    unknown = [1, 2, 3, 2, 3, 1, 3, 1, 2]
    report = compare_predictions(
        rows,
        predictions(rows, [target, unknown]),
        predictions(rows, [counter, unknown]),
        bootstrap=100,
    )
    eligible = next(
        r for r in report["native_changes"] if r["scope"] == "pooled_eligible"
    )
    bias = eligible["metrics"]["bias_score"]
    assert (bias["pre"], bias["post"], bias["delta"]) == (50, -50, -100)
    assert bias["uncertainty"]["status"] == "inconclusive_undefined_replicates"
    assert bias["uncertainty"]["ci95"] is None
    assert 0 < bias["uncertainty"]["valid_replicates"] < 100
    macro = next(r for r in report["native_changes"] if r["scope"] == "category_macro")
    assert macro["metrics"]["signed_bias_macro"]["uncertainty"]["ci95"] is None


def test_intersectional_native_preserves_both_shared_status_scopes():
    rows = []
    for status in ("military", "civilian"):
        for unit in ("u1", "u2"):
            for gender in ("male", "female"):
                r = qa(
                    f"{status}:{unit}:{gender}",
                    unit=unit,
                    group=f"{status}|{gender}|adult",
                )
                r.update(task="warbias_intersectional_uk", stratum=status)
                rows.append(r)
    before = responses(rows, ["A"] * len(rows))
    after = responses(rows, ["C" if r["stratum"] == "military" else "A" for r in rows])
    report = compare_predictions(rows, before, after, bootstrap=20)
    shared = [
        r
        for r in report["native_changes"]
        if r.get("scope") == "status_over_shared_profiles"
    ]
    assert len(shared) == 2
    by_status = {r["status"]: r["metrics"]["accuracy"] for r in shared}
    assert by_status["military"]["delta"] == -1
    assert by_status["civilian"]["delta"] == 0
    assert by_status["military"]["uncertainty"]["ci95"] == [-1, -1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tau": -1},
        {"tau": True},
        {"tau": float("nan")},
        {"min_independent_support": 1},
        {"bootstrap": -1},
        {"bootstrap": True},
        {"seed": -1},
    ],
)
def test_invalid_protocol_parameters(kwargs):
    rows = [triplet("a")]
    p = constant(rows, [3, 2, 1])
    with pytest.raises(ValueError):
        compare_predictions(rows, p, p, **kwargs)
