"""Pure scored fixtures for transfer effects; no models or datasets loaded."""

from copy import deepcopy

import pytest

from lm_eval.fair_uk.transfer_metrics import transfer_report


def association_pair(
    task="warbias_triplets_uk",
    language="uk",
    components=2,
    target_before=1,
    target_after=0.5,
    other_before=0.5,
    other_after=1,
):
    rows, before, after = [], [], []
    for component in range(components):
        for actor in ("soldier", "civilian"):
            for polarity in ("positive", "negative"):
                item_id = f"{language}/{component}/{actor}/{polarity}"
                row = {
                    "id": item_id,
                    "task": task,
                    "language": language,
                    "protocol": "triplet_fixture_v1",
                    "stratum": "expansion",
                    "cluster": f"component{component}",
                    "comparison_unit": f"case{component}",
                    "condition": "triplet",
                    "polarity": polarity,
                    "actor": actor,
                    "gender": "not_stated",
                    "age": "adult",
                    "group": actor,
                }
                rows.append(row)
                pre = target_before if actor == "soldier" else other_before
                post = target_after if actor == "soldier" else other_after
                before.append(
                    {"id": item_id, "task": task, "stereotype_preference": pre}
                )
                after.append(
                    {"id": item_id, "task": task, "stereotype_preference": post}
                )
    return {"task": task, "rows": rows, "before_scored": before, "after_scored": after}


def spec(**targets):
    return {"training_language": "uk", "target_values": {"actor": "soldier", **targets}}


def test_reduction_and_redistribution_are_separate_from_acceptance():
    pair = association_pair()
    saved = deepcopy(pair)
    report = transfer_report([pair], spec(), bootstrap=200, min_target_improvement=1)
    actor = report["M3a"][0]
    assert actor["native_weighting"] == "component_macro"
    assert actor["cells"]["soldier"]["delta"] == -50
    assert actor["cells"]["civilian"]["delta"] == 50
    assert actor["cells"]["civilian"]["harm_change"] == 50
    assert actor["redistribution"]["denominator"] == 50
    assert actor["redistribution"]["numerator"] == 50
    assert actor["redistribution"]["ratio"] == 1
    assert actor["redistribution"]["denominator_ci95"] == [50, 50]
    assert actor["redistribution"]["ratio_ci95"] == [1, 1]
    assert report["acceptance_decision"] is None
    assert pair == saved


def test_no_update_has_zero_effect_and_null_ratio():
    pair = association_pair(target_after=1, other_after=0.5)
    actor = transfer_report([pair], spec(), bootstrap=100, min_target_improvement=1)[
        "M3a"
    ][0]
    assert all(cell["delta"] == 0 for cell in actor["cells"].values())
    assert actor["redistribution"]["numerator"] == 0
    assert actor["redistribution"]["denominator"] == 0
    assert actor["redistribution"]["ratio"] is None


def test_reversed_denominator_preserves_non_target_harm():
    pair = association_pair(target_before=0.5, target_after=1)
    actor = transfer_report([pair], spec(), bootstrap=100, min_target_improvement=1)[
        "M3a"
    ][0]
    assert actor["redistribution"]["denominator"] == -50
    assert actor["redistribution"]["numerator"] == 50
    assert actor["redistribution"]["ratio"] is None
    assert (
        "target harm did not decrease"
        in actor["redistribution"]["ratio_unavailable_reasons"]
    )


@pytest.mark.parametrize("threshold", [None, 51])
def test_practical_threshold_must_be_declared_and_met(threshold):
    actor = transfer_report(
        [association_pair()], spec(), bootstrap=100, min_target_improvement=threshold
    )["M3a"][0]
    assert actor["cells"]["soldier"]["delta"] == -50
    assert actor["redistribution"]["ratio"] is None
    assert actor["redistribution"]["minimum_target_improvement"] == threshold


def test_nonshared_content_and_incomplete_variants_are_listed():
    pair = association_pair(components=3)
    for row in pair["rows"]:
        if row["cluster"] == "component0" and row["actor"] == "civilian":
            row["comparison_unit"] = "different_content"
        if row["cluster"] == "component1" and row["actor"] == "civilian":
            row["target_mention_position"] = "different_variant"
    actor = transfer_report([pair], spec())["M3a"][0]
    assert actor["coverage"] == {
        "input_rows": 12,
        "matched_rows": 4,
        "excluded_rows": 8,
    }
    assert len(actor["unmatched"]) == 3
    assert {item["reason"] for item in actor["unmatched"]} == {
        "incomplete axis-value coverage",
        "nonmatching variant conditions or remaining attributes",
    }


def test_no_comparison_unit_cannot_establish_actor_comparison():
    pair = association_pair()
    for row in pair["rows"]:
        del row["comparison_unit"]
    actor = transfer_report([pair], spec())["M3a"][0]
    assert actor["status"] == "unavailable"
    assert len(actor["unmatched"]) == 8


def test_opposite_language_transfer_is_change_not_baseline_gap():
    uk = association_pair(
        target_before=1, target_after=0.5, other_before=1, other_after=0.5
    )
    en = association_pair(
        "warbias_triplets_en",
        "en",
        target_before=0.5,
        target_after=0.5,
        other_before=0.5,
        other_after=0.5,
    )
    language = transfer_report(
        [uk, en], spec(), bootstrap=100, min_target_improvement=1
    )["M3d"][0]
    assert language["cells"]["uk"]["before"] == 100
    assert language["cells"]["en"]["before"] == 50
    assert language["cells"]["uk"]["delta"] == -50
    assert language["cells"]["en"]["delta"] == 0
    assert language["cells"]["en"]["effect_kind"] == "opposite_language_transfer"
    assert language["redistribution"]["ratio"] == 0
    assert language["coverage"]["matched_rows"] == 16


def test_sparse_stratum_does_not_produce_confidence_or_ratio():
    actor = transfer_report(
        [association_pair(components=1)],
        spec(),
        bootstrap=100,
        min_target_improvement=1,
    )["M3a"][0]
    assert actor["cells"]["soldier"]["ci95"] is None
    assert actor["redistribution"]["ratio"] is None
    assert actor["redistribution"]["numerator"] == 50
    assert actor["bootstrap"]["completed"] == 0
    assert "fewer than two components" in actor["bootstrap"]["unavailable_reasons"][0]


def test_positive_point_improvement_without_positive_paired_ci_has_no_ratio():
    pair = association_pair()
    rows = {row["id"]: row for row in pair["rows"]}
    for score in pair["after_scored"]:
        row = rows[score["id"]]
        if row["actor"] == "soldier":
            score["stereotype_preference"] = (
                0 if row["cluster"] == "component0" else 0.9
            )
    actor = transfer_report([pair], spec(), bootstrap=500, min_target_improvement=1)[
        "M3a"
    ][0]
    assert actor["redistribution"]["denominator"] > 1
    assert actor["redistribution"]["denominator_ci95"][0] == 0
    assert actor["redistribution"]["ratio"] is None


@pytest.mark.parametrize("axis", ["gender", "age"])
def test_demographic_effect_uses_only_declared_matched_axis(axis):
    pair = association_pair()
    values = ("woman", "man") if axis == "gender" else ("adult", "older")
    for row in pair["rows"]:
        row[axis] = values[0] if row["actor"] == "soldier" else values[1]
        row["actor"] = "civilian"
    demographic = transfer_report(
        [pair], spec(**{axis: values[0]}), bootstrap=100, min_target_improvement=1
    )["M3c"][axis][0]
    assert demographic["cells"][values[0]]["delta"] == -50
    assert demographic["cells"][values[1]]["delta"] == 50
    assert demographic["redistribution"]["ratio"] == 1


def test_provisional_topics_and_grammatical_gender_are_not_inferred():
    pair = association_pair()
    for row in pair["rows"]:
        row.pop("gender")
        row.update(
            topic="war",
            pronoun_gender="masculine",
            analysis_topic="war",
            analysis_topic_distance="near",
            analysis_annotations_reviewed=False,
        )
    report = transfer_report([pair], spec(topic="war", gender="male"))
    assert report["M3b"][0]["status"] == "unavailable"
    assert report["M3c"]["gender"][0]["status"] == "unavailable"
    assert "reviewed topic" in report["M3b"][0]["unavailable_reasons"][0]


def test_reviewed_topic_names_define_target_and_near_far_cells():
    pair = association_pair(components=3)
    for row in pair["rows"]:
        component = int(row["cluster"][-1])
        row.update(
            analysis_annotations_reviewed=True,
            analysis_topic=("war", "occupation", "weather")[component],
            analysis_topic_distance=("target", "near", "far")[component],
        )
    topic = transfer_report([pair], spec(topic="war"))["M3b"][0]
    assert set(topic["cells"]) == {"target", "near", "far"}
    assert topic["cells"]["target"]["role"] == "target"
    assert topic["cells"]["near"]["role"] == "non_target"


def test_stereoset_preserves_equal_target_weights():
    pair = association_pair(task="stereoset_uk", components=3)
    for row, pre, post in zip(
        pair["rows"], pair["before_scored"], pair["after_scored"], strict=True
    ):
        component = int(row["cluster"][-1])
        row.update(
            group="large" if component < 2 else "small",
            condition="intrasentence",
            analysis_annotations_reviewed=True,
            analysis_topic="war",
            analysis_topic_distance="target",
        )
        pre["stereotype_preference"] = 1 if component < 2 else 0
        post["stereotype_preference"] = 0.5
    topic = transfer_report([pair], spec(topic="war"))["M3b"][0]
    assert topic["native_weighting"] == "target_macro"
    assert topic["cells"]["target"]["before"] == 50  # pooled rows would be 66.7


def test_qa_harm_is_explicit_bounded_and_condition_specific():
    pair = association_pair(task="warbias_expanded_qa_uk")
    for row, pre, post in zip(
        pair["rows"], pair["before_scored"], pair["after_scored"], strict=True
    ):
        row.update(condition="stereotype_conflicting", expansion_version="fixture")
        pre["unknown_rate"] = 1 if row["actor"] == "soldier" else 0
        post["unknown_rate"] = 0 if row["actor"] == "soldier" else 1
    declared = {
        **spec(),
        "harm_metric": {
            "id": "unknown_rate",
            "condition": "stereotype_conflicting",
            "unit": "rate",
        },
    }
    actor = transfer_report(
        [pair], declared, bootstrap=100, min_target_improvement=0.1
    )["M3a"][0]
    assert actor["metric"] == "unknown_rate"
    assert actor["unit"] == "rate"
    assert actor["redistribution"]["ratio"] == 1
    assert transfer_report([pair], spec())["M3a"][0]["status"] == "unavailable"
    declared["harm_metric"]["condition"] = "ambiguous"
    assert transfer_report([pair], declared)["M3a"][0]["status"] == "unavailable"


@pytest.mark.parametrize("fault", ["different_ids", "duplicate", "missing_score"])
def test_exact_pre_post_inventories_and_full_scoring_are_enforced(fault):
    pair = association_pair()
    if fault == "missing_score":
        pair["after_scored"][0]["stereotype_preference"] = None
        actor = transfer_report([pair], spec())["M3a"][0]
        assert actor["status"] == "unavailable"
        assert actor["unscored_rows"]
    else:
        if fault == "different_ids":
            pair["after_scored"][0]["id"] = "wrong_id"
        else:
            pair["after_scored"].append(pair["after_scored"][0])
        with pytest.raises(ValueError):
            transfer_report([pair], spec())


def test_seed_is_deterministic_and_invalid_controls_rejected():
    pair = association_pair()
    assert transfer_report([pair], spec(), bootstrap=30, seed=4) == transfer_report(
        [pair], spec(), bootstrap=30, seed=4
    )
    with pytest.raises(ValueError):
        transfer_report([pair], spec(), min_target_improvement=-1)
    with pytest.raises(ValueError):
        transfer_report([pair], {"training_language": "fr"})
    with pytest.raises(ValueError):
        transfer_report([pair], spec(), bootstrap=1.5)
