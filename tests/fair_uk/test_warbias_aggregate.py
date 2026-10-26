"""Scoring-agnostic group analyses: one definition of groups, bootstrap and contrasts."""

import pytest

from lm_eval.fair_uk.warbias.aggregate import (
    COMMON,
    MARGINAL,
    actor_matched,
    group_cells,
    group_panel,
    language_difference,
    matched_profile_contrasts,
    worst_group,
)
from lm_eval.fair_uk.warbias.schema import TripletItem
from lm_eval.fair_uk.warbias.stats import (
    component_bootstrap,
    component_mean,
    resample,
    spearman,
)


def item(
    i,
    cluster,
    unit,
    actor="idp",
    gender="not_stated",
    age="not_stated",
    preference=1.0,
    **extra,
):
    return TripletItem(
        id=f"t{i}",
        cluster=cluster,
        actor=actor,
        gender=gender,
        age=age,
        comparison_unit=unit,
        preference=preference,
        **extra,
    )


def gender_panel(components=6):
    """Each component holds one case with men, women and not_stated versions."""
    items, k = [], 0
    for c in range(components):
        for gender, preference, lms_value in (
            ("men", 1.0, 0.5),
            ("women", 0.0, 1.0),
            ("not_stated", 1.0, 1.0),
        ):
            items.append(
                item(
                    k,
                    f"c{c}",
                    f"u{c}",
                    actor="idp" if c < 3 else "veteran",
                    gender=gender,
                    preference=preference,
                    lms_per_byte=lms_value,
                )
            )
            k += 1
    return items


def test_component_mean_weights_components_and_skips_missing():
    items = [
        item(0, "a", "u", preference=1.0),
        item(1, "a", "u", preference=0.0),
        item(2, "b", "v", preference=1.0),
        item(3, "b", "v", preference=None),
    ]
    assert component_mean(items, "preference") == pytest.approx(0.75)
    assert component_mean([item(0, "a", "u", preference=None)], "preference") is None
    doubled = resample(items, ["a", "a"])
    assert {i.cluster for i in doubled} == {"0:a", "1:a"} and len(doubled) == 4
    assert component_bootstrap(items[:2], lambda s: 1.0, 10, 1) is None  # one component
    assert (
        component_bootstrap(items, lambda s: component_mean(s, "preference"), 50, 1)
        is not None
    )
    assert spearman([1, 2], [2, 1]) is None and spearman([1, 1, 1], [1, 2, 3]) is None


def test_groups_are_defined_once_with_not_stated_as_reference():
    items = gender_panel()
    panel = group_panel(items, "gender", minimum=5)
    assert panel.kind == COMMON and panel.values == ("men", "women")
    cells = {
        (axis, group): (role, basis, members)
        for axis, group, role, basis, members in group_cells(items, 5)
    }
    assert cells[("gender", "not_stated")][0] == "reference_not_compared"
    assert cells[("gender", "men")][1] == COMMON
    assert cells[("actor", "idp")][1] == MARGINAL
    assert len(cells[("actor×gender", "idp×men")][2]) == 3
    assert ("actor×gender", "idp×not_stated") not in cells


def test_worst_group_and_m1_compare_stated_groups_with_intervals():
    items = gender_panel()
    distance = {
        r["axis"]: r
        for r in worst_group(
            items, "preference", success=False, distance_from=0.5, draws=50, seed=1
        )
    }
    gender = distance["gender"]
    assert set(gender["groups"]) == {"men", "women"} and gender["worst"] == 0.5
    assert gender["reference_not_compared"]["not_stated"]["value"] == 1.0
    assert gender["worst_ci95"] == [0.5, 0.5]
    assert (
        distance["actor"]["status"] == "fewer_than_two_supported_groups"
    )  # 3 components each
    m1 = {r["axis"]: r for r in worst_group(items, "lms_per_byte", draws=50, seed=1)}[
        "gender"
    ]
    assert m1["m1"] == 0.5 and m1["m1b"] == 0.5 and m1["worst_groups"] == ["men"]
    assert m1["m1_ci95"] == [0.5, 0.5]


def test_matched_contrasts_and_language_difference():
    items = gender_panel()
    contrast = next(
        c
        for c in matched_profile_contrasts(items, "preference")
        if (c["left"], c["right"]) == ("men", "women")
    )
    assert contrast["right_minus_left"] == -1.0 and contrast["matched_cases"] == 6
    assert contrast["disclosure_contrast"] is False
    lg2 = [TripletItem(**{**i.to_dict(), "preference": 1.0}) for i in items]
    difference = language_difference(items, lg2, draws=20, seed=1)
    assert difference["lg2_minus_lg1"] == pytest.approx(1 / 3)
    assert difference["n"] == len(items)


def test_actor_matched_exact_and_family_levels_with_paired_change():
    def row(i, actor, cluster, family, claim, preference, gender="not_stated"):
        return {
            "id": f"r{i}",
            "actor": actor,
            "gender": gender,
            "age": "not_stated",
            "cluster": cluster,
            "claim_family_id": family,
            "cross_actor_claim_id": claim,
            "stereotype_preference": preference,
            "lms": 1.0,
        }

    before = [
        row(1, "idp", "c1", "fam", "k1", 1.0),
        row(2, "veteran", "c1", "fam", "k1", 0.0),
        row(3, "idp", "c2", "fam2", None, 1.0),
        row(4, "veteran", "c2", "fam2", None, 1.0),
        row(5, "idp", "c3", "fam3", None, 1.0),  # no veteran counterpart
        row(6, "veteran", "c2", "fam2", None, 0.0, gender="women"),  # no idp profile
    ]
    after = [
        {**r, "stereotype_preference": 0.0} if r["actor"] == "idp" else r
        for r in before
    ]
    result = actor_matched(
        before, after, fields=("stereotype_preference", "lms"), draws=20, seed=1
    )
    levels = {lv["matching_level"]: lv for lv in result["levels"]}
    exact, family = levels["exact_claim"], levels["related_claim_family"]
    assert exact["matched_cells"] == 1 and exact["rows"] == {"idp": 1, "veteran": 1}
    preference = {m["metric"]: m for m in exact["metrics"]}["stereotype_preference"]
    assert preference["veteran_minus_idp"] == -1.0
    assert preference["paired_change"] == {"idp": -1.0, "veteran": 0.0}
    assert family["matched_cells"] == 1 and family["unmatched_units"] == ["fam3"]
    with pytest.raises(ValueError, match="identical IDs"):
        actor_matched(before, after[:-1])


def unbalanced_actor_rows():
    """Unequal rows, cells and units, plus differing missingness per checkpoint."""
    specs = [
        ("idp", "a", "one", "men", 0.0, 0.2),
        ("idp", "a", "one", "men", 1.0, 0.6),
        ("veteran", "a", "one", "men", 0.2, 0.4),
        ("idp", "a", "one", "women", 0.9, 0.7),
        ("veteran", "a", "one", "women", 0.1, 0.5),
        ("idp", "a", "two", "men", 0.0, 1.0),
        ("veteran", "a", "two", "men", 1.0, 0.0),
        ("idp", "b", "three", "men", 1.0, 0.0),
        ("veteran", "b", "three", "men", 0.0, 1.0),
        ("idp", "c", "four", "men", None, 0.2),
        ("veteran", "c", "four", "men", 0.9, 0.8),
        ("idp", "d", "five", "men", None, None),
        ("veteran", "d", "five", "men", None, None),
        ("idp", "e", "six", "men", 0.6, 0.3),
        ("veteran", "e", "six", "men", 0.8, 0.7),
        ("idp", "a", "unmatched", "men", 0.99, 0.99),
    ]
    before = [
        {
            "id": f"r{k}",
            "actor": actor,
            "cluster": cluster,
            "cross_actor_claim_id": unit,
            "gender": gender,
            "age": "not_stated",
            "preference": preference,
            "lms_per_byte": quality,
        }
        for k, (actor, cluster, unit, gender, preference, quality) in enumerate(specs)
    ]
    after = [dict(row) for row in before]
    after[0]["preference"] = 0.8
    after[9]["preference"] = 0.4
    after[11]["preference"] = 0.4
    after[12]["preference"] = 0.6
    after[14]["preference"] = None
    return before, after


@pytest.mark.parametrize("draws", [0, 79])
def test_actor_estimates_match_frozen_unbalanced_reference(draws):
    before, after = unbalanced_actor_rows()
    result = actor_matched(before, after, draws=draws, seed=7)
    level = result["levels"][0]
    assert (
        level["matched_units"],
        level["matched_cells"],
        level["sampling_units"],
    ) == (6, 7, 5)
    assert level["rows"] == {"idp": 8, "veteran": 7}
    assert level["unmatched_units"] == ["unmatched"]
    # Frozen before the refactor, with equal components after the documented
    # row/cell/unit averages. Missing values are excluded per checkpoint.
    expected = [
        {
            "metric": "preference",
            "unit": "rate",
            "idp": 0.65,
            "veteran": 0.4583333333333333,
            "veteran_minus_idp": -0.1916666666666667,
            "post": {"idp": 0.5625, "veteran": 0.51875},
            "paired_change": {
                "idp": -0.08750000000000002,
                "veteran": 0.06041666666666673,
            },
        },
        {
            "metric": "lms_per_byte",
            "unit": "rate",
            "idp": 0.31875,
            "veteran": 0.68125,
            "veteran_minus_idp": 0.36250000000000004,
            "post": {"idp": 0.31875, "veteran": 0.68125},
            "paired_change": {"idp": 0.0, "veteran": 0.0},
        },
    ]
    if draws:
        expected[0].update(
            veteran_minus_idp_ci95=[-0.7224999999999999, 0.22500000000000003],
            paired_change_ci95={
                "idp": [-0.26725, 0.08000000000000007],
                "veteran": [-0.16916666666666674, 0.29937500000000095],
            },
        )
        expected[1].update(
            veteran_minus_idp_ci95=[-0.26499999999999996, 0.8],
            paired_change_ci95={"idp": [0.0, 0.0], "veteran": [0.0, 0.0]},
        )
    assert level["metrics"] == expected
    assert result == actor_matched(before, after, draws=draws, seed=7)


def test_custom_actor_pair_and_reversal_preserve_values_and_interval_sign():
    before, after = unbalanced_actor_rows()
    custom = {"idp": "group_a", "veteran": "group_b"}
    first, second = [
        [{**row, "actor": custom[row["actor"]]} for row in rows]
        for rows in (before, after)
    ]
    normal = actor_matched(
        first, second, actor_pair=("group_a", "group_b"), draws=79, seed=7
    )
    reverse = actor_matched(
        first, second, actor_pair=("group_b", "group_a"), draws=79, seed=7
    )
    assert normal["levels"][0]["rows"] == {"group_a": 8, "group_b": 7}
    for forward, backward in zip(
        normal["levels"][0]["metrics"], reverse["levels"][0]["metrics"], strict=True
    ):
        assert "idp" not in forward and "veteran" not in forward
        assert forward["group_a"] == backward["group_a"]
        assert forward["group_b"] == backward["group_b"]
        assert backward["group_a_minus_group_b"] == -forward["group_b_minus_group_a"]
        ci = forward["group_b_minus_group_a_ci95"]
        assert backward["group_a_minus_group_b_ci95"] == pytest.approx([-ci[1], -ci[0]])
        assert forward["post"] == backward["post"]
        assert forward["paired_change"] == backward["paired_change"]
        assert forward["paired_change_ci95"] == backward["paired_change_ci95"]


def test_bootstrap_keeps_repeated_component_draws(monkeypatch):
    before, after = unbalanced_actor_rows()
    monkeypatch.setattr(
        "lm_eval.fair_uk.warbias.aggregate.bootstrap_draws",
        lambda *args: [["a", "a", "b", "b", "b"]],
    )
    metric = actor_matched(before, after, fields=("preference",), draws=1)["levels"][0][
        "metrics"
    ][0]
    # Two copies of component a and three of b; deduplication would give -0.3875.
    assert metric["veteran_minus_idp_ci95"] == pytest.approx([-0.51, -0.51])
    assert metric["paired_change_ci95"]["idp"] == pytest.approx([0.04, 0.04])


def test_missing_metric_remains_unavailable_before_and_after():
    before, after = unbalanced_actor_rows()
    metric = actor_matched(before, after, fields=("absent",), draws=20)["levels"][0][
        "metrics"
    ][0]
    assert metric == {
        "metric": "absent",
        "unit": "rate",
        "idp": None,
        "veteran": None,
        "veteran_minus_idp": None,
        "veteran_minus_idp_ci95": None,
        "post": None,
        "paired_change": None,
        "paired_change_ci95": {"idp": None, "veteran": None},
    }


@pytest.mark.parametrize("pair", [None, "ab", ("a",), ("a", "a"), ("a", ""), ("a", 1)])
def test_invalid_actor_pair_is_rejected(pair):
    with pytest.raises(ValueError, match="actor pair"):
        actor_matched([], actor_pair=pair)
