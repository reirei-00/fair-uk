import copy

import pytest

from lm_eval.fair_uk.comparison import align, paired_rates
from lm_eval.fair_uk.reporting import native_entries


def outcomes(values):
    return [
        {"group": group, "cluster": str(i), "stratum": "shared", "rate": value}
        for group, rates in values.items()
        for i, value in enumerate(rates)
    ]


def test_paired_bootstrap_preserves_languages_and_shared_profiles():
    uk = outcomes({"a": [1, 0], "b": [0, 1]})
    en = outcomes({"a": [0, 1], "b": [1, 0]})
    result = paired_rates(uk, en, "rate", "harm", ["a", "b"], 1000, 17)
    # Group differences vary, but each draw's maxima are equal in both languages.
    assert result["groups"]["a"]["ci95"] == [-1, 1]
    assert result["worst_delta_ci95"] == [0, 0]
    assert result["bootstrap_valid_replicates"] == 1000
    duplicated = paired_rates(uk * 5, en * 5, "rate", "harm", ["a", "b"], 1000, 17)
    assert duplicated["groups"]["a"]["source_cases"] == 2
    assert duplicated["groups"]["a"]["ci95"] == result["groups"]["a"]["ci95"]
    assert duplicated["worst_delta_ci95"] == result["worst_delta_ci95"]


def test_missing_and_single_case_groups_withhold_intervals():
    uk = outcomes({"a": [0, 1]})
    missing = paired_rates(uk, uk, "rate", "success", ["a", "b"], 100, 17)
    assert missing["missing_groups"] == ["b"]
    assert not missing["coverage_complete"]
    assert missing["worst_delta_ci95"] is None
    single = paired_rates(uk[:1], uk[:1], "rate", "success", ["a"], 100, 17)
    assert single["worst_delta_ci95"] is None


def test_alignment_rejects_different_cases_keys_and_tracks():
    row = {
        "id": "x",
        "task": "warbias_uk",
        "language": "uk",
        "cluster": "case",
        "stratum": "idp",
        "panel": "p",
        "group": "idp",
        "group_kind": "target_status",
        "condition": "ambiguous",
        "polarity": "negative",
        "gold": 2,
        "stereotype": 0,
        "unknown": 2,
        "eligible": True,
        "protocol": "strict",
    }
    other = {**row, "task": "warbias_en", "language": "en"}
    align([row], [other])
    for key, value in (
        ("id", "different"),
        ("cluster", "different"),
        ("gold", 1),
        ("task", "warbias_intersectional_en"),
    ):
        changed = copy.deepcopy(other)
        changed[key] = value
        with pytest.raises(ValueError):
            align([row], [changed])
    with pytest.raises(ValueError):
        align([row, row], [other, other])


def test_native_units_remain_benchmark_specific():
    assert list(
        native_entries({"task": "warbias_uk", "native": [{"accuracy": 0.5, "n": 20}]})
    ) == [("", "accuracy", 0.5, "rate"), ("", "n", 20, "count / selection mass")]
    assert (
        next(native_entries({"task": "stereoset_uk", "native": [{"ss": 50.0}]}))[-1]
        == "percent / percentage points"
    )
