"""Language comparisons use ordered slots independently of dataset language codes."""

import pytest

from lm_eval.fair_uk.warbias.aggregate import language_difference
from lm_eval.fair_uk.warbias.detection import REPORT_VERSION, detection_analysis
from lm_eval.fair_uk.warbias.schema import TripletItem, validate_language_pair


def items(preferences):
    return [
        TripletItem(
            id=f"item{i}",
            cluster="a" if i < 2 else "b",
            actor="idp",
            gender="not_stated",
            age="not_stated",
            comparison_unit=f"case{i}",
            preference=preference,
            lms_per_byte=0.5,
        )
        for i, preference in enumerate(preferences)
    ]


def test_difference_is_component_weighted_and_reverses_with_slots():
    lg1, lg2 = items([0.0, 0.0, 1.0]), items([1.0, 1.0, 1.0])
    forward = language_difference(lg1, lg2, draws=100, seed=7)
    reverse = language_difference(lg2, lg1, draws=100, seed=7)
    assert set(forward) == {
        "n",
        "sampling_units",
        "lg1",
        "lg2",
        "lg2_minus_lg1",
        "ci95",
    }
    assert forward["lg1"] == 0.5 and forward["lg2"] == 1.0
    assert forward["lg2_minus_lg1"] == 0.5  # components, not 2/3 by rows
    assert reverse["lg1"] == forward["lg2"]
    assert reverse["lg2"] == forward["lg1"]
    assert reverse["lg2_minus_lg1"] == -forward["lg2_minus_lg1"]
    assert reverse["ci95"] == pytest.approx([-x for x in reversed(forward["ci95"])])


def test_report_uses_explicit_pair_instead_of_mapping_insertion_order():
    first, second = items([0.0, 0.0, 1.0]), items([1.0, 1.0, 1.0])
    ordered = detection_analysis(
        {"fr": first, "de": second}, language_pair=("fr", "de"), draws=20
    )
    reordered = detection_analysis(
        {"de": second, "fr": first}, language_pair=("fr", "de"), draws=20
    )
    assert ordered["version"] == REPORT_VERSION == "warbias_detection_report_v3"
    contrast = ordered["language_contrast"]
    assert contrast == reordered["language_contrast"]
    assert contrast["language_pair"] == {"lg1": "fr", "lg2": "de"}
    assert contrast["all"]["lg2_minus_lg1"] == 0.5
    assert contrast["by_actor"]["idp"]["lg2_minus_lg1"] == 0.5
    assert all(cell["preference"]["lg2_minus_lg1"] == 0.5 for cell in contrast["cells"])
    reversed_pair = detection_analysis(
        {"fr": first, "de": second}, language_pair=("de", "fr"), draws=20
    )["language_contrast"]
    assert reversed_pair["language_pair"] == {"lg1": "de", "lg2": "fr"}
    assert reversed_pair["all"]["lg2_minus_lg1"] == -0.5


@pytest.mark.parametrize("pair", [("fr", "fr"), ("fr",), ("fr", ""), "fr", None])
def test_invalid_pairs_are_rejected(pair):
    with pytest.raises(ValueError, match="two distinct nonempty codes"):
        validate_language_pair(pair)


def test_pair_must_be_present_in_supplied_panels():
    with pytest.raises(ValueError, match="must be present"):
        detection_analysis({"fr": items([1.0])}, language_pair=("fr", "de"))


def test_no_common_valid_scores_remain_unavailable():
    first, second = items([None, None, None]), items([1.0, 1.0, 1.0])
    assert language_difference(first, second) is None
