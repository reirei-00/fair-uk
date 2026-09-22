import copy

import pytest

from lm_eval.fair_uk.comparison import align, compare_rows, validate_model_versions
from lm_eval.fair_uk.reporting import paired_markdown


def stereo_rows(language, preferences):
    return [
        {
            "id": str(i),
            "cluster": str(i),
            "panel": str(i),
            "task": f"stereoset_{language}",
            "language": language,
            "stratum": "target",
            "group": "target",
            "group_kind": "lexical_target",
            "condition": "intrasentence",
            "category": "profession",
            "eligible": True,
            "protocol": "causal_full_sentence_mean_token_logprob_v1",
            "lms": 1.0,
            "stereotype_preference": value,
            "stereotype_tie": 0.0,
            "tie": 0.0,
        }
        for i, value in enumerate(preferences)
    ]


def test_paired_native_recomputes_nonlinear_icat():
    uk, en = stereo_rows("uk", [0, 1]), stereo_rows("en", [1, 1])
    report = compare_rows(uk, en, uk, en, bootstrap=100, seed=3)
    icat = next(
        r
        for r in report["native_comparisons"]
        if r["scope"] == "target_macro" and r["metric"] == "icat"
    )
    assert (icat["uk"], icat["en"], icat["delta_en_minus_uk"]) == (100, 0, -100)
    assert icat["ci95"] == [-100, 0]
    assert icat["bootstrap_valid_replicates"] == 100
    assert "Dataset-specific metric comparisons" in paired_markdown(report)
    assert "Additional group comparisons" in paired_markdown(report)


def test_same_predictions_have_zero_paired_uncertainty():
    uk, en = stereo_rows("uk", [0, 1]), stereo_rows("en", [0, 1])
    report = compare_rows(uk, en, uk, en, bootstrap=100, seed=7)
    assert all(
        r["ci95"] == [0, 0]
        for r in report["native_comparisons"]
        if r["delta_en_minus_uk"] is not None
    )


def test_only_declared_unpaired_rows_can_be_excluded():
    uk, en = stereo_rows("uk", [0, 1]), stereo_rows("en", [0, 1])
    with pytest.raises(ValueError, match="IDs"):
        compare_rows(uk, en[:1], uk, en, bootstrap=0)
    uk[1]["paired_eligible"] = False
    report = compare_rows(uk, en[:1], uk, en[:1], bootstrap=0)
    assert report["paired_rows"] == 1
    assert report["coverage"]["uk"]["excluded_unpaired_rows"] == 1
    assert report["coverage"]["uk"]["evaluated_rows"] == 2


def test_category_swap_cannot_masquerade_as_language_difference():
    uk, en = stereo_rows("uk", [0, 1]), stereo_rows("en", [0, 1])
    uk[0]["category"] = en[0]["category"] = "gender"
    en[0]["category"], en[1]["category"] = en[1]["category"], en[0]["category"]
    with pytest.raises(ValueError, match="metadata"):
        align(uk, en)


def test_hosted_alias_does_not_hide_returned_model_changes():
    evidence = {"returned_models": ["actual-v1"], "responses_without_model_version": 0}
    assert (
        validate_model_versions(evidence, evidence)["status"]
        == "matching_returned_identifiers"
    )
    changed = copy.deepcopy(evidence)
    changed["returned_models"] = ["actual-v2"]
    with pytest.raises(ValueError, match="model versions"):
        validate_model_versions(evidence, changed)
    changed["returned_models"] = ["actual-v1", "actual-v2"]
    with pytest.raises(ValueError):
        validate_model_versions(changed, changed)
    missing = {"returned_models": [], "responses_without_model_version": 1}
    assert (
        validate_model_versions(evidence, missing)["status"]
        == "version_evidence_incomplete"
    )
