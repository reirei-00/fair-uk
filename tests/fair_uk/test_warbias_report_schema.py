"""Saved schema normalization is pure, idempotent and independent of files."""

from copy import deepcopy

import pytest

from lm_eval.fair_uk.warbias.report_schema import (
    LEGACY_REPORT_VERSION,
    REPORT_VERSION,
    normalize_detection_report,
    report_language_pair,
)


def legacy_report():
    block = {"n": 2, "uk": 0.25, "en": 0.75, "en_minus_uk": 0.5, "ci95": [-0.2, 0.8]}
    return {
        "version": LEGACY_REPORT_VERSION,
        "languages": {"uk": {"status": "available"}, "en": {}},
        "language_contrast": {
            "all": block,
            "by_actor": {"idp": deepcopy(block), "veteran": None},
            "cells": [{"preference": deepcopy(block), "lms_per_byte": None}],
        },
    }


def test_legacy_normalization_preserves_values_and_input():
    source = legacy_report()
    before = deepcopy(source)
    canonical = normalize_detection_report(source)
    assert source == before
    assert canonical["version"] == REPORT_VERSION
    assert canonical["source_report_version"] == LEGACY_REPORT_VERSION
    assert report_language_pair(canonical) == ("uk", "en")
    assert canonical["language_contrast"]["all"] == {
        "n": 2,
        "lg1": 0.25,
        "lg2": 0.75,
        "lg2_minus_lg1": 0.5,
        "ci95": [-0.2, 0.8],
    }
    assert canonical["language_contrast"]["by_actor"]["veteran"] is None
    assert canonical["language_contrast"]["cells"][0]["lms_per_byte"] is None
    assert normalize_detection_report(canonical) == canonical
    canonical["languages"]["uk"]["status"] = "edited"
    assert source == before


def test_current_report_is_copied_without_reinterpreting_pair():
    report = {
        "version": REPORT_VERSION,
        "languages": {"fr": {}, "de": {}},
        "language_contrast": {"language_pair": {"lg1": "de", "lg2": "fr"}},
    }
    canonical = normalize_detection_report(report)
    assert report_language_pair(canonical) == ("de", "fr")
    assert canonical == report and canonical is not report


@pytest.mark.parametrize(
    "report, error",
    [
        (None, TypeError),
        ([], TypeError),
        ({}, ValueError),
        ({"version": "future"}, ValueError),
    ],
)
def test_unknown_shapes_and_versions_are_not_guessed(report, error):
    with pytest.raises(error):
        normalize_detection_report(report)


def test_malformed_legacy_report_is_not_filled_with_defaults():
    source = legacy_report()
    del source["language_contrast"]["all"]["en_minus_uk"]
    before = deepcopy(source)
    with pytest.raises(ValueError, match="version 2 language difference"):
        normalize_detection_report(source)
    assert source == before
