"""Opt-in regression against the frozen WarBias baseline (saved predictions, no inference).

Set FAIR_UK_WARBIAS_GOLDEN to a directory holding ``runs/``, ``runs_neutral/``,
``v2/`` and ``v3/`` (the 16 Oct 2026 baseline layout). FAIR_UK_WARBIAS_GOLDEN_MODELS
selects models (space-separated; default ``qwen3_8b``). The report is rebuilt
from the saved predictions and its key numbers must equal the expected values.
"""

import json
import math
import os
from pathlib import Path

import pytest

from lm_eval.fair_uk.warbias.panel import panel_report


ROOT = os.environ.get("FAIR_UK_WARBIAS_GOLDEN")
EXPECTED = Path(__file__).with_name("data") / "warbias_golden_expected.json"
PHASES = {
    "train": "train_signal",
    "development": "development",
    "evaluation": "evaluation",
    "deferred": "deferred",
}
pytestmark = pytest.mark.skipif(not ROOT, reason="FAIR_UK_WARBIAS_GOLDEN not set")


def key_numbers(panel):
    """The values the paper reports, from one panel report directory."""
    out = {}
    for language in ("uk", "en"):
        report = json.loads((panel / language / "prior_normalised.json").read_text())
        out[language] = {
            "all": {
                k: report["all"][k]
                for k in ("n", "ss", "ss_ci95", "tie_rate", "mean_margin")
            },
            "cells": {
                f"{c['axis']}:{c['group']}": [
                    c["n"],
                    c["ss"],
                    c["ss_ci95"],
                    c["lms_per_byte"],
                ]
                for c in report["cells"]
            },
        }
    detection = json.loads((panel / "detection_analysis.json").read_text())
    contrast = detection["language_contrast"]
    assert contrast["language_pair"] == {"lg1": "uk", "lg2": "en"}
    out["language"] = {
        "all": [contrast["all"]["lg2_minus_lg1"], contrast["all"]["ci95"]],
        **{a: [v["lg2_minus_lg1"], v["ci95"]] for a, v in contrast["by_actor"].items()},
    }
    for language, analysis in detection["languages"].items():
        out[f"worst_{language}"] = {
            r["axis"]: [r.get("worst"), r.get("worst_ci95")]
            for r in analysis["worst_group_ss_distance"]
        }
        out[f"contrasts_{language}"] = {
            f"{c['axis']}:{c['left']}-{c['right']}": [c["right_minus_left"], c["ci95"]]
            for c in analysis["matched_profile_contrasts"]
        }
    return out


def assert_close(actual, expected, where=""):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), where
        for k in expected:
            assert_close(actual[k], expected[k], f"{where}/{k}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), where
        for k, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_close(a, e, f"{where}[{k}]")
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9), where
    else:
        assert actual == expected, where


def build(model, output):
    root = Path(ROOT)
    runs, neutral = {"uk": [], "en": []}, {"uk": [], "en": []}
    for language in ("uk", "en"):
        for bundle, run in PHASES.items():
            runs[language].append(
                (
                    root / "v2" / bundle / "manifest.json",
                    root
                    / "runs"
                    / model
                    / run
                    / f"warbias_triplets_annotated_{language}",
                )
            )
            neutral[language].append(
                (
                    root / "v3" / bundle / "manifest.json",
                    root
                    / "runs_neutral"
                    / model
                    / run
                    / f"warbias_triplets_annotated_neutral_{language}",
                )
            )
    return panel_report(
        root / "v3" / "all" / "manifest.json",
        runs,
        output,
        neutral_runs=neutral,
        min_independent_support=5,
        bootstrap=1000,
        seed=42,
    )


@pytest.mark.parametrize(
    "model", os.environ.get("FAIR_UK_WARBIAS_GOLDEN_MODELS", "qwen3_8b").split()
)
def test_panel_reproduces_frozen_baseline(model, tmp_path):
    expected = json.loads(EXPECTED.read_text())[model]
    assert_close(key_numbers(build(model, tmp_path / model)), expected, model)
