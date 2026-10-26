"""Generic fair-uk reports on annotated panel rows (raw-scoring diagnostics)."""

import pytest
from warbias_support import scores_for

from lm_eval.fair_uk.baseline_metrics import baseline_profile
from lm_eval.fair_uk.data import load
from lm_eval.fair_uk.expansion_metrics import length_diagnostics, native
from lm_eval.fair_uk.metrics import make_report, score_all
from lm_eval.fair_uk.warbias.prepare import prepare


UK = "warbias_triplets_annotated_uk"


@pytest.fixture
def panel(warbias_inputs):
    files, inv, tmp = warbias_inputs
    return load(UK, dataset_bundle=prepare(files["uk"], files["en"], inv, tmp / "all"))


def test_native_marginals_report_and_profile_include_length_diagnostics(panel):
    predictions = scores_for(panel)
    scored = score_all(panel, predictions)
    summaries = native(scored)
    actors = {r["actor"]: r for r in summaries if r["scope"] == "actor"}
    assert set(actors) == {"idp", "veteran"}
    assert actors["idp"]["n"] == 3 and actors["idp"]["sampling_units"] == 1
    assert {r["gender"] for r in summaries if r["scope"] == "gender"} == {
        "men",
        "not_stated",
        "women",
    }
    report = make_report(scored, panel, bootstrap=0)
    assert report["status"] == "released_annotated_panel"
    assert report["length_diagnostics"]["status"] == "available"
    diagnostics = baseline_profile(panel, predictions)["length_diagnostics"]
    assert diagnostics["rows_counter_longer"] == len(panel)
    bucket = {(b["tolerance_words"], b["bucket"]): b for b in diagnostics["buckets"]}
    assert bucket[0, "counter_longer"]["n"] == len(panel)
    assert bucket[0, "within_tolerance"]["ss"] is None


def test_length_diagnostics_edges():
    rows = [
        {
            "id": f"x{i}",
            "cluster": f"c{i}",
            "stereotype_preference": float(i < 2),
            "lms": 1.0,
            "candidate_means": [-1.0 - i, -1.0, -5.0],
        }
        for i in range(4)
    ]
    sentences = {f"x{i}": ["a b", "a b " + "c " * i, "u v"] for i in range(4)}
    diagnostics = length_diagnostics(rows, sentences)
    assert (
        diagnostics["rows_equal_length"] == 1
        and diagnostics["rows_counter_longer"] == 3
    )
    assert diagnostics[
        "spearman_length_difference_vs_association_margin"
    ] == pytest.approx(-1.0)
    assert length_diagnostics(rows)["status"] == "unavailable"
