"""Shared BBQ draws must match separate draws exactly, including undefined CIs."""

from copy import deepcopy

import pytest

from lm_eval.fair_uk import intervention_metrics as metrics
from lm_eval.fair_uk.data import PROTOCOLS


def inputs():
    rows = []
    for category in ("one", "two"):
        for component in range(3):
            for condition in ("ambiguous", "disambiguated"):
                identifier = f"{category}-{component}-{condition}"
                rows.append(
                    {
                        "id": identifier,
                        "task": "bbq_uk",
                        "language": "uk",
                        "protocol": PROTOCOLS["bbq_uk"],
                        "cluster": f"{category}-{component}",
                        "stratum": category,
                        "group": category,
                        "condition": condition,
                        "eligible": True,
                        "gold": 2 if condition == "ambiguous" else 0,
                        "stereotype": 0,
                        "unknown": 2,
                        "choices": ["target", "other", "unknown"],
                        "source": {"context_uk": "Evidence", "question_uk": "Who?"},
                    }
                )
    scores = [
        [3, 2, 1, 2, 1, 3, 1, 3, 2],
        [2, 3, 1, 3, 1, 2, 1, 2, 3],
        [1, 2, 3, 2, 3, 1, 3, 1, 2],
    ]
    before = [
        {"id": row["id"], "scores": scores[index % 3]} for index, row in enumerate(rows)
    ]
    after = [
        {"id": row["id"], "scores": scores[(index + 1) % 3]}
        for index, row in enumerate(rows)
    ]
    return rows, before, after


@pytest.mark.parametrize("draws", [0, 20, 100, 1000])
@pytest.mark.parametrize("seed", [41, 42])
def test_same_results_as_independent_metric_draws(monkeypatch, draws, seed):
    rows, before, after = inputs()
    original_inputs = deepcopy((rows, before, after))
    calls = 0
    native = metrics.native_summaries

    def count(*args, **kwargs):
        nonlocal calls
        calls += 1
        return native(*args, **kwargs)

    monkeypatch.setattr(metrics, "native_summaries", count)
    shared = metrics.compare_predictions(
        rows, before, after, bootstrap=draws, seed=seed
    )
    shared_calls = calls

    def independent(pairs, statistic, names, bootstrap, seed):
        return {
            name: metrics._interval(
                pairs,
                lambda sample, name=name: statistic(sample)[name],
                bootstrap,
                seed,
            )
            for name in names
        }

    monkeypatch.setattr(metrics, "_shared_intervals", independent)
    calls = 0
    separate = metrics.compare_predictions(
        rows, before, after, bootstrap=draws, seed=seed
    )
    assert shared == separate
    assert (rows, before, after) == original_inputs
    if draws:
        assert shared_calls < calls / 2
