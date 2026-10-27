"""Likelihood scoring: raw length scorings, prior-normalised preference and LMS."""

import pytest

from lm_eval.fair_uk.warbias.detection import scoring_variants_report
from lm_eval.fair_uk.warbias.schema import LENGTH_SCORINGS
from lm_eval.fair_uk.warbias.scoring import (
    lms,
    prior_normalised_items,
    raw_items,
    sentence_scores,
    summed_logprobs,
)


def row(item, sentences, actor="idp", cluster="k"):
    return {
        "id": item,
        "cluster": cluster,
        "actor": actor,
        "gender": "not_stated",
        "age": "not_stated",
        "comparison_unit": "C1",
        "sentences": sentences,
    }


def test_length_scorings_reconstruct_sum_and_byte_normalisation():
    prediction = {"id": "a", "scores": [-1.0, -0.6, -2.0], "token_counts": [2, 5, 1]}
    sentences = ["ab", "abcdefgh", "a"]
    assert summed_logprobs(prediction) == [-2.0, -3.0, -2.0]
    assert sentence_scores(prediction, sentences, "per_token") == [-1.0, -0.6, -2.0]
    assert sentence_scores(prediction, sentences, "sum") == [-2.0, -3.0, -2.0]
    assert sentence_scores(prediction, sentences, "per_byte") == [-1.0, -3.0 / 8, -2.0]
    assert lms([-1.0, -3.0, -2.0]) == 0.5
    with pytest.raises(ValueError, match="Unknown scoring"):
        sentence_scores(prediction, sentences, "per_word")
    with pytest.raises(ValueError, match="three scores"):
        summed_logprobs({"id": "b", "scores": [-1.0], "token_counts": [1]})


def test_scorings_disagree_and_report_has_no_icat():
    rows = [row("a", ["ab", "abcdefgh", "a"])]
    predictions = [{"id": "a", "scores": [-1.0, -0.6, -2.0], "token_counts": [2, 5, 1]}]
    per_token = raw_items(rows, predictions, "per_token")[0]
    total = raw_items(rows, predictions, "sum")[0]
    assert per_token.preference == 0.0 and total.preference == 1.0
    variants = scoring_variants_report(rows, predictions)["variants"]
    assert (
        variants["sum"]["all"]["ss"] == 1.0
        and variants["per_token"]["all"]["ss"] == 0.0
    )
    assert (
        variants["per_byte"]["idp"]["lms"] == 1.0
        and variants["sum"]["idp"]["lms"] == 0.0
    )
    assert all("icat" not in record for v in variants.values() for record in v.values())
    with pytest.raises(ValueError, match="exactly the panel"):
        raw_items(rows + [row("b", ["x", "y", "z"])], predictions, "sum")


def test_prior_normalised_preference_cancels_shared_wording():
    rows = [
        row("a", ["s s s", "c c c c c", "u"]),
        row("b", ["s", "c", "u"], actor="veteran"),
    ]
    predictions = [
        {"id": "a", "scores": [-2.0, -2.0, -3.0], "token_counts": [10, 20, 10]},
        {"id": "b", "scores": [-1.0, -1.0, -3.0], "token_counts": [1, 1, 1]},
    ]
    # Only "a" has a neutral variant: the group term raises S (delta 5) over C (delta -18).
    neutral_rows = [{"id": "a"}]
    neutral_predictions = [
        {"id": "a", "scores": [-2.5, -1.1, -3.0], "token_counts": [10, 20, 10]}
    ]
    items = prior_normalised_items(rows, predictions, neutral_rows, neutral_predictions)
    assert [i.id for i in items] == ["a"]
    item = items[0]
    assert item.preference == 1.0 and item.tie == 0.0
    assert item.delta_stereotype == pytest.approx(5.0)
    assert item.delta_counter == pytest.approx(-18.0)
    assert item.margin == pytest.approx(23.0)
    assert item.length_difference == 2
    assert item.lms_per_token == 1.0
    with pytest.raises(ValueError, match="cover exactly"):
        prior_normalised_items(rows, predictions, neutral_rows, [])


@pytest.mark.parametrize("count", [0, -1, True, 1.5, float("nan"), float("inf")])
def test_summed_logprobs_requires_positive_integer_token_counts(count):
    with pytest.raises(ValueError, match="positive integer token counts"):
        summed_logprobs(
            {"id": "bad", "scores": [-1.0, -2.0, -3.0], "token_counts": [count, 1, 1]}
        )


@pytest.mark.parametrize("score", [True, None, "-1", float("nan"), float("inf")])
def test_likelihood_scores_must_be_finite_numbers(score):
    prediction = {"id": "bad", "scores": [score, -2.0, -3.0], "token_counts": [1, 1, 1]}
    for scoring in LENGTH_SCORINGS:
        with pytest.raises(ValueError, match="finite numbers"):
            sentence_scores(prediction, ["S", "C", "U"], scoring)


def test_summed_logprobs_rejects_nonfinite_reconstruction():
    with pytest.raises(ValueError, match="must be finite"):
        summed_logprobs({"scores": [-1e308, -2.0, -3.0], "token_counts": [2, 1, 1]})
