import copy
import json

import httpx
import pytest

from lm_eval.fair_uk.data import adapt
from lm_eval.fair_uk.gemini_runner import (
    identity,
    response_text,
    run,
    usage_summary,
    validate_prediction,
)


MODEL = "gemini-3.6-flash"


def row():
    return adapt(
        "warbias_uk",
        {
            "id": "one",
            "scenario_id": "s",
            "source_case_id": "c",
            "target_group": "idp",
            "evidence_condition": "ambiguous",
            "question_polarity": "negative",
            "answer_index": 2,
            "stereotype_consistent_index": 0,
            "unknown_index": 2,
            "choices": ["one", "two", "unknown"],
            "context": "Дві людини.",
            "question": "Хто?",
        },
    )


def raw():
    return {
        "modelVersion": MODEL,
        "candidates": [{"content": {"parts": [{"text": "C"}]}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 1,
            "thoughtsTokenCount": 0,
        },
    }


def test_gemini_request_checkpoint_and_metering(tmp_path):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        assert (
            str(request.url)
            == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"
        )
        return httpx.Response(200, json=raw())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        predictions, manifest = run([row()], {}, MODEL, 42, tmp_path, client=client)
        run([row()], {}, MODEL, 42, tmp_path, client=client)
    assert len(calls) == 1
    assert calls[0]["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "minimal"
    }
    assert manifest["model_identity"]["provider"] == "google"
    assert usage_summary(predictions, MODEL)[
        "standard_paid_tier_cost_estimate_usd"
    ] == pytest.approx(0.00007875)
    corrupted = copy.deepcopy(predictions[0])
    corrupted["response"] = "A"
    with pytest.raises(ValueError, match="raw Gemini"):
        validate_prediction(row(), corrupted, identity(MODEL, 42))


def test_gemini_blocks_thoughts_and_malformed_responses():
    assert response_text({"promptFeedback": {"blockReason": "SAFETY"}}) == ""
    response = raw()
    response["candidates"][0]["content"]["parts"].insert(
        0, {"text": "private intermediate text", "thought": True}
    )
    assert response_text(response) == "C"
    with pytest.raises(ValueError, match="no candidate"):
        response_text({})
    response["candidates"] *= 2
    with pytest.raises(ValueError, match="multiple candidates"):
        response_text(response)


def test_gemini_http_failure_is_not_a_model_answer(tmp_path):
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    403, json={"error": {"message": "sensitive details"}}
                )
            )
        ) as client,
        pytest.raises(RuntimeError, match="HTTP 403") as error,
    ):
        run([row()], {}, MODEL, 42, tmp_path, client=client)
    assert "sensitive details" not in str(error.value)
    assert not (tmp_path / "predictions.jsonl").exists()


def test_gemini_daily_quota_stops_without_retry(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            429,
            json={
                "error": {
                    "details": [
                        {
                            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
                        },
                        {"retryDelay": "27s"},
                    ]
                }
            },
        )

    def unexpected_sleep(_):
        pytest.fail("Daily quota must not trigger retry waits")

    monkeypatch.setattr("lm_eval.fair_uk.gemini_runner.time.sleep", unexpected_sleep)
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(RuntimeError, match="HTTP 429"),
    ):
        run([row()], {}, MODEL, 42, tmp_path, client=client)
    assert len(calls) == 1
    assert not (tmp_path / "predictions.jsonl").exists()


def test_gemini_minute_quota_retries_then_saves_one_answer(tmp_path, monkeypatch):
    calls, delays = [], []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                429, json={"error": {"details": [{"retryDelay": "7s"}]}}
            )
        return httpx.Response(200, json=raw())

    monkeypatch.setattr("lm_eval.fair_uk.gemini_runner.time.sleep", delays.append)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        predictions, _ = run([row()], {}, MODEL, 42, tmp_path, client=client)
    assert delays == [7]
    assert len(calls) == 2
    assert len(predictions) == 1
    assert predictions[0]["response"] == "C"
    assert len((tmp_path / "predictions.jsonl").read_text().splitlines()) == 1
