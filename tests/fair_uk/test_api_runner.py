import copy
import json
from unittest.mock import Mock

import httpx
import pytest

from lm_eval.fair_uk import api_runner
from lm_eval.fair_uk.data import adapt, prompts


def row(identifier="one"):
    return adapt(
        "warbias_uk",
        {
            "id": identifier,
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


def config(provider="openai-compatible", **extra):
    return {"provider": provider, "model": "any-user-selected-model", **extra}


def raw(provider="openai-compatible"):
    if provider == "gemini":
        return {
            "modelVersion": "actual-provider-snapshot",
            "candidates": [
                {"content": {"parts": [{"text": "C"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 1,
                "thoughtsTokenCount": 5,
            },
        }
    return {
        "model": "actual-provider-snapshot",
        "choices": [{"message": {"content": "C"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 20},
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
    }


@pytest.mark.parametrize("provider", ["openai-compatible", "gemini"])
def test_planning_is_pure_and_no_model_or_price_allowlist(provider, monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("Planning accessed credentials or HTTP")

    monkeypatch.setattr(api_runner.os.environ, "get", unexpected)
    monkeypatch.setattr(httpx, "Client", unexpected)
    chosen = config(provider)
    before = copy.deepcopy(chosen)
    model = api_runner.identity(chosen)
    plan = api_runner.estimate([row()], chosen)
    assert chosen == before
    assert model["model"] == "any-user-selected-model"
    assert model["capabilities"] == ["generate"]
    assert "temperature" not in model["parameters"]
    assert "seed" not in model["parameters"]
    assert "thinkingConfig" not in model["parameters"]
    assert plan["pricing_status"] == "unavailable"
    assert plan["estimated_cost_usd"] is None
    assert plan["configured_output_token_allowance"] == 128


def test_configurable_endpoint_parameters_and_pricing_are_separate():
    chosen = config(
        base_url="https://api.lapathoniia.top/",
        api_key_env="LAPA_API_KEY",
        parameters={"temperature": 0, "seed": 42},
    )
    model = api_runner.identity(chosen)
    payload = api_runner.request_payload(row(), model)
    assert model["endpoint"] == "https://api.lapathoniia.top/chat/completions"
    assert payload["messages"] == [
        {"role": "user", "content": prompts(row())[0]["context"]}
    ]
    assert payload["max_tokens"] == 128
    assert payload["seed"] == 42
    priced = {**chosen, "prices": {"input": 2, "output": 8}}
    assert api_runner.identity(priced) == model
    assert api_runner.estimate([row()], priced)["estimated_cost_usd"] > 0
    other = api_runner.request_payload(
        row(),
        config(output_token_parameter="max_completion_tokens", max_output_tokens=512),  # noqa: S106 -- token limit parameter, not a credential
    )
    assert "max_tokens" not in other
    assert other["max_completion_tokens"] == 512


@pytest.mark.parametrize(
    "invalid",
    [
        {"api_key": "SECRET"},
        {"base_url": "http://insecure.invalid"},
        {"base_url": "https://name:SECRET@example.invalid"},
        {"base_url": "https://example.invalid?key=SECRET"},
        {"base_url": "https://example.invalid/chat/completions"},
        {"api_key_env": "not-a-variable"},
        {"parameters": {"messages": []}},
        {"parameters": {"api_key": "SECRET"}},
        {"parameters": {"nested": {"authorization": "SECRET"}}},
        {"parameters": {"max_tokens": 16}},
        {"parameters": {"response_format": {"type": "json_object"}}},
        {"parameters": {"stream": True}},
        {"parameters": {"n": 2}},
        {"parameters": {"thinkingConfig": {"thinkingLevel": "minimal"}}},
        {"max_output_tokens": 0},
        {"requests_per_minute": 0},
        {"max_retries": 4},
        {"prices": {"input": -1, "output": 2}},
        {"prices": {"input": 1}},
    ],
)
def test_invalid_configuration_rejected_before_artifacts(invalid, tmp_path):
    output = tmp_path / "absent"
    with pytest.raises(ValueError) as error:
        api_runner.run([row()], {}, config(**invalid), output, client=Mock())
    assert "SECRET" not in str(error.value)
    assert not output.exists()


def test_likelihood_task_fails_before_creating_output(tmp_path):
    unsupported = {"id": "one", "task": "bbq_uk", "protocol": "candidate_protocol"}
    output = tmp_path / "absent"
    with pytest.raises(ValueError, match="candidate log-likelihood"):
        api_runner.run([unsupported], {}, config(), output, client=Mock())
    assert not output.exists()


@pytest.mark.parametrize("provider", ["openai-compatible", "gemini"])
def test_run_saves_raw_evidence_resumes_and_does_not_read_keys(
    provider, tmp_path, monkeypatch
):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200, json=raw(provider), headers={"x-request-id": "req-one"}
        )

    monkeypatch.setattr(
        api_runner.os.environ,
        "get",
        lambda *_args: pytest.fail("Injected client should not read secrets"),
    )
    with httpx.Client(
        transport=httpx.MockTransport(handler), trust_env=False
    ) as client:
        predictions, manifest = api_runner.run(
            [row()], {"task": "warbias_uk"}, config(provider), tmp_path, client=client
        )
        raw_path = next((tmp_path / "responses").glob("*.json"))
        original = raw_path.read_bytes()
        again, again_manifest = api_runner.run(
            [row()], {"task": "warbias_uk"}, config(provider), tmp_path, client=client
        )
    assert calls == [predictions[0]["request"]]
    assert again == predictions and again_manifest == manifest
    assert raw_path.read_bytes() == original
    assert not (tmp_path / "inflight.json").exists()
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "complete"
    assert predictions[0]["response"] == "C"
    assert predictions[0]["request_id"] == "req-one"
    assert manifest["model_identity"]["model_snapshot"] == "any-user-selected-model"
    assert predictions[0]["diagnostics"]["returned_model"] == "actual-provider-snapshot"
    api_runner.validate_prediction(row(), predictions[0], manifest["model_identity"])
    if provider == "gemini":
        assert calls[0]["generationConfig"] == {"maxOutputTokens": 128}
        assert calls[0]["contents"] == [
            {"role": "user", "parts": [{"text": prompts(row())[0]["context"]}]}
        ]


def test_known_token_usage_price_and_missing_usage():
    value = config(prices={"input": 2, "output": 8, "cached_input": 1})
    result = api_runner.usage_summary([{"api_response": raw()}], value)
    assert result["estimated_recorded_cost_usd"] == pytest.approx(
        (80 * 2 + 20 + 6 * 8) / 1e6
    )
    assert result["token_counts"]["reasoning_tokens"] == 5
    missing = raw()
    del missing["usage"]
    result = api_runner.usage_summary([{"api_response": missing}], value)
    assert result["estimated_recorded_cost_usd"] is None
    assert result["responses_without_complete_usage"] == 1
    assert result["token_counts"]["input_tokens"] is None
    assert (
        api_runner.usage_summary([{"api_response": raw()}], config())[
            "estimated_recorded_cost_usd"
        ]
        is None
    )
    gemini = api_runner.usage_summary(
        [{"api_response": raw("gemini")}],
        config("gemini", prices={"input": 2, "output": 8}),
    )
    assert gemini["estimated_recorded_cost_usd"] == pytest.approx(
        (100 * 2 + 6 * 8) / 1e6
    )


def test_refusals_truncation_thought_exclusion_and_exact_model_pin():
    response = raw("gemini")
    response["candidates"][0]["content"]["parts"].insert(
        0, {"thought": True, "text": "hidden"}
    )
    response["candidates"][0]["finishReason"] = "MAX_TOKENS"
    text, detail = api_runner.response_details(response, config("gemini"))
    assert text == "C" and detail["truncated"]
    text, detail = api_runner.response_details(
        {"promptFeedback": {"blockReason": "SAFETY"}}, config("gemini")
    )
    assert text == "" and detail["refusal"] and not detail["usage_complete"]
    refusal = raw()
    refusal["choices"][0]["message"] = {"content": None, "refusal": "Cannot answer"}
    assert api_runner.response_details(refusal, config())[1]["refusal"]
    response = raw("gemini")
    response["promptFeedback"] = {"blockReason": "BLOCK_REASON_UNSPECIFIED"}
    text, detail = api_runner.response_details(response, config("gemini"))
    assert text == "C" and detail["prompt_block"] is None and not detail["refusal"]
    with pytest.raises(ValueError, match="expected_response_model"):
        api_runner.response_details(
            raw(), config(expected_response_model="different-snapshot")
        )


def test_corrupted_prediction_or_request_rejected(tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=raw()))
    ) as client:
        predictions, _ = api_runner.run([row()], {}, config(), tmp_path, client=client)
    for field in ("response", "request", "diagnostics"):
        corrupted = copy.deepcopy(predictions[0])
        corrupted[field] = "corrupted"
        with pytest.raises(ValueError):
            api_runner.validate_prediction(row(), corrupted, config())
    with pytest.raises(ValueError, match="different run"):
        api_runner.run(
            [row()], {}, config(max_output_tokens=64), tmp_path, client=Mock()
        )


@pytest.mark.parametrize(
    "status,body,category",
    [
        (403, {"error": {"message": "SECRET"}}, "authentication_or_permission"),
        (429, {"error": {"message": "insufficient_quota SECRET"}}, "quota_or_budget"),
        (
            429,
            {"error": {"details": [{"quotaId": "RequestsPerDay"}]}},
            "quota_or_budget",
        ),
        (400, {"error": {"message": "SECRET"}}, "provider_error"),
    ],
)
def test_terminal_http_errors_no_retry_or_fake_answer(
    status, body, category, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        api_runner.time, "sleep", lambda _: pytest.fail("Terminal error must not retry")
    )

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json=body)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(RuntimeError, match=category) as error,
    ):
        api_runner.run([row()], {}, config(), tmp_path, client=client)
    assert "SECRET" not in str(error.value)
    assert len(calls) == 1
    assert not (tmp_path / "predictions.jsonl").exists()
    assert not (tmp_path / "inflight.json").exists()
    assert not list((tmp_path / "responses").glob("*.json"))
    assert json.loads((tmp_path / "status.json").read_text())["category"] == category


def test_temporary_429_retries_bounded_and_records_one_answer(tmp_path, monkeypatch):
    calls, delays = [], []
    monkeypatch.setattr(api_runner.time, "sleep", delays.append)

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(
                429,
                json={"error": {"message": "temporary"}},
                headers={"retry-after": "7"},
            )
        return httpx.Response(200, json=raw())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        predictions, _ = api_runner.run([row()], {}, config(), tmp_path, client=client)
    assert len(calls) == 3 and len(predictions) == 1
    assert delays.count(7) == 2
    assert len(list((tmp_path / "responses").glob("*.json"))) == 1


def test_retry_delay_above_one_minute_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(
        api_runner.time, "sleep", lambda _: pytest.fail("Must not wait over a minute")
    )
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(429, json={}, headers={"retry-after": "120"})
            )
        ) as client,
        pytest.raises(RuntimeError, match="rate_limit"),
    ):
        api_runner.run([row()], {}, config(), tmp_path, client=client)


def test_uncertain_transport_blocks_resume_and_redacts_exception(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("SECRET", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="unknown request outcome") as error:
            api_runner.run([row()], {}, config(), tmp_path, client=client)
        assert "SECRET" not in str(error.value)
        with pytest.raises(RuntimeError, match="previous request"):
            api_runner.run([row()], {}, config(), tmp_path, client=client)
    assert len(calls) == 1
    assert (tmp_path / "inflight.json").exists()
    assert not (tmp_path / "predictions.jsonl").exists()


def test_server_error_blocks_uncertain_resume(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"error": {"message": "SECRET"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="uncertain_provider_error"):
            api_runner.run([row()], {}, config(), tmp_path, client=client)
        with pytest.raises(RuntimeError, match="previous request"):
            api_runner.run([row()], {}, config(), tmp_path, client=client)
    assert len(calls) == 1
    assert (tmp_path / "inflight.json").exists()


def test_resume_reconstructs_export_and_detects_changed_gold(tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=raw()))
    ) as client:
        predictions, _ = api_runner.run([row()], {}, config(), tmp_path, client=client)
    (tmp_path / "predictions.jsonl").unlink()
    api_runner.run([row()], {}, config(), tmp_path, client=Mock())
    assert json.loads((tmp_path / "predictions.jsonl").read_text()) == predictions[0]
    changed = row()
    changed["gold"] = 0
    with pytest.raises(ValueError, match="different run"):
        api_runner.run([changed], {}, config(), tmp_path, client=Mock())


def test_malformed_success_keeps_raw_evidence_and_never_repeats(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"unexpected": "raw provider data"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="raw checkpoint was saved"):
            api_runner.run([row()], {}, config(), tmp_path, client=client)
        with pytest.raises(ValueError, match="exactly one"):
            api_runner.run([row()], {}, config(), tmp_path, client=client)
    assert len(calls) == 1
    checkpoint = json.loads(next((tmp_path / "responses").glob("*.json")).read_text())
    assert checkpoint["api_response"] == {"unexpected": "raw provider data"}


@pytest.mark.parametrize(
    "provider,body",
    [
        ("openai-compatible", {"choices": [{}]}),
        ("openai-compatible", {"choices": [{"finish_reason": "stop"}]}),
        ("openai-compatible", {"choices": [{"message": {"content": "C"}}]}),
        (
            "openai-compatible",
            {"choices": [{"message": {}, "finish_reason": "stop"}]},
        ),
        *[
            (
                "openai-compatible",
                {"choices": [{"message": {"content": value}, "finish_reason": "stop"}]},
            )
            for value in (None, False, 0, [])
        ],
        ("gemini", {"candidates": [{}]}),
        ("gemini", {"promptFeedback": {"blockReason": "BLOCK_REASON_UNSPECIFIED"}}),
        ("gemini", {"candidates": [{"finishReason": "STOP"}]}),
        ("gemini", {"candidates": [{"content": {"parts": [{"text": "C"}]}}]}),
        *[
            (
                "gemini",
                {"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}]},
            )
            for parts in ([], [{}], [{"text": None}], [{"text": False}], "C")
        ],
    ],
)
def test_malformed_candidate_stops_without_scoring_or_repeating(
    provider, body, tmp_path
):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="raw checkpoint was saved"):
            api_runner.run([row()], {}, config(provider), tmp_path, client=client)
        raw_path = next((tmp_path / "responses").glob("*.json"))
        checkpoint = raw_path.read_bytes()
        with pytest.raises(ValueError):
            api_runner.run([row()], {}, config(provider), tmp_path, client=client)
    assert len(calls) == 1
    assert raw_path.read_bytes() == checkpoint
    assert json.loads(checkpoint)["api_response"] == body
    assert not (tmp_path / "predictions.jsonl").exists()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "invalid_provider_response"
    assert status["completed"] == 0


@pytest.mark.parametrize(
    "provider,body,refusal,truncated",
    [
        (
            "openai-compatible",
            {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
            False,
            False,
        ),
        (
            "openai-compatible",
            {
                "choices": [
                    {"message": {"refusal": "Cannot answer"}, "finish_reason": "stop"}
                ]
            },
            True,
            False,
        ),
        *[
            (
                "openai-compatible",
                {"choices": [{"message": {"content": None}, "finish_reason": finish}]},
                finish == "content_filter",
                finish == "length",
            )
            for finish in ("content_filter", "length")
        ],
        (
            "gemini",
            {
                "candidates": [
                    {"content": {"parts": [{"text": ""}]}, "finishReason": "STOP"}
                ]
            },
            False,
            False,
        ),
        *[
            (
                "gemini",
                {"candidates": [{"finishReason": finish}]},
                finish == "SAFETY",
                finish == "MAX_TOKENS",
            )
            for finish in ("SAFETY", "MAX_TOKENS")
        ],
        ("gemini", {"promptFeedback": {"blockReason": "SAFETY"}}, True, False),
    ],
)
def test_explicit_empty_answers_refusals_and_truncations_remain_scored(
    provider, body, refusal, truncated, tmp_path
):
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as client:
        predictions, _ = api_runner.run(
            [row()], {}, config(provider), tmp_path, client=client
        )
    prediction = predictions[0]
    assert prediction["response"] == ""
    assert prediction["diagnostics"]["refusal"] is refusal
    assert prediction["diagnostics"]["truncated"] is truncated
    assert api_runner.score_item(row(), prediction)["invalid"] == 1
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "complete"


def test_completed_records_survive_later_failure_and_stop_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(api_runner.time, "sleep", lambda _: None)
    calls = []

    def handler(request):
        calls.append(request)
        return (
            httpx.Response(200, json=raw())
            if len(calls) == 1
            else httpx.Response(403, json={})
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="HTTP 403"):
            api_runner.run([row(), row("two")], {}, config(), tmp_path, client=client)
        assert len((tmp_path / "predictions.jsonl").read_text().splitlines()) == 1
        (tmp_path / "STOP").touch()
        with pytest.raises(RuntimeError, match="STOP"):
            api_runner.run([row(), row("two")], {}, config(), tmp_path, client=client)
    assert len(calls) == 2


def test_stop_during_rate_wait_prevents_the_next_request(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=raw())

    monkeypatch.setattr(api_runner.time, "sleep", lambda _: (tmp_path / "STOP").touch())
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(RuntimeError, match="STOP"),
    ):
        api_runner.run([row(), row("two")], {}, config(), tmp_path, client=client)
    assert len(calls) == 1
    assert not (tmp_path / "inflight.json").exists()


@pytest.mark.parametrize(
    "provider,header",
    [("openai-compatible", "Authorization"), ("gemini", "x-goog-api-key")],
)
def test_owned_client_reads_only_named_secret_at_execution(
    provider, header, tmp_path, monkeypatch
):
    calls = []
    real_client = httpx.Client
    monkeypatch.setenv("USER_SELECTED_KEY", "SECRET")

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=raw(provider))

    def make_client(**kwargs):
        assert kwargs["follow_redirects"] is False
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", make_client)
    api_runner.run(
        [row()], {}, config(provider, api_key_env="USER_SELECTED_KEY"), tmp_path
    )
    assert calls[0].headers[header] == (
        "SECRET" if provider == "gemini" else "Bearer SECRET"
    )
    for path in tmp_path.rglob("*.json*"):
        assert "SECRET" not in path.read_text()
