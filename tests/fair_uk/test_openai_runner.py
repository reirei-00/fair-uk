import copy
import json
from types import SimpleNamespace

import pytest

from lm_eval.fair_uk.data import adapt
from lm_eval.fair_uk.openai_runner import estimate, identity, run, usage_summary


SNAPSHOT = "gpt-4.1-mini-2025-04-14"


def rows():
    return [
        adapt(
            "warbias_uk",
            {
                "id": str(i),
                "scenario_id": str(i),
                "source_case_id": str(i),
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
        for i in range(2)
    ]


class FakeClient:
    def __init__(self, fail_call=None, returned_model=SNAPSHOT):
        self.calls = []
        self.fail_call = fail_call
        self.returned_model = returned_model
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **payload):
        self.calls.append(payload)
        if len(self.calls) == self.fail_call:
            raise RuntimeError("sensitive provider error must not appear")
        response = {
            "model": self.returned_model,
            "system_fingerprint": "fp-test",
            "choices": [
                {"finish_reason": "stop", "message": {"content": "C", "refusal": None}}
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 50},
            },
        }
        return SimpleNamespace(model_dump=lambda **_: response, _request_id="req-test")


def test_checkpoint_survives_failure_and_resume_does_not_repeat(tmp_path):
    client = FakeClient(fail_call=2)
    with pytest.raises(RuntimeError, match="OpenAI request failed") as error:
        run(rows(), {}, SNAPSHOT, 42, tmp_path, concurrency=1, client=client)
    assert "sensitive" not in str(error.value)
    checkpoint = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text().splitlines()
    ]
    assert [p["id"] for p in checkpoint] == ["0"]
    resumed = FakeClient()
    predictions, manifest = run(
        rows(), {}, SNAPSHOT, 42, tmp_path, concurrency=1, client=resumed
    )
    assert len(resumed.calls) == 1 and len(predictions) == 2
    assert manifest["model_identity"]["provider"] == "openai"
    assert usage_summary(predictions, SNAPSHOT)[
        "estimated_recorded_cost_usd"
    ] == pytest.approx(0.0000532)
    run(rows(), {}, SNAPSHOT, 42, tmp_path, client=resumed)
    assert len(resumed.calls) == 1
    payload = resumed.calls[0]
    assert [m["role"] for m in payload["messages"]] == ["user"]
    assert payload["store"] is False and payload["temperature"] == 0
    assert payload["max_completion_tokens"] == 16


def test_changed_settings_response_and_snapshot_fail_closed(tmp_path):
    predictions, _ = run(rows(), {}, SNAPSHOT, 42, tmp_path, client=FakeClient())
    with pytest.raises(ValueError, match="different run"):
        run(rows(), {}, SNAPSHOT, 43, tmp_path, client=FakeClient())
    changed = copy.deepcopy(predictions)
    changed[0]["response"] = "A"
    (tmp_path / "predictions.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in changed)
    )
    with pytest.raises(ValueError, match="raw API response"):
        run(rows(), {}, SNAPSHOT, 42, tmp_path, client=FakeClient())
    with pytest.raises(ValueError, match="dated"):
        identity("gpt-4.1-mini", 42)
    with pytest.raises(RuntimeError, match="OpenAI request failed"):
        run(
            rows(),
            {},
            SNAPSHOT,
            42,
            tmp_path / "mismatch",
            client=FakeClient(returned_model="different"),
        )


def test_budget_and_credentials_fail_before_requests(tmp_path, monkeypatch):
    client = FakeClient()
    with pytest.raises(ValueError, match="preflight"):
        run(rows(), {}, SNAPSHOT, 42, tmp_path, max_estimated_usd=1e-10, client=client)
    assert client.calls == []
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Configure OPENAI_API_KEY"):
        run(rows(), {}, SNAPSHOT, 42, tmp_path)
    assert not (tmp_path / "run.json").exists()
    assert estimate(rows(), SNAPSHOT)["requests"] == 2
