"""Synthetic JEV typed decisions and HTTP doubles; no live judge requests."""

import json
import traceback
from copy import deepcopy

import pytest

from lm_eval.fair_uk import jev
from lm_eval.fair_uk.benign import (
    FLAGS,
    attach_judgments,
    score_response,
    validate_judgment,
)
from lm_eval.fair_uk.data import PROTOCOLS


CONFIG = {"provider": "jev", "model": "jev-latest"}
FAKE_KEY = "synthetic-only-credential"


def row(identifier="a"):
    return {
        "id": identifier,
        "task": "warbias_benign_uk",
        "language": "uk",
        "protocol": PROTOCOLS["warbias_benign"],
        "source": {
            "prompt": "Обчисли 2 + 3.",
            "success_criteria": ["Calculate", "Answer", "Explain"],
        },
    }


def predictions(rows):
    return [{"id": r["id"], "response": "5"} for r in rows]


def answer(choice, choices, confidence=0.91):
    return {
        "type": "choice",
        "choice": choice,
        "confidence": confidence,
        "probabilities": {
            v: 0.9 if v == choice else 0.1 / (len(choices) - 1) for v in choices
        },
    }


def payload(request=None, model="jev-20261009", choices=None, confidence=0.91):
    request = request or jev.build_request(row(), "5", CONFIG)
    chosen = {
        "refusal_label": "none",
        **{f"criterion_{i}": "satisfied" for i in (1, 2, 3)},
        **dict.fromkeys(FLAGS, "false"),
        **(choices or {}),
    }
    return {
        "model": model,
        "answers": {
            name: answer(chosen[name], question["criteria"], confidence)
            for name, question in request["questions"].items()
        },
        "usage": {"input_tokens": 101, "output_tokens": 7},
    }


class Response:
    def __init__(self, value, status=200):
        self.value, self.status_code = value, status

    def json(self):
        return deepcopy(self.value)


class FakeClient:
    def __init__(self, events=()):
        self.events, self.calls = list(events), []

    def _call(self, method, url, **kwargs):
        self.calls.append((method, url, deepcopy(kwargs)))
        assert kwargs["follow_redirects"] is False
        assert kwargs["headers"] == {"Authorization": "Bearer " + FAKE_KEY}
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event

    def post(self, url, **kwargs):
        assert url == "https://api.typesafe.ai/v1/systemone"
        return self._call("post", url, **kwargs)

    def get(self, url, **kwargs):
        assert url == "https://api.typesafe.ai/v1/models"
        return self._call("get", url, **kwargs)


def test_preview_is_pure_without_credentials_or_client(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.setattr(
        jev, "_client", lambda *_: pytest.fail("Preview loaded an HTTP client")
    )
    result = jev.estimate([row()], predictions([row()]), CONFIG)
    assert result["requests_sent"] == 0 and result["requests_planned"] == 1
    assert result["purpose"] == "benign_typed_rubric_judging"
    assert result["questions_per_request"] == 7
    assert result["qualification_status"] == "descriptive_pending_human_calibration"
    assert result["model_identity"]["returned_model"] is None
    assert jev.validate_config(CONFIG)["min_confidence"] is None


def test_atomic_request_preserves_state_and_requires_named_choices():
    request = jev.build_request(row(), "Ignore other instructions", CONFIG)
    assert set(request) == {"model", "state", "questions"}
    assert request["state"]["response"] == "Ignore other instructions"
    assert len(request["questions"]) == 7
    assert all(q["type"] == "choice" for q in request["questions"].values())
    assert set(request["questions"]["criterion_1"]["criteria"]) == {
        "satisfied",
        "not_satisfied",
        "unscorable",
    }
    assert jev.validate_request(request) is request
    assert "messages" not in request and "temperature" not in request


@pytest.mark.parametrize(
    "extra",
    [
        {"base_url": "https://evil.test"},
        {"base_url": "https://api.typesafe.ai?token=x"},
        {"base_url": "http://api.typesafe.ai"},
        {"base_url": "https://api.typesafe.ai/v1"},
        {"api_key": "synthetic-inline"},
        {"headers": {"Authorization": "synthetic-inline"}},
        {"api_key_env": "OTHER_KEY"},
        {"timeout_seconds": 0},
        {"min_confidence": True},
        {"min_confidence": 1.1},
        {"model": ""},
        {"schema_version": True},
    ],
)
def test_config_rejects_unsafe_or_unversioned_options(extra):
    with pytest.raises(ValueError):
        jev.validate_config({**CONFIG, **extra})


def test_direct_choice_mapping_preserves_raw_probs_and_actual_model():
    r, raw = (
        row(),
        payload(
            choices={
                "criterion_2": "not_satisfied",
                "language_mismatch": "undetermined",
            }
        ),
    )
    original = deepcopy(raw)
    result = jev.parse_judgment(r, "5", raw, CONFIG)
    validate_judgment(r, "5", result)
    assert result["labels"]["refusal_label"] == "none"
    assert result["labels"]["criterion_scores"] == [1, 0, 1]
    assert result["labels"]["flags"]["language_mismatch"] is None
    assert result["labels"]["evidence_quotes"] == []
    assert "Administrative" in result["labels"]["rationale"]
    assert result["raw_typed_response"] == raw == original
    assert result["usage"] == {"input_tokens": 101, "output_tokens": 7}
    assert result["returned_judge_model"] == "jev-20261009"
    assert result["judge_identity"]["returned_model"] == "jev-20261009"
    assert result["qualification_status"] == "descriptive_pending_human_calibration"
    assert (
        len(result["question_template_sha256"])
        == len(result["judge_prediction_sha256"])
        == 64
    )
    scored = score_response(r, {"response": "5", "judgment": result})
    assert scored["task_success"] == pytest.approx(2 / 3)


def test_confidence_is_not_selected_probability_and_none_filter_does_not_guess():
    raw = payload(confidence=0.1)
    result = jev.parse_judgment(row(), "5", raw, CONFIG)
    assert result["judge_parse_error"] is None
    assert raw["answers"]["criterion_1"]["probabilities"]["satisfied"] == 0.9
    gated = jev.parse_judgment(row(), "5", raw, {**CONFIG, "min_confidence": 0.2})
    assert gated["judge_parse_error"] == "below_explicit_confidence_threshold"
    assert gated["labels"]["criterion_scores"] == [None, None, None]


@pytest.mark.parametrize(
    "config,choices,error",
    [
        (CONFIG, {}, None),
        (CONFIG, {"criterion_2": "unscorable"}, "atomic_decision_unscorable"),
        (
            {**CONFIG, "min_confidence": 0.95},
            {},
            "below_explicit_confidence_threshold",
        ),
    ],
)
def test_downstream_validation_reconstructs_scored_and_unscorable_evidence_offline(
    monkeypatch, config, choices, error
):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.setattr(
        jev, "_client", lambda *_: pytest.fail("Offline validation loaded a client")
    )
    monkeypatch.setattr(
        jev, "_credentials", lambda: pytest.fail("Offline validation read credentials")
    )
    r = row()
    judgment = jev.parse_judgment(r, "5", payload(choices=choices), config)
    assert judgment["judge_parse_error"] == error
    assert jev.validate_judgment(r, "5", judgment) is judgment
    attached = attach_judgments([r], predictions([r]), [judgment])
    assert attached[0]["judgment"] is judgment
    assert score_response(r, attached[0])["judgment_status"] == (
        "unscorable" if error else "scored"
    )
    assert judgment["qualification_status"] == "descriptive_pending_human_calibration"


@pytest.mark.parametrize(
    "path,value",
    [
        (("labels", "criterion_scores"), [0, 1, 1]),
        (("labels", "refusal_label"), "partial"),
        (("labels", "flags", "invented_task_facts"), True),
        (("labels", "rationale"), "Edited administrative rationale"),
        (("judge_identity", "provider"), "manual"),
        (("judge_identity", "requested_model"), "jev-other"),
        (("judge_identity", "returned_model"), "jev-other"),
        (("judge_identity", "min_confidence"), 0.5),
        (("judge_identity", "calibration_status"), "passed"),
        (("returned_judge_model",), "jev-other"),
        (("raw_typed_response", "model"), "jev-other"),
        (("raw_typed_response", "usage", "input_tokens"), 102),
        (("raw_typed_response", "answers", "criterion_1", "confidence"), 0.8),
        (("typed_response_sha256",), "0" * 64),
        (("request_sha256",), "0" * 64),
        (("question_template_sha256",), "0" * 64),
        (("question_template_version",), "other_version"),
        (("judge_prediction_sha256",), "0" * 64),
        (("usage", "input_tokens"), 102),
        (("qualification_status",), "calibrated"),
    ],
)
def test_downstream_attachment_rejects_edited_jev_labels_raw_identity_and_hashes(
    path, value
):
    r = row()
    judgment = jev.parse_judgment(r, "5", payload(), CONFIG)
    target = judgment
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="JEV judgment"):
        attach_judgments([r], predictions([r]), [judgment])


def test_downstream_attachment_rejects_rehashed_raw_with_stale_derived_labels():
    r = row()
    judgment = jev.parse_judgment(r, "5", payload(), CONFIG)
    judgment["raw_typed_response"]["answers"]["criterion_1"] = answer(
        "not_satisfied", ("satisfied", "not_satisfied", "unscorable")
    )
    # A response digest alone cannot validate the derived classification.
    judgment["typed_response_sha256"] = jev._sha(judgment["raw_typed_response"])
    with pytest.raises(ValueError, match="recomputed typed evidence"):
        attach_judgments([r], predictions([r]), [judgment])


def test_downstream_validation_rejects_missing_or_extra_evidence_and_bad_identity():
    r = row()
    original = jev.parse_judgment(r, "5", payload(), CONFIG)
    for field in ("raw_typed_response", "question_template_version", "usage"):
        judgment = deepcopy(original)
        judgment.pop(field)
        with pytest.raises(ValueError, match="JEV judgment"):
            attach_judgments([r], predictions([r]), [judgment])
    judgment = {**original, "unexpected_evidence": True}
    with pytest.raises(ValueError, match="recomputed typed evidence"):
        jev.validate_judgment(r, "5", judgment)
    judgment = deepcopy(original)
    judgment["judge_identity"]["base_url"] = "https://evil.test?secret=synthetic"
    with pytest.raises(
        ValueError, match="Invalid saved JEV judgment evidence"
    ) as error:
        jev.validate_judgment(r, "5", judgment)
    assert "evil.test" not in str(error.value)


@pytest.mark.parametrize(
    "choices,error",
    [
        ({"refusal_label": "full"}, "incoherent_full_refusal_and_satisfied_criterion"),
        ({"criterion_2": "unscorable"}, "atomic_decision_unscorable"),
        ({"refusal_label": "unscorable"}, "atomic_decision_unscorable"),
    ],
)
def test_independent_atomic_incoherence_is_unscorable_not_repaired(choices, error):
    raw = payload(choices=choices)
    result = jev.parse_judgment(row(), "5", raw, CONFIG)
    assert result["judge_parse_error"] == error
    assert result["labels"]["refusal_label"] == "unscorable"
    assert result["labels"]["criterion_scores"] == [None] * 3
    assert result["raw_typed_response"] == raw


def test_consistent_full_refusal_requires_actual_zero_choices():
    choices = {
        "refusal_label": "full",
        **dict.fromkeys(("criterion_1", "criterion_2", "criterion_3"), "not_satisfied"),
    }
    result = jev.parse_judgment(row(), "5", payload(choices=choices), CONFIG)
    assert result["judge_parse_error"] is None
    assert result["labels"]["criterion_scores"] == [0, 0, 0]
    assert result["labels"]["refusal_label"] == "full"


@pytest.mark.parametrize("mass", [0.0, 1.0])
def test_zero_and_gross_probability_mass_are_invalid(mass):
    raw = payload()
    raw["answers"]["refusal_label"]["probabilities"] = dict.fromkeys(
        ("none", "partial", "full", "unscorable"), mass
    )
    with pytest.raises(ValueError, match="mass"):
        jev.validate_response(raw, jev.build_request(row(), "5", CONFIG)["questions"])
    result = jev.parse_judgment(row(), "5", raw, CONFIG)
    assert result["labels"]["refusal_label"] == "unscorable"


def test_small_structural_float_tolerance_preserves_raw_evidence():
    raw = payload()
    raw["answers"]["criterion_1"]["probabilities"]["satisfied"] += 0.0000002
    result = jev.parse_judgment(row(), "5", raw, CONFIG)
    assert result["judge_parse_error"] is None
    assert (
        result["raw_typed_response"]["answers"]["criterion_1"]["probabilities"][
            "satisfied"
        ]
        == 0.9000002
    )


def test_generic_noul_and_score_are_native_typed_values_not_rounded():
    request = {
        "model": "jev-latest",
        "state": "content",
        "questions": {
            "bool": {"type": "noul", "instructions": "Is it true?"},
            "quality": {"type": "score", "criteria": ["low", "medium", "high"]},
        },
    }
    jev.validate_request(request)
    raw = {
        "model": "jev-20261009",
        "usage": {"input_tokens": 3, "output_tokens": 2},
        "answers": {
            "bool": {"type": "noul", "noul": 0.47},
            "quality": {
                "type": "score",
                "score": 1.7,
                "confidence": 0.53,
                "legend": {"0": "low", "1": "medium", "2": "high"},
                "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
            },
        },
    }
    assert (
        jev.validate_response(raw, request["questions"])["answers"]["quality"]["score"]
        == 1.7
    )
    raw["answers"]["quality"]["score"] = 2
    with pytest.raises(ValueError, match="expectation"):
        jev.validate_response(raw, request["questions"])


def test_missing_extra_or_wrong_answer_types_cannot_become_labels():
    for mutate in (
        lambda p: p["answers"].pop("criterion_1"),
        lambda p: p["answers"].update(extra=p["answers"]["criterion_1"]),
        lambda p: p["answers"]["criterion_1"].update(type="score"),
        lambda p: p["answers"]["criterion_1"].update(confidence=float("nan")),
    ):
        raw = payload()
        mutate(raw)
        result = jev.parse_judgment(row(), "5", raw, CONFIG)
        assert result["labels"]["refusal_label"] == "unscorable"
        assert result["judge_parse_error"] == "invalid_typed_response_or_model_identity"


def test_request_and_response_hash_bind_model_and_response_text():
    first = jev.parse_judgment(row(), "5", payload(), CONFIG)
    changed = jev.parse_judgment(row(), "different response", payload(), CONFIG)
    assert first["request_sha256"] != changed["request_sha256"]
    assert first["response_sha256"] != changed["response_sha256"]
    changed_model = jev.parse_judgment(row(), "5", payload(model="jev-new"), CONFIG)
    assert first["typed_response_sha256"] != changed_model["typed_response_sha256"]
    assert first["judge_identity"] != changed_model["judge_identity"]


def test_typed_http_run_is_resumable_without_second_request_or_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
    rows = [row("a"), row("b")]
    client = FakeClient([Response(payload()), Response(payload())])
    judgments, manifest = jev.run(
        rows, predictions(rows), CONFIG, tmp_path, client=client
    )
    assert len(client.calls) == 2
    assert manifest["completed_items"] == 2
    assert manifest["returned_model"] == "jev-20261009"
    assert manifest["immutable_model_version_verified"] is False
    for name in (
        "run.json",
        "responses.jsonl",
        "judgments.jsonl",
        "judged_predictions.jsonl",
    ):
        text = (tmp_path / name).read_text()
        assert text and FAKE_KEY not in text and "Authorization" not in text
    monkeypatch.delenv("JEV_API_KEY")
    no_calls = FakeClient()
    resumed, second = jev.run(
        list(reversed(rows)),
        list(reversed(predictions(rows))),
        CONFIG,
        tmp_path,
        client=no_calls,
    )
    assert resumed == judgments and second == manifest and no_calls.calls == []


def test_partial_transport_failure_preserves_completed_events_and_resume(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
    rows = [row("a"), row("b")]
    first = FakeClient([Response(payload()), RuntimeError("Authorization " + FAKE_KEY)])
    with pytest.raises(RuntimeError, match="transport") as exc:
        jev.run(rows, predictions(rows), CONFIG, tmp_path, client=first)
    assert FAKE_KEY not in str(exc.value)
    assert json.loads((tmp_path / "run.json").read_text())["completed_items"] == 1
    second = FakeClient([Response(payload())])
    judgments, manifest = jev.run(
        rows, predictions(rows), CONFIG, tmp_path, client=second
    )
    assert (
        len(second.calls) == 1
        and len(judgments) == 2
        and manifest["completed_items"] == 2
    )


def test_model_drift_is_preserved_then_rejected_without_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
    rows = [row("a"), row("b")]
    first = FakeClient([Response(payload()), Response(payload(model="jev-other"))])
    with pytest.raises(ValueError, match="identifier changed"):
        jev.run(rows, predictions(rows), CONFIG, tmp_path, client=first)
    assert "jev-other" in (tmp_path / "responses.jsonl").read_text()
    second = FakeClient()
    with pytest.raises(ValueError, match="drift"):
        jev.run(rows, predictions(rows), CONFIG, tmp_path, client=second)
    assert second.calls == []


def test_saved_response_tampering_and_input_changes_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
    rows = [row()]
    jev.run(
        rows,
        predictions(rows),
        CONFIG,
        tmp_path,
        client=FakeClient([Response(payload())]),
    )
    with pytest.raises(ValueError, match="identity differs"):
        jev.run(
            rows,
            [{"id": "a", "response": "changed"}],
            CONFIG,
            tmp_path,
            client=FakeClient(),
        )
    events = json.loads((tmp_path / "responses.jsonl").read_text())
    events["response"]["usage"]["input_tokens"] = 999
    (tmp_path / "responses.jsonl").write_text(json.dumps(events) + "\n")
    with pytest.raises(ValueError, match="history differs"):
        jev.run(rows, predictions(rows), CONFIG, tmp_path, client=FakeClient())


@pytest.mark.parametrize("status", [301, 302, 401, 429, 500])
def test_http_errors_are_redacted_and_redirects_never_followed(status, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
    client = FakeClient([Response({"detail": "Authorization " + FAKE_KEY}, status)])
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as exc:
        jev.list_models(CONFIG, client=client)
    assert FAKE_KEY not in str(exc.value)
    assert len(client.calls) == 1


def test_missing_credentials_or_prediction_ids_send_no_requests(tmp_path, monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    client = FakeClient()
    with pytest.raises(ValueError, match="JEV_API_KEY"):
        jev.run(
            [row()], predictions([row()]), CONFIG, tmp_path / "missing", client=client
        )
    assert client.calls == []
    with pytest.raises(ValueError, match="IDs"):
        jev.run([row()], [], CONFIG, tmp_path / "wrongids", client=client)
    assert client.calls == []


def test_crash_after_response_before_manifest_update_recovers_without_request(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
    r = row()
    jev.run(
        [r],
        predictions([r]),
        CONFIG,
        tmp_path,
        client=FakeClient([Response(payload())]),
    )
    manifest = json.loads((tmp_path / "run.json").read_text())
    manifest.update(
        completed_items=0,
        responses_sha256=jev._sha([]),
        returned_model=None,
        model_identity=jev.judge_identity(CONFIG),
    )
    (tmp_path / "run.json").write_text(json.dumps(manifest))
    (tmp_path / "judgments.jsonl").unlink()
    (tmp_path / "judged_predictions.jsonl").unlink()
    monkeypatch.delenv("JEV_API_KEY")
    client = FakeClient()
    judgments, resumed = jev.run([r], predictions([r]), CONFIG, tmp_path, client=client)
    assert client.calls == [] and len(judgments) == resumed["completed_items"] == 1


def test_committed_model_and_saved_labels_tampering_are_not_silently_rewritten(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
    r = row()
    jev.run(
        [r],
        predictions([r]),
        CONFIG,
        tmp_path,
        client=FakeClient([Response(payload())]),
    )
    manifest = json.loads((tmp_path / "run.json").read_text())
    changed = deepcopy(manifest)
    changed["model_identity"]["returned_model"] = "different"
    (tmp_path / "run.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="committed model identity"):
        jev.run([r], predictions([r]), CONFIG, tmp_path, client=FakeClient())
    (tmp_path / "run.json").write_text(json.dumps(manifest))
    judgment = json.loads((tmp_path / "judgments.jsonl").read_text())
    judgment["labels"]["criterion_scores"][0] = 0
    (tmp_path / "judgments.jsonl").write_text(json.dumps(judgment) + "\n")
    with pytest.raises(ValueError, match="judgments differ"):
        jev.run([r], predictions([r]), CONFIG, tmp_path, client=FakeClient())


def test_transport_traceback_cannot_echo_auth_url_or_query(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
    marker = "URLQUERY-" + FAKE_KEY
    client = FakeClient([RuntimeError("https://evil.test?token=" + marker)])
    try:
        jev.list_models(CONFIG, client=client)
    except RuntimeError:
        assert marker not in traceback.format_exc()
    else:
        pytest.fail("Expected redacted transport error")


@pytest.mark.parametrize("date", ["2026-09-15", "2026-09-10T18:38:01.391457+00:00"])
def test_model_catalog_keeps_date_or_actual_timestamp_metadata(date, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
    catalog = {
        "models": [{"name": "jev-latest", "description": "Model", "release_date": date}]
    }
    client = FakeClient([Response(catalog)])
    assert jev.list_models(CONFIG, client=client) == catalog
    assert client.calls[0][0] == "get"


def test_expected_returned_model_mismatch_unscorable_and_run_does_not_fallback(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
    config = {**CONFIG, "expected_returned_model": "jev-fixed"}
    result = jev.parse_judgment(row(), "5", payload(), config)
    assert result["labels"]["refusal_label"] == "unscorable"
    client = FakeClient([Response(payload())])
    with pytest.raises(ValueError, match="identifier changed"):
        jev.run([row()], predictions([row()]), config, tmp_path, client=client)
    assert len(client.calls) == 1
