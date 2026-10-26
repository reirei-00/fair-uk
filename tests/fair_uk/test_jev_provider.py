"""The public typed provider boundary is reusable independently of benign rubrics."""

from copy import deepcopy

import pytest

from lm_eval.fair_uk import jev


CONFIG = {"model": "jev-latest"}
REQUEST = {
    "model": "jev-latest",
    "state": {"quoted_evidence": "A synthetic case"},
    "questions": {
        "label": {
            "type": "choice",
            "instructions": "Classify the quoted evidence",
            "criteria": {"present": "Present", "absent": "Absent"},
        }
    },
}
PAYLOAD = {
    "model": "jev-fixture",
    "usage": {"input_tokens": 3, "output_tokens": 1},
    "answers": {
        "label": {
            "type": "choice",
            "choice": "present",
            "confidence": 1.0,
            "probabilities": {"present": 1.0, "absent": 0.0},
        }
    },
}


class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class Client:
    def __init__(self, payload=None, *, fail=False):
        self.payload = deepcopy(PAYLOAD if payload is None else payload)
        self.calls, self.closed, self.fail = [], False, fail

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.fail:
            raise RuntimeError("untrusted transport details")
        return Response(self.payload)

    def close(self):
        self.closed = True


def test_create_client_normalizes_config_without_reading_credentials(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    client = Client()
    configs = []

    def factory(config):
        configs.append(config)
        return client

    monkeypatch.setattr(jev, "_client", factory)
    assert jev.create_client(CONFIG) is client
    assert configs == [jev.validate_config(CONFIG)]
    assert client.calls == [] and not client.closed


def test_submit_uses_official_endpoint_and_preserves_raw_evidence(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "fixture")
    client = Client()
    before = deepcopy(REQUEST)
    result = jev.submit_systemone(REQUEST, CONFIG, client=client)
    assert result is client.payload and result == PAYLOAD
    assert before == REQUEST and not client.closed
    url, arguments = client.calls[0]
    assert url == jev.BASE_URL + jev.SYSTEMONE_PATH
    assert arguments["json"] is REQUEST
    assert arguments["follow_redirects"] is False
    assert arguments["headers"] == {"Authorization": "Bearer fixture"}
    assert jev.validate_returned_model(result, CONFIG) == "jev-fixture"


@pytest.mark.parametrize("fail", [False, True])
def test_submit_closes_only_owned_clients(monkeypatch, fail):
    monkeypatch.setenv("JEV_API_KEY", "fixture")
    client = Client(fail=fail)
    monkeypatch.setattr(jev, "create_client", lambda _config: client)
    if fail:
        with pytest.raises(RuntimeError, match="no raw transport details"):
            jev.submit_systemone(REQUEST, CONFIG)
    else:
        assert jev.submit_systemone(REQUEST, CONFIG) is client.payload
    assert client.closed


def test_invalid_or_mismatched_request_never_sends(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    client = Client()
    with pytest.raises(ValueError, match="configured model"):
        jev.submit_systemone({**REQUEST, "model": "different"}, CONFIG, client=client)
    with pytest.raises(ValueError, match="named questions"):
        jev.submit_systemone({**REQUEST, "questions": {}}, CONFIG, client=client)
    assert client.calls == [] and not client.closed


def test_model_validation_is_offline_and_checks_expectation_and_drift(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    before = deepcopy(PAYLOAD)
    config = {**CONFIG, "expected_returned_model": "jev-fixture"}
    assert (
        jev.validate_returned_model(PAYLOAD, config, previous_model="jev-fixture")
        == "jev-fixture"
    )
    with pytest.raises(ValueError, match="frozen expectation"):
        jev.validate_returned_model(
            PAYLOAD, {**config, "expected_returned_model": "other"}
        )
    with pytest.raises(ValueError, match="identifier changed"):
        jev.validate_returned_model(PAYLOAD, CONFIG, previous_model="other")
    assert before == PAYLOAD


def test_submit_leaves_model_and_rubric_policy_to_evidence_consumer(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "fixture")
    malformed = deepcopy(PAYLOAD)
    malformed["answers"]["label"]["choice"] = "outside-codebook"
    client = Client(malformed)
    result = jev.submit_systemone(
        REQUEST, {**CONFIG, "expected_returned_model": "required-model"}, client=client
    )
    # The caller can persist this paid response before deciding whether model
    # drift stops the run or a typed answer becomes unscorable.
    assert result is client.payload
    with pytest.raises(ValueError, match="frozen expectation"):
        jev.validate_returned_model(
            result, {**CONFIG, "expected_returned_model": "required-model"}
        )
    with pytest.raises(ValueError):
        jev.validate_response(result, REQUEST["questions"])
