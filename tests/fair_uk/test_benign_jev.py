"""JEV judging CLI integration uses fixtures and no live decision requests."""

import json

import pytest

from lm_eval.fair_uk import benign, credentials, jev
from lm_eval.fair_uk.data import PROTOCOLS


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    row = {
        "id": "synthetic-1",
        "task": "warbias_benign_uk",
        "language": "uk",
        "protocol": PROTOCOLS["warbias_benign"],
        "cluster": "case-1",
        "stratum": "synthetic",
        "condition": "benign",
        "group": "fixture",
        "source": {
            "prompt": "Compute 2 + 3.",
            "success_criteria": ["Give a number", "Give 5", "Answer briefly"],
        },
    }
    monkeypatch.setattr(benign, "load", lambda *args, **kwargs: [row])
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"id": row["id"], "response": "5"}) + "\n")
    config = tmp_path / "judge.json"
    config.write_text(
        json.dumps(
            {
                "provider": "jev",
                "model": "jev-latest",
                "keychain": {
                    "service": "fixture-service",
                    "account": "fixture-account",
                },
            }
        )
    )
    args = [
        "--task",
        row["task"],
        "--dataset-bundle",
        str(tmp_path / "bundle.json"),
        "--predictions",
        str(predictions),
        "--config",
        str(config),
        "--output",
        str(tmp_path / "judged"),
    ]
    return row, args, predictions


def test_preview_does_not_read_credentials_or_send_requests(
    inputs, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        pytest.fail("Preview attempted credential access or judging")

    monkeypatch.setattr(credentials, "mac_keychain_secret", forbidden)
    monkeypatch.setattr(jev, "run", forbidden)
    benign.main(inputs[1])
    preview = json.loads(capsys.readouterr().out)
    assert preview["requests_sent"] == 0
    assert preview["purpose"] == "benign_typed_rubric_judging"


def test_explicit_execute_scopes_keychain_and_preserves_raw_predictions(
    inputs, monkeypatch
):
    import os

    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.setattr(
        credentials, "mac_keychain_secret", lambda *args: "fixture-only"
    )
    called = []

    def fake_run(rows, predictions, config, output):
        assert os.environ["JEV_API_KEY"] == "fixture-only"
        assert "keychain" not in config
        assert rows[0]["id"] == predictions[0]["id"]
        called.append(config["model"])

    monkeypatch.setattr(jev, "run", fake_run)
    original = inputs[2].read_bytes()
    benign.main([*inputs[1], "--execute"])
    assert called == ["jev-latest"]
    assert "JEV_API_KEY" not in os.environ
    assert inputs[2].read_bytes() == original


def test_incomplete_predictions_fail_before_credentials(inputs, monkeypatch):
    inputs[2].write_text("")

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid predictions attempted credential access")

    monkeypatch.setattr(credentials, "mac_keychain_secret", forbidden)
    with pytest.raises(ValueError, match="Prediction IDs"):
        benign.main([*inputs[1], "--execute"])


def test_invalid_endpoint_fails_before_reading_credentials(inputs, monkeypatch):
    config_path = inputs[1][inputs[1].index("--config") + 1]
    from pathlib import Path

    config = json.loads(Path(config_path).read_text())
    config["base_url"] = "https://unrelated.example"
    Path(config_path).write_text(json.dumps(config))

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid endpoint attempted credential access")

    monkeypatch.setattr(credentials, "mac_keychain_secret", forbidden)
    with pytest.raises(ValueError, match="official TypeSafe"):
        benign.main([*inputs[1], "--execute"])
