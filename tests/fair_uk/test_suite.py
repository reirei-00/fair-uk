import json
import sys
from types import SimpleNamespace

import pytest

from lm_eval.fair_uk.__main__ import main
from lm_eval.fair_uk.data import REGISTRY
from lm_eval.fair_uk.suite import DEFAULT_TASKS


def test_registry_listing_and_language_filter(capsys):
    main(["list"])
    assert json.loads(capsys.readouterr().out) == REGISTRY
    main(["list", "--language", "en", "--format", "table"])
    output = capsys.readouterr().out
    assert "warbias_en" in output and "warbias_intersectional_en" in output
    assert "bbq_uk" not in output
    assert "human validation pending" in output


def test_listing_does_not_require_optional_gemini_client(monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "lm_eval.fair_uk.gemini_runner", raising=False)
    monkeypatch.setitem(sys.modules, "httpx", None)
    main(["list"])
    assert json.loads(capsys.readouterr().out) == REGISTRY


def test_default_suite_is_offline_and_does_not_echo_credentials(
    tmp_path, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "Planning must not launch inference, fetch data, or fingerprint model files"
        )

    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("lm_eval.fair_uk.data.load", forbidden)
    monkeypatch.setattr("lm_eval.fair_uk.provenance.model_identity", forbidden)
    output = tmp_path / "not_created"
    main(
        [
            "suite",
            "--output",
            str(output),
            "--model-args",
            "pretrained=model,token=SECRET_SENTINEL",
        ]
    )
    text = capsys.readouterr().out
    assert "SECRET_SENTINEL" not in text
    plan = json.loads(text)
    assert plan["status"] == "prepared_not_running"
    assert [r["task"] for r in plan["tasks"]] == list(DEFAULT_TASKS)
    assert all(r["dataset"]["language"] == "uk" for r in plan["tasks"])
    assert not output.exists()


def test_suite_routes_every_task_and_summary_without_inference(tmp_path, monkeypatch):
    calls = []
    identities = []
    monkeypatch.setattr("lm_eval.fair_uk.provenance.model_identity", identities.append)

    def run(command, check):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", run)
    main(
        [
            "suite",
            "--execute",
            "--tasks",
            "warbias_uk",
            "warbias_en",
            "--model-args",
            "pretrained=pinned-model",
            "--output",
            str(tmp_path),
            "--limit-clusters-per-stratum",
            "2",
            "--answer-policy",
            "strict_abc_v1",
        ]
    )
    assert identities == ["pretrained=pinned-model"]
    assert len(calls) == 3
    for command, task in zip(calls[:2], ["warbias_uk", "warbias_en"], strict=True):
        assert command[command.index("--task") + 1] == task
        assert command[command.index("--output") + 1] == str(tmp_path / task)
        assert command[command.index("--answer-policy") + 1] == "strict_abc_v1"
        assert command[command.index("--limit-clusters-per-stratum") + 1] == "2"
    assert calls[-1][3] == "summarize"


def test_failed_task_stops_suite_without_leaking_arguments(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr("lm_eval.fair_uk.provenance.model_identity", lambda _: {})

    def run(command, check):
        calls.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr("subprocess.run", run)
    with pytest.raises(SystemExit) as error:
        main(
            ["suite", "--execute", "--model-args", "pretrained=m,token=SECRET_SENTINEL"]
        )
    assert len(calls) == 1
    assert "Later tasks were not started" in str(error.value)
    assert "SECRET_SENTINEL" not in str(error.value) + capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        ["--execute"],
        ["--tasks", "warbias_uk", "warbias_uk"],
        ["--limit-clusters-per-stratum", "0"],
        ["--bootstrap", "-1"],
    ],
)
def test_invalid_suite_arguments_fail_before_execution(arguments, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid plans must never execute")

    monkeypatch.setattr("subprocess.run", forbidden)
    with pytest.raises(SystemExit):
        main(["suite", *arguments])
