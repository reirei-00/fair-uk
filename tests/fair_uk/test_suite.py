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

    monkeypatch.setattr("lm_eval.fair_uk.suite._run_child", forbidden)
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

    monkeypatch.setattr(
        "lm_eval.fair_uk.suite.load", lambda *a, **k: [{"id": "fixture"}]
    )
    monkeypatch.setattr("lm_eval.fair_uk.suite.select_clusters", lambda rows, _: rows)
    monkeypatch.setattr(
        "lm_eval.fair_uk.suite.verify_result", lambda *a: {"status": "complete"}
    )

    def run(command):
        calls.append(command)
        if command[3] in ("compare", "summarize"):
            destination = tmp_path / (
                "comparisons/warbias" if command[3] == "compare" else "summary"
            )
            destination.mkdir(parents=True, exist_ok=True)
            name = "comparison.json" if command[3] == "compare" else "summary.json"
            (destination / name).write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("lm_eval.fair_uk.suite._run_child", run)
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
    assert len(calls) == 4
    for command, task in zip(calls[:2], ["warbias_uk", "warbias_en"], strict=True):
        assert command[command.index("--task") + 1] == task
        assert command[command.index("--output") + 1] == str(tmp_path / task)
        assert command[command.index("--answer-policy") + 1] == "strict_abc_v1"
        assert command[command.index("--limit-clusters-per-stratum") + 1] == "2"
    assert calls[-1][3] == "summarize"
    calls.clear()
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
    assert [command[3] for command in calls] == ["compare", "summarize"]


def test_failed_task_stops_suite_without_leaking_arguments(
    tmp_path, monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr("lm_eval.fair_uk.provenance.model_identity", lambda _: {})

    def run(command):
        calls.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr("lm_eval.fair_uk.suite._run_child", run)
    monkeypatch.setattr(
        "lm_eval.fair_uk.suite.load", lambda *a, **k: [{"id": "fixture"}]
    )
    with pytest.raises(RuntimeError) as error:
        main(
            [
                "suite",
                "--execute",
                "--model-args",
                "pretrained=m,token=SECRET_SENTINEL",
                "--output",
                str(tmp_path),
            ]
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

    monkeypatch.setattr("lm_eval.fair_uk.suite._run_child", forbidden)
    with pytest.raises(SystemExit):
        main(["suite", *arguments])


def test_bilingual_preview_reports_missing_counterparts(capsys):
    main(["suite", "--languages", "uk", "en"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["ready"] is False
    assert set(plan["missing_tasks"]) == {
        "bbq_en",
        "stereoset_en",
        "winobias_en_natural",
        "winobias_en_controlled",
    }
    assert ["warbias_uk", "warbias_en"] in plan["paired_comparisons"]
    assert (
        next(t for t in plan["tasks"] if t["task"] == "bbq_uk")[
            "estimated_requests_full_dataset"
        ]
        == 58492 * 9
    )


def test_api_rejects_likelihood_task_before_client_or_data(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported capabilities must fail before data access")

    monkeypatch.setattr("lm_eval.fair_uk.suite.load", forbidden)
    with pytest.raises(SystemExit):
        main(["suite", "--model", "api", "--tasks", "bbq_uk"])


def test_language_selection_expands_selected_benchmark(capsys):
    main(["suite", "--benchmarks", "warbias", "--languages", "uk", "en"])
    plan = json.loads(capsys.readouterr().out)
    assert [r["task"] for r in plan["tasks"]] == ["warbias_uk", "warbias_en"]
    assert plan["ready"] is True


def test_successful_exit_without_results_cannot_complete_suite(tmp_path, monkeypatch):
    monkeypatch.setattr("lm_eval.fair_uk.provenance.model_identity", lambda _: {})
    monkeypatch.setattr(
        "lm_eval.fair_uk.suite.load", lambda *a, **k: [{"id": "fixture"}]
    )
    monkeypatch.setattr(
        "lm_eval.fair_uk.suite._run_child",
        lambda *a, **k: SimpleNamespace(returncode=0),
    )
    with pytest.raises(FileNotFoundError):
        main(
            [
                "suite",
                "--tasks",
                "warbias_uk",
                "--model-args",
                "pretrained=fixture",
                "--output",
                str(tmp_path),
                "--execute",
            ]
        )
    state = json.loads((tmp_path / "suite.json").read_text())
    assert state["status"] == "incomplete"
    assert state["failed_step"] == "warbias_uk"


def test_suite_forwards_termination_to_model_process_group(monkeypatch):
    import signal

    from lm_eval.fair_uk.suite import _run_child

    handlers, signals = {}, []

    class Child:
        pid = 123456789
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if self.returncode is None:
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            return self.returncode

    child = Child()

    def forward(pid, signum):
        signals.append((pid, signum))
        child.returncode = -signum

    def handler(signum, callback):
        old = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = callback
        return old

    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: child)
    monkeypatch.setattr("os.killpg", forward)
    monkeypatch.setattr("signal.signal", handler)
    with pytest.raises(KeyboardInterrupt):
        _run_child(["fixture"])
    assert signals == [(child.pid, signal.SIGTERM)]
    assert all(value == signal.SIG_DFL for value in handlers.values())
