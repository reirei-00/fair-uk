"""Selected-language analysis retains full source and run verification."""

import json
import shutil

import pytest
from test_warbias_baseline import _saved_runs, release as release  # noqa: PLC0414
from test_warbias_baseline_neutral import supplement as supplement  # noqa: PLC0414

from lm_eval.fair_uk.warbias.baseline import analyze_checkpoints, write_baseline_results
from lm_eval.fair_uk.warbias.baseline_inputs import sha


def _bind_tasks(results, tasks):
    root = results / "fixture"
    path = root / "binding.json"
    binding = json.loads(path.read_text())
    binding["tasks"] = tasks
    path.write_text(json.dumps(binding) + "\n")
    path = root / "qualification.json"
    qualification = json.loads(path.read_text())
    qualification["binding_sha256"] = sha(root / "binding.json")
    path.write_text(json.dumps(qualification) + "\n")


@pytest.mark.parametrize("language", ["uk", "en"])
@pytest.mark.parametrize("bind_subset", [False, True])
def test_single_language_needs_only_selected_outputs_and_keeps_all_metrics(
    release, supplement, tmp_path, language, bind_subset
):
    results, controls = tmp_path / "results", tmp_path / "controls"
    _saved_runs(release, results)
    _saved_runs(supplement, controls)
    settings = {
        "bootstrap": 0,
        "neutral_release": supplement,
        "neutral_results": controls,
    }
    bilingual = analyze_checkpoints(release, results, **settings)["models"]["fixture"]
    for root in (results, controls):
        tasks = sorted(path.name for path in (root / "fixture/full").iterdir())
        for task in tasks:
            if not task.endswith(f"_{language}"):
                shutil.rmtree(root / "fixture/full" / task)
        _bind_tasks(
            root,
            [
                task
                for task in tasks
                if not bind_subset or task.endswith(f"_{language}")
            ],
        )
    source_before = {
        path: path.read_bytes() for path in release.rglob("*") if path.is_file()
    }
    report = analyze_checkpoints(release, results, languages=[language], **settings)
    assert report["languages"] == [language]
    assert "language_pair" not in report
    assert report["primary_prior_normalised_ss"]["status"] == "available"
    model = report["models"]["fixture"]
    assert model["status"] == "complete", model
    assert len(model["runs"]) == 2
    assert model["language_contrasts"] == []
    assert all(
        value["status"] == "not_requested"
        for value in model["language_inference"].values()
    )
    for scoring in ("per_token", "per_byte", "sum", "prior_normalised"):
        assert model["triplets"][scoring] == {
            language: bilingual["triplets"][scoring][language]
        }
        assert model["group_diagnostics"][scoring] == {
            language: bilingual["group_diagnostics"][scoring][language]
        }
    assert model["qa"] == {language: bilingual["qa"][language]}
    write_baseline_results(report, tmp_path / "analysis")
    assert source_before == {
        path: path.read_bytes() for path in release.rglob("*") if path.is_file()
    }
    assert (
        analyze_checkpoints(release, results, **settings)["models"]["fixture"]["status"]
        == "pending_collection"
    )


@pytest.mark.parametrize("languages", [[], ["uk", "uk"], ["fr"], "uk"])
def test_invalid_language_selection_is_rejected(release, tmp_path, languages):
    with pytest.raises(ValueError, match="unique languages"):
        analyze_checkpoints(release, tmp_path, languages=languages)


@pytest.mark.parametrize("neutral", [False, True])
def test_selected_task_requires_matching_qualified_scope(
    release, supplement, tmp_path, neutral
):
    results, controls = tmp_path / "results", tmp_path / "controls"
    _saved_runs(release, results)
    _saved_runs(supplement, controls)
    root = controls if neutral else results
    tasks = [
        path.name
        for path in (root / "fixture/full").iterdir()
        if path.name.endswith("_en")
    ]
    _bind_tasks(root, tasks)
    report = analyze_checkpoints(
        release,
        results,
        languages=["uk"],
        bootstrap=0,
        neutral_release=supplement,
        neutral_results=controls,
    )
    status = (
        report["primary_prior_normalised_ss"]
        if neutral
        else report["models"]["fixture"]
    )
    assert status["status"] == "validation_failed"


def test_selection_still_verifies_selected_predictions(release, tmp_path):
    results = tmp_path / "results"
    _saved_runs(release, results)
    path = results / "fixture/full/warbias_uk/predictions.jsonl"
    path.write_text(path.read_text().replace('"response": "C"', '"response": "A"'))
    report = analyze_checkpoints(release, results, languages=["uk"], bootstrap=0)
    assert report["models"]["fixture"]["status"] == "validation_failed"


def test_cli_selects_one_language(release, tmp_path):
    from lm_eval.fair_uk.warbias.cli import main

    results = tmp_path / "results"
    _saved_runs(release, results)
    output = tmp_path / "analysis"
    main(
        [
            "baseline",
            "--release",
            str(release),
            "--results",
            str(results),
            "--output",
            str(output),
            "--language",
            "en",
            "--bootstrap",
            "0",
        ]
    )
    report = json.loads((output / "summary.json").read_text())
    assert report["languages"] == ["en"]
    assert set(report["models"]["fixture"]["qa"]) == {"en"}
