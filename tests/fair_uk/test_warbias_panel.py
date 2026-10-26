"""End to end: phase runs -> panel report -> calibration -> tables (synthetic, offline)."""

import csv
import json
from pathlib import Path

import pytest
from warbias_support import PROFILES, neutral_records, phase_runs, write_jsonl

from lm_eval.fair_uk.data import digest
from lm_eval.fair_uk.warbias.calibration import TAU_RULE, calibrate_tau, development_tau
from lm_eval.fair_uk.warbias.cli import main as cli
from lm_eval.fair_uk.warbias.panel import load_language_runs, panel_report
from lm_eval.fair_uk.warbias.rows import adapt_annotated, task_name
from lm_eval.fair_uk.warbias.tables import write_tables


@pytest.fixture
def report(warbias_inputs, monkeypatch):
    files, inv, tmp = warbias_inputs
    records = neutral_records(files, monkeypatch)
    full, phases, runs, neutral_runs = phase_runs(files, inv, tmp, records)
    out = panel_report(
        full,
        runs,
        tmp / "results" / "panel_m",
        neutral_runs=neutral_runs,
        min_independent_support=2,
        bootstrap=20,
    )
    return out, full, phases, runs, neutral_runs, tmp


def test_report_writes_primary_and_diagnostic_outputs(report):
    out, *_ = report
    primary = json.loads((out / "uk" / "prior_normalised.json").read_text())
    assert primary["coverage"] == {
        "triplets_with_neutral": len(PROFILES) - 1,
        "panel_triplets": len(PROFILES),
    }
    assert 0.0 <= primary["all"]["ss"] <= 1.0 and primary["units"]["rates"].startswith(
        "proportion"
    )
    assert {c["axis"] for c in primary["cells"]} >= {"all", "actor", "gender", "age"}
    assert len(primary["items"]) == len(PROFILES) - 1
    raw = out / "uk" / "raw_scoring_diagnostics"
    assert "not primary" in (raw / "README.md").read_text()
    assert json.loads((raw / "report.json").read_text())["rows"] == len(PROFILES)
    assert (out / "raw_scoring_diagnostics" / "uk_en_comparison.json").exists()
    detection = json.loads((out / "detection_analysis.json").read_text())
    contrast = detection["language_contrast"]
    assert contrast["language_pair"] == {"lg1": "uk", "lg2": "en"}
    assert contrast["all"]["lg2_minus_lg1"] == 0.0  # identical scores
    sources = json.loads((out / "panel_sources.json").read_text())
    assert sum(r["rows"] for r in sources["runs"]["en"]) == len(PROFILES)
    assert sources["shared_inference_identity"]["model_backend"] == "vllm"


def test_report_rejects_gaps_and_mixed_identities(report):
    out, full, _phases, runs, _neutral_runs, tmp = report
    partial = {lang: entries[:-1] for lang, entries in runs.items()}
    with pytest.raises(ValueError, match="cover"):
        panel_report(full, partial, tmp / "partial", bootstrap=0)
    _bundle, directory = runs["uk"][0]
    run = json.loads((directory / "run.json").read_text())
    (directory / "run.json").write_text(json.dumps({**run, "seed": 7}))
    with pytest.raises(ValueError, match="identity"):
        panel_report(full, runs, tmp / "mixed", bootstrap=0)
    with pytest.raises(ValueError, match="empty output"):
        panel_report(full, runs, out, bootstrap=0)


def test_phase_collection_consumes_verified_predictions_without_rereading(
    report, monkeypatch
):
    _out, full, _phases, runs, _neutral_runs, _tmp = report
    paths = [directory / "predictions.jsonl" for _bundle, directory in runs["uk"]]
    expected = [
        json.loads(line) for path in paths for line in path.read_text().splitlines()
    ]
    digests = {path: digest(path) for path in paths}
    original_read_bytes, original_read_text = Path.read_bytes, Path.read_text
    reads = []

    def read_bytes(path):
        if path in paths:
            reads.append(path)
        return original_read_bytes(path)

    def read_text(path, *args, **kwargs):
        if path in paths:
            pytest.fail("Panel reread the verified predictions as text")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(Path, "read_text", read_text)
    collected = load_language_runs(task_name("uk"), full, runs["uk"])
    assert collected.predictions == expected
    assert reads == paths
    assert [source["predictions_sha256"] for source in collected.sources] == [
        digests[path] for path in paths
    ]


def _change_full_panel_source(bundle, *, neutral, change):
    """Change valid full-panel inputs without changing the bound phase runs."""
    manifest = json.loads(bundle.read_text())
    task = task_name("uk", neutral=neutral)
    entry = manifest["datasets"][task]
    path = bundle.parent / entry["filename"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    source = rows[0]["source"]
    change(source)
    rows[0] = adapt_annotated(source["released"], source["metadata"], "uk")
    write_jsonl(path, rows)
    entry["sha256"] = digest(path)
    bundle.write_text(json.dumps(manifest))


@pytest.mark.parametrize("changed_input", ["original", "neutral", "neutral_source"])
def test_report_rejects_phase_full_text_mismatch(report, changed_input):
    _out, full, _phases, runs, neutral_runs, tmp = report

    def change(source):
        target = (
            source["metadata"]["neutral_variant"]
            if changed_input == "neutral"
            else source["released"]
        )
        target["stereotype"] = "A different claim with valid adapted text."

    _change_full_panel_source(full, neutral=changed_input != "original", change=change)
    output = tmp / "mismatched_text"
    with pytest.raises(ValueError, match="phase input text differs"):
        panel_report(full, runs, output, neutral_runs=neutral_runs, bootstrap=0)
    assert not output.exists()


def test_report_allows_source_annotation_differences(report):
    _out, full, _phases, runs, neutral_runs, tmp = report

    def change(source):
        source["metadata"]["attributes"]["review_note"] = "Additional source annotation"

    for neutral in (False, True):
        _change_full_panel_source(full, neutral=neutral, change=change)
    output = panel_report(
        full, runs, tmp / "annotation_only", neutral_runs=neutral_runs, bootstrap=0
    )
    assert (output / "uk" / "prior_normalised.json").exists()


def test_calibration_uses_development_margins(report):
    out, full, phases, *_ = report
    record = development_tau(phases["development"], out, model_id="m")
    assert record["tau_rule"]["id"] == TAU_RULE["id"]
    assert record["languages"]["uk"]["development_rows"] == 1
    assert record["languages"]["uk"]["margin_unit"] == "nats_summed_prior_normalised"
    with pytest.raises(ValueError, match="development"):
        development_tau(full, out, model_id="m")
    items = [
        {"id": f"d{k}", "cluster": f"k{k % 2}", "margin": m}
        for k, m in enumerate([-1.0, 2.0, -3.0, 4.0, 5.0])
    ]
    result = calibrate_tau(
        items + [{"id": "e", "cluster": "z", "margin": 99.0}],
        [f"d{k}" for k in range(5)] + ["gone"],
    )
    assert result["tau"] == pytest.approx(1.8)
    assert result["development_rows_without_neutral"] == 1
    assert calibrate_tau(items, ["d0"], floor=5.0)["tau"] == 5.0


def test_tables_render_saved_reports_without_recomputation(report, capsys):
    out, *_ = report
    paths = write_tables(out.parent)
    rows = list(csv.DictReader(paths[0].open()))
    assert {r["model"] for r in rows} == {"m"} and {r["language"] for r in rows} == {
        "uk",
        "en",
    }
    primary = json.loads((out / "uk" / "prior_normalised.json").read_text())
    first = next(r for r in rows if r["axis"] == "all" and r["language"] == "uk")
    assert float(first["ss"]) == pytest.approx(100 * primary["all"]["ss"])
    text = paths[1].read_text()
    assert "## m" in text and "ICAT" in text  # caption explains why there is none
    assert "| all: all | LG1 (uk) |" in text
    assert "Worst-group SS distance" in paths[2].read_text()
    cli(["tables", "--results", str(out.parent), "--output", str(out.parent / "t2")])
    assert (out.parent / "t2" / "baseline_long.csv").read_text() == paths[0].read_text()
