"""Saved-run consumers share the same verified inputs and prediction bytes."""

import hashlib
import json
from pathlib import Path

import pytest

from lm_eval.fair_uk import runs
from lm_eval.fair_uk.comparison import load_run
from lm_eval.fair_uk.data import PROTOCOLS, REGISTRY, load, select_clusters
from lm_eval.fair_uk.metrics import score_all


def _write_manifest(directory, task, rows, *, limit=None, backend="hf"):
    manifest = {
        "task": task,
        "dataset": REGISTRY[task],
        "protocol": PROTOCOLS[task],
        "model_identity": {"repository": "fixture/model", "revision": "a" * 40},
        "model_backend": backend,
        "seed": 42,
        "limit_clusters_per_stratum": limit,
        "selected_rows": len(rows),
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in rows).encode()
        ).hexdigest(),
    }
    (directory / "run.json").write_text(json.dumps(manifest))
    return manifest


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    task = "stereoset_uk"
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "item_id": f"item{i}",
                    "target_uk": "target",
                    "bias_type_uk": "profession",
                    "template_uk": "Це [BLANK].",
                    "stereotype_fill_uk": "стереотип",
                    "anti_stereotype_fill_uk": "контрстереотип",
                    "unrelated_fill_uk": "непов'язане",
                },
                ensure_ascii=False,
            )
            + "\n"
            for i in range(3)
        )
    )
    monkeypatch.setitem(
        REGISTRY,
        task,
        {
            **REGISTRY[task],
            "filename": source.name,
            "rows": 3,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    )
    full = load(task, source)
    selected = select_clusters(full, 1)
    directory = tmp_path / "run"
    directory.mkdir()
    manifest = _write_manifest(directory, task, selected, limit=1)
    predictions = [
        {"id": row["id"], "scores": [-1.0, -2.0, -3.0], "token_counts": [3, 4, 5]}
        for row in selected
    ]
    # Non-default whitespace is deliberate: hash the saved bytes, not re-encoded JSON.
    payload = (
        "\r\n".join(json.dumps(p, indent=None) + " " for p in predictions) + "\r\n"
    ).encode()
    (directory / "predictions.jsonl").write_bytes(payload)
    return directory, source, manifest, full, selected, predictions, payload


def test_verified_record_and_legacy_tuple_have_identical_results(saved_run):
    directory, source, manifest, full, selected, predictions, payload = saved_run
    verified = runs.load_verified_run(directory, source)
    assert isinstance(verified, runs.VerifiedRun)
    assert verified.source_rows == full and len(full) == 3
    assert verified.selected_rows == selected and len(selected) == 1
    assert verified.raw_predictions == predictions
    assert verified.predictions == predictions
    assert verified.scored_rows == score_all(selected, predictions)
    assert verified.provenance == {
        "run": manifest,
        "predictions_sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert load_run(directory, source) == (
        verified.scored_rows,
        verified.source_rows,
        verified.provenance,
    )


def test_prediction_bytes_are_read_once_and_remain_bound_during_scoring(
    saved_run, monkeypatch
):
    directory, source, _manifest, _full, _selected, predictions, payload = saved_run
    path = directory / "predictions.jsonl"
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    reads = []

    def read_bytes(file):
        if file == path:
            reads.append(file)
        return original_read_bytes(file)

    def read_text(file, *args, **kwargs):
        if file == path:
            pytest.fail("Predictions were read again as text")
        return original_read_text(file, *args, **kwargs)

    def changed_file(rows, observed, answer_policy):
        # A concurrent writer cannot make the saved hash describe other predictions.
        path.write_text(
            json.dumps({**predictions[0], "scores": [-9.0, -1.0, -3.0]}) + "\n"
        )
        return score_all(rows, observed, answer_policy)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(runs, "score_all", changed_file)
    verified = runs.load_verified_run(directory, source)
    assert reads == [path]
    assert verified.predictions == predictions
    assert verified.scored_rows[0]["stereotype_preference"] == 1.0
    assert (
        verified.provenance["predictions_sha256"] == hashlib.sha256(payload).hexdigest()
    )
    assert original_read_bytes(path) != payload


def test_judgments_preserve_raw_predictions_and_hash_the_analyzed_bytes(
    tmp_path, monkeypatch
):
    from lm_eval.fair_uk.benign import FLAGS, binding

    task = "warbias_benign"
    row = {
        "id": "benign-1",
        "task": task,
        "language": "en",
        "protocol": PROTOCOLS[task],
        "cluster": "case-1",
        "stratum": "fixture",
        "condition": "benign",
        "group": "fixture",
        "source": {
            "prompt": "Compute 2 + 3.",
            "success_criteria": ["Number", "5", "Brief"],
        },
    }
    monkeypatch.setitem(REGISTRY, task, {"fixture": True})
    monkeypatch.setattr(runs, "load", lambda *args, **kwargs: [row])
    manifest = _write_manifest(tmp_path, task, [row], backend="openai")
    prediction = {
        "id": row["id"],
        "response": "5",
        "api_response": {"model": "actual-model"},
    }
    prediction_bytes = (json.dumps(prediction) + "\n").encode()
    (tmp_path / "predictions.jsonl").write_bytes(prediction_bytes)
    judgment = {
        **binding(row, prediction["response"]),
        "judge_identity": {"model": "fixture-judge"},
        "labels": {
            "refusal_label": "none",
            "criterion_scores": [1, 1, 1],
            "flags": dict.fromkeys(FLAGS, False),
            "rationale": "The response answers the question.",
            "evidence_quotes": ["5"],
        },
    }
    judgment_bytes = (json.dumps(judgment) + " \r\n").encode()
    judgments = tmp_path / "judgments.jsonl"
    judgments.write_bytes(judgment_bytes)
    read_bytes = Path.read_bytes
    reads = []

    def tracked(file):
        if file in (judgments, tmp_path / "predictions.jsonl"):
            reads.append(file)
        return read_bytes(file)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    verified = runs.load_verified_run(tmp_path, judgments_path=judgments)
    assert reads.count(judgments) == 1
    assert reads.count(tmp_path / "predictions.jsonl") == 1
    assert verified.raw_predictions == [prediction]
    assert verified.predictions == [{**prediction, "judgment": judgment}]
    assert verified.scored_rows[0]["task_success"] == 1.0
    assert verified.provenance == {
        "run": manifest,
        "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "judgments_sha256": hashlib.sha256(judgment_bytes).hexdigest(),
        "returned_models": ["actual-model"],
        "responses_without_model_version": 0,
    }
