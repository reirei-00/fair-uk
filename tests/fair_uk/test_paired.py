"""Before/after evaluation delegates to native scorers on fixed source cohorts."""

import copy
import hashlib
import json
import shutil

import pytest
from test_verified_runs import (
    saved_run as saved_run,  # noqa: PLC0414 -- shared pytest fixture re-export
)

from lm_eval.fair_uk.benign import FLAGS, binding
from lm_eval.fair_uk.contracts import canonical_sha256
from lm_eval.fair_uk.data import PROTOCOLS, family
from lm_eval.fair_uk.metrics import score_all
from lm_eval.fair_uk.paired import compare_runs, compare_verified
from lm_eval.fair_uk.runs import VerifiedRun


def rows(task="stereoset_uk"):
    return [
        {
            "id": f"i{i}",
            "task": task,
            "language": "uk",
            "protocol": PROTOCOLS[family(task)],
            "cluster": f"c{i}",
            "panel": f"i{i}",
            "stratum": "fixture",
            "group": "fixture",
            "group_kind": "fixture",
            "condition": "intrasentence",
            "eligible": True,
            "category": "profession",
            "independent_case_unit": f"c{i}",
            "source": {
                "prompt": "Compute 2 + 3.",
                "success_criteria": ["Number", "5", "Brief"],
            },
            "sentences": ["S", "C", "U"],
        }
        for i in range(4)
    ]


def record(source, predictions=None, scored=None, after=False):
    if predictions is None:
        predictions = [{"id": r["id"], "scores": [-1.0, -2.0, -3.0]} for r in source]
    if scored is None:
        scored = score_all(source, predictions)
    run = {
        "task": source[0]["task"],
        "protocol": source[0]["protocol"],
        "dataset": {"pinned_fixture": True},
        "model_backend": "hf",
        "seed": 42,
        "model_identity": {
            "repository": "fixture/after" if after else "fixture/before",
            "revision": "b" * 40 if after else "a" * 40,
        },
        "model_args": "pretrained=fixture/after"
        if after
        else "pretrained=fixture/before",
        "batch_size": "1",
        "code_identity": {"fixture": "hash"},
        "evaluation_asset_identity": {
            "tokenizer_sha256": "f" * 64,
            "input_sha256": canonical_sha256(source),
        },
    }
    return VerifiedRun(
        source,
        source,
        predictions,
        predictions,
        scored,
        {
            "run": run,
            "predictions_sha256": canonical_sha256(predictions),
        },
    )


def test_native_stereoset_changes_recompute_nonlinear_metric():
    source = rows()
    before_predictions = [
        {"id": r["id"], "scores": [-1.0 if i % 2 else -2.0, -1.5, -3.0]}
        for i, r in enumerate(source)
    ]
    before = record(source, before_predictions)
    after = record(source, after=True)
    result = compare_verified(before, after, draws=40)
    icat = next(
        e
        for e in result["effects"]
        if e["scope"] == "target_macro" and e["metric"] == "icat"
    )
    assert (icat["before"], icat["after"], icat["after_minus_before"]) == (
        100.0,
        0.0,
        -100.0,
    )
    assert icat["unit"] == "score_points"
    assert icat["valid_replicates"] == 40
    assert result["direction"] == "after_minus_before"


def test_no_update_is_zero_and_swap_reverses_signed_effects():
    source = rows()
    first = record(source)
    no_update = compare_verified(first, first, draws=20)
    for effect in no_update["effects"]:
        if effect["status"] == "available":
            assert effect["after_minus_before"] == 0.0
            assert effect["ci95"] == [0.0, 0.0]
    changed = record(
        source,
        [{"id": r["id"], "scores": [-2.0, -1.0, -3.0]} for r in source],
        after=True,
    )
    forward = compare_verified(first, changed, draws=20)
    reverse = compare_verified(changed, first, draws=20)
    assert [e["after_minus_before"] for e in forward["effects"]] == [
        -e["after_minus_before"] for e in reverse["effects"]
    ]


@pytest.mark.parametrize(
    "change", ["source", "ids", "tokenizer", "protocol", "missing_assets"]
)
def test_fixed_source_and_inference_contract_is_enforced(change):
    before = record(rows())
    after = copy.deepcopy(before)
    if change == "source":
        after.selected_rows[0]["sentences"][0] = "Changed text"
    elif change == "ids":
        after.selected_rows.pop()
    elif change == "missing_assets":
        del after.provenance["run"]["evaluation_asset_identity"]
    elif change == "protocol":
        after.provenance["run"]["protocol"] = "different"
    else:
        after.provenance["run"]["evaluation_asset_identity"]["tokenizer_sha256"] = (
            "0" * 64
        )
    with pytest.raises(
        ValueError, match="source|identical IDs|inference settings|fingerprints"
    ):
        compare_verified(before, after, draws=0)


def test_primary_annotated_warbias_is_not_replaced_by_native_raw_scoring():
    source = rows()
    source[0]["annotation_panel_version"] = "fixture"
    before = record(source)
    with pytest.raises(ValueError, match="original\\+neutral"):
        compare_verified(before, before, draws=0)


def test_undefined_bbq_denominator_retains_native_zero_but_not_a_paired_effect():
    source = rows("bbq_uk")
    for row in source:
        row["condition"] = "ambiguous"
    scored = [
        {
            **r,
            "accuracy": 1.0,
            "unknown_rate": 1.0,
            "stereotype_rate": 0.0,
            "counter_rate": 0.0,
            "tie": 0.0,
        }
        for r in source
    ]
    before = record(source, scored=scored)
    result = compare_verified(before, before, draws=20)
    bias = [
        e
        for e in result["effects"]
        if e["metric"]
        in ("raw_bias", "bias_score", "signed_bias_macro", "absolute_bias_macro")
    ]
    assert bias
    for effect in bias:
        assert effect["before"] == effect["after"] == 0.0
        assert effect["after_minus_before"] is None
        assert effect["ci95"] is None
        assert effect["reason"] == "undefined_no_non_unknown_mass"
    accuracy = next(e for e in result["effects"] if e["metric"] == "accuracy_all_rows")
    assert accuracy["after_minus_before"] == 0.0


def test_missing_judgments_never_become_refusal_or_neutrality():
    source = rows("warbias_benign_uk")
    pending = [{"id": r["id"], "response": "5"} for r in source]
    complete = []
    for row, prediction in zip(source, pending, strict=True):
        judgment = {
            **binding(row, "5"),
            "judge_identity": {"model": "fixture"},
            "labels": {
                "refusal_label": "none",
                "criterion_scores": [1, 1, 1],
                "flags": dict.fromkeys(FLAGS, False),
                "rationale": "Answered.",
                "evidence_quotes": ["5"],
            },
        }
        complete.append({**prediction, "judgment": judgment})
    result = compare_verified(
        record(source, pending), record(source, complete, after=True), draws=20
    )
    outcomes = [
        e
        for e in result["effects"]
        if e["metric"] in ("task_success", "full_refusal_rate", "any_refusal_rate")
    ]
    assert outcomes and all(e["after_minus_before"] is None for e in outcomes)
    assert result["coverage"]["judgments"]["paired_scorable"] == 0
    assert result["coverage"]["judgments"]["after_scorable"] == 4
    assert any(
        r["task_success"] == 1.0 for r in result["native_checkpoint_scores"]["after"]
    )


def test_public_saved_run_api_loads_verified_records_and_writes_immutable_report(
    saved_run, tmp_path
):
    directory, source, _manifest, _full, _selected, _predictions, _payload = saved_run
    path = directory / "run.json"
    run = json.loads(path.read_text())
    run["model_args"] = "pretrained=fixture/before"
    run["evaluation_asset_identity"] = {
        "tokenizer_sha256": "f" * 64,
        "input_sha256": "d" * 64,
    }
    path.write_text(json.dumps(run))
    after = tmp_path / "after"
    shutil.copytree(directory, after)
    run["model_args"] = "pretrained=fixture/after"
    run["model_identity"]["revision"] = "b" * 40
    (after / "run.json").write_text(json.dumps(run))
    output = tmp_path / "report"
    result = compare_runs(
        directory,
        after,
        before_input=source,
        after_input=source,
        output=output,
        draws=20,
    )
    assert json.loads((output / "paired.json").read_text()) == result
    assert (
        result["provenance"]["before"]["predictions_sha256"]
        == hashlib.sha256((directory / "predictions.jsonl").read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="new or empty"):
        compare_runs(
            directory,
            after,
            before_input=source,
            after_input=source,
            output=output,
            draws=0,
        )


def test_winopron_native_slices_and_order_pairs_survive_component_bootstrap():
    from test_winopron import correct, panel

    source = [*panel("p1"), *panel("p2")]
    # A second synthetic pronoun type must remain a separate native slice.
    other = copy.deepcopy(source)
    for row in other:
        for field in (
            "id",
            "row_id",
            "pair_id",
            "panel",
            "cluster",
            "comparison_unit",
            "independent_case_unit",
        ):
            if field in row:
                row[field] = "other:" + row[field]
        row["pronoun_type"] = "accusative"
        row["stratum"] = "accusative"
    source += other
    predictions = correct(source)
    before = record(source, predictions)
    result = compare_verified(before, before, draws=10)
    assert not any(
        e["metric"] in ("ss", "lms", "icat", "stereotype_preference")
        for e in result["effects"]
    )
    scoped = [
        e
        for e in result["effects"]
        if e["metric"] == "accuracy" and "context_pronoun_type_set" in e["scope"]
    ]
    assert any("nominative" in e["scope"] for e in scoped)
    assert any("accusative" in e["scope"] for e in scoped)
    for effect in result["effects"]:
        if effect["status"] == "available":
            assert effect["after_minus_before"] == 0.0
            assert effect["ci95"] == [0.0, 0.0]


def test_paired_settings_accept_actual_encoder_evidence_and_detect_changed_tokens():
    from test_evaluation_assets import FakeBackend

    from lm_eval.fair_uk.evaluation_assets import evaluation_asset_identity
    from lm_eval.fair_uk.provenance import paired_inference_settings

    source = rows()
    backend = FakeBackend()
    before, after = record(source), record(source, after=True)
    evidence = evaluation_asset_identity(backend, source, "raw")
    before.provenance["run"]["evaluation_asset_identity"] = evidence
    after.provenance["run"]["evaluation_asset_identity"] = evidence
    assert paired_inference_settings(
        before.provenance["run"]
    ) == paired_inference_settings(after.provenance["run"])
    assert compare_verified(before, after, draws=0)["effects"]
    backend.tokenizer.offset = 1
    after.provenance["run"]["evaluation_asset_identity"] = evaluation_asset_identity(
        backend, source, "raw"
    )
    with pytest.raises(ValueError, match="effective|inputs differ"):
        compare_verified(before, after, draws=0)


def test_collected_safe_model_parameters_preserve_precision_when_checkpoint_changes():
    from lm_eval.fair_uk.provenance import model_identity

    source = rows()
    before, after = record(source), record(source, after=True)
    del before.provenance["run"]["model_args"]
    del after.provenance["run"]["model_args"]
    before.provenance["run"]["model_identity"] = model_identity(
        "pretrained=fixture/before,revision=" + "a" * 40 + ",dtype=bfloat16"
    )
    after.provenance["run"]["model_identity"] = model_identity(
        "pretrained=fixture/after,revision=" + "b" * 40 + ",dtype=bfloat16"
    )
    assert compare_verified(before, after, draws=0)["effects"]
    after.provenance["run"]["model_identity"]["parameters"]["dtype"] = "float16"
    with pytest.raises(ValueError, match="inference settings"):
        compare_verified(before, after, draws=0)
