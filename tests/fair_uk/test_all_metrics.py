"""Saved-evidence integration and honest unavailable metric states."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
from test_intervention_metrics import constant, predictions, qa, triplet

from lm_eval.fair_uk import all_metrics
from lm_eval.fair_uk.answers import NORMALIZED, STRICT
from lm_eval.fair_uk.baseline import measurement_policy
from lm_eval.fair_uk.data import PROTOCOLS, dataset_spec
from lm_eval.fair_uk.metrics import score_all
from lm_eval.fair_uk.provenance import code_identity


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n")


def exported(rows, pre, post, policy=NORMALIZED):
    return {
        "rows": rows,
        "before_predictions": pre,
        "after_predictions": post,
        "before_scored": score_all(rows, pre, policy),
        "answer_policy": policy,
        "provenance": {"before": {"predictions_sha256": "a" * 64}},
    }


def test_no_pairs_keeps_all_eighteen_entries_and_blocks(tmp_path):
    config = tmp_path / "config.json"
    save(
        config,
        {
            "schema_version": 1,
            "kind": "fair_uk_all_metrics",
            "pairs": [],
            "intervention": {
                "method": "fixture",
                "training_language": "uk",
                "target_values": {},
            },
            "required_tasks": ["stereoset_uk", "stereoset_en"],
            "task_blocks": {"stereoset_en": "Unpublished source"},
        },
    )
    report = all_metrics.analyze(config, tmp_path / "result")
    assert len(report["metrics"]) == 18 and report["model_requests"] == 0
    assert all(m["status"] == "pending" for m in report["metrics"])
    assert report["rules"]["min_independent_support"] is None
    assert report["task_status"]["stereoset_en"]["status"] == "blocked"
    assert report["task_status"]["stereoset_uk"]["status"] == "pending"


@pytest.mark.parametrize("minimum", [True, 1, -1, 2.5])
def test_support_rule_has_no_silent_five_default(minimum):
    with pytest.raises(ValueError):
        all_metrics.rules({"min_independent_support": minimum})


def test_frozen_parser_policy_is_used_for_qa(tmp_path):
    rows = [qa(str(i)) for i in range(2)]
    pre = [{"id": r["id"], "response": "A."} for r in rows]
    pair = {"task": rows[0]["task"]}
    strict, _ = all_metrics.analyze_pair(
        exported(rows, pre, pre, STRICT),
        pair,
        all_metrics.rules({"bootstrap": 0}),
        tmp_path,
    )
    normalized, _ = all_metrics.analyze_pair(
        exported(rows, pre, pre), pair, all_metrics.rules({"bootstrap": 0}), tmp_path
    )
    assert strict["answer_policy"] == STRICT
    assert (
        strict["comparison"]["checkpoint_pre"]["qa_caution"]
        != normalized["comparison"]["checkpoint_pre"]["qa_caution"]
    )


def test_nnr_calibration_without_tau_cannot_be_available(tmp_path):
    rows = [triplet(str(i)) for i in range(2)]
    scores = constant(rows, [-2, -1, -3])
    pair = {"task": rows[0]["task"]}
    rules = all_metrics.rules(
        {
            "bootstrap": 0,
            "neutrality_calibration": {pair["task"]: {"calibration_id": "named"}},
        }
    )
    result, _ = all_metrics.analyze_pair(
        exported(rows, scores, scores), pair, rules, tmp_path
    )
    inventory = all_metrics.metric_inventory(
        [result], {}, {"status": "pending"}, [pair["task"]], [pair["task"]], rules
    )
    assert next(m for m in inventory if m["id"] == "NNR")["status"] == "pending"


def test_empty_frozen_nnr_cohort_is_undefined(tmp_path):
    rows = [triplet(str(i)) for i in range(2)]
    scores = constant(rows, [-2, -1, -3])
    task = rows[0]["task"]
    rules = all_metrics.rules(
        {
            "bootstrap": 0,
            "tau_by_task": {task: 0.1},
            "neutrality_calibration": {
                task: {
                    "calibration_id": "frozen",
                    "baseline_predictions_sha256": "a" * 64,
                    "baseline_eligible_ids": [],
                }
            },
        }
    )
    result, _ = all_metrics.analyze_pair(
        exported(rows, scores, scores), {"task": task}, rules, tmp_path
    )
    inventory = all_metrics.metric_inventory(
        [result], {}, {"status": "pending"}, [task], [task], rules
    )
    entry = next(m for m in inventory if m["id"] == "NNR")
    assert entry["status"] == "pending" and "undefined" in entry["reason"]
    rules["neutrality_calibration"][task]["baseline_predictions_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="baseline predictions"):
        all_metrics.analyze_pair(
            exported(rows, scores, scores), {"task": task}, rules, tmp_path
        )


def test_mixed_transfer_contexts_remain_partial():
    transfer = {
        "M3d": [
            {"status": "available", "target_values": ["uk"]},
            {"status": "unavailable", "target_values": ["uk"]},
        ]
    }
    inventory = all_metrics.metric_inventory(
        [], transfer, {"status": "pending"}, [], [], {}
    )
    assert next(m for m in inventory if m["id"] == "M3d")["status"] == "partial"


def test_warbias_undefined_adaptation_is_partial_and_referenced(tmp_path):
    rows = [qa(str(i)) for i in range(2)]
    unknown = [{"id": row["id"], "response": "C"} for row in rows]
    task = rows[0]["task"]
    result, _ = all_metrics.analyze_pair(
        exported(rows, unknown, unknown),
        {"task": task},
        all_metrics.rules({"bootstrap": 0}),
        tmp_path,
    )
    inventory = all_metrics.metric_inventory(
        [result], {}, {"status": "pending"}, [task], [task], {}
    )
    entry = next(m for m in inventory if m["id"] == "BBQ")
    assert entry["status"] == "partial"
    assert any(
        reference.endswith("/paired_profiles/warbias_bbq")
        for reference in entry["references"]
    )


def test_unpublished_policy_verifies_local_bytes(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("[]")
    sha = all_metrics.digest(source)
    config = {
        "approved_dataset_scope": {
            "namespace": "FairForget",
            "published_only": True,
            "training_repositories": [],
            "evaluation_only_repositories": [],
            "preparation_only_repositories": ["FairForget/Staged"],
        }
    }
    spec = {
        "measurement_eligibility": "preparation_only",
        "measurement_eligibility_reason": "unpublished",
        "source_role": "evaluation",
        "measurement_purpose": "preparation",
        "source_provenance": [
            {
                "repo_id": "FairForget/Staged",
                "path": str(source),
                "sha256": sha,
                "published": False,
                "revision": None,
            }
        ],
    }
    assert measurement_policy(config, spec, "evaluation")[0] == "preparation_only"
    source.write_text("[1]")
    with pytest.raises(ValueError, match="bytes changed"):
        measurement_policy(config, spec, "evaluation")


@pytest.fixture
def frozen_pair(tmp_path):
    task = "warbias_triplets_uk"
    rows = [
        {
            **triplet(str(i)),
            "panel": str(i),
            "split": "evaluation",
            "paired_eligible": False,
            "pairing_exclusion_reason": "UK-only fixture",
        }
        for i in range(4)
    ]
    source = tmp_path / "data.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec = {
        "format": "adapted_jsonl_v1",
        "filename": source.name,
        "rows": len(rows),
        "sha256": all_metrics.digest(source),
        "language": "uk",
        "protocol": PROTOCOLS["warbias_triplets"],
        "counterpart": None,
        "pairing": {"mode": "unpaired", "reason": "UK-only fixture"},
    }
    bundle = tmp_path / "manifest.json"
    save(bundle, {"schema_version": 1, "datasets": {task: spec}})
    spec = dataset_spec(task, bundle)
    directories = []
    for phase in ("before", "after"):
        directory = tmp_path / phase
        directory.mkdir()
        values = (
            constant(rows, [-1, -2, -3])
            if phase == "before"
            else predictions(rows, [[-1, -2, -3], [-2, -1, -3]] * 2)
        )
        manifest = {
            "task": task,
            "dataset": spec,
            "protocol": PROTOCOLS["warbias_triplets"],
            "model_identity": {
                "repository": "fixture/" + phase,
                "revision": "a" * 40,
                "parameters": {"pretrained": "fixture/" + phase},
            },
            "model_backend": "hf",
            "seed": 42,
            "batch_size": "1",
            "limit_clusters_per_stratum": None,
            "selected_rows": len(rows),
            "selected_ids_sha256": hashlib.sha256(
                "\n".join(r["id"] for r in rows).encode()
            ).hexdigest(),
            "code_identity": code_identity(),
            "evaluation_asset_identity": {
                "tokenizer_sha256": "a" * 64,
                "input_sha256": all_metrics.digest(source),
            },
        }
        save(directory / "run.json", manifest)
        save(directory / "report.json", {})
        (directory / "predictions.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in values)
        )
        directories.append(directory)
    pair = {
        "task": task,
        "phase": "evaluation",
        "before_run": str(directories[0]),
        "after_run": str(directories[1]),
        "dataset_bundle": str(bundle),
    }
    collector = {
        "python": sys.executable,
        "evaluator_root": str(Path(__file__).resolve().parents[2]),
    }
    return pair, collector, rows, source, bundle, directories


def test_frozen_subprocess_preserves_runs_and_computes_native_effect(
    frozen_pair, tmp_path
):
    pair, collector, _, _, _, directories = frozen_pair
    before_bytes = [(d / "run.json").read_bytes() for d in directories]
    config = tmp_path / "config.json"
    save(
        config,
        {
            "schema_version": 1,
            "kind": "fair_uk_all_metrics",
            "collector": collector,
            "intervention": {
                "id": "fixture",
                "method": "fixture",
                "training_language": "uk",
                "target_values": {},
            },
            "required_tasks": [pair["task"]],
            "pairs": [pair],
            "rules": {"bootstrap": 0},
        },
    )
    report = all_metrics.analyze(config, tmp_path / "result")
    scopes = report["pairs"][0]["comparison"]["association"]["scopes"]
    assert (
        next(r for r in scopes if r["scope"] == "component_macro")["forget_efficacy"]
        == 50
    )
    assert before_bytes == [(d / "run.json").read_bytes() for d in directories]
    assert (
        report["task_status"][pair["task"]]["collector_code_identity"]
        == code_identity()
    )
    assert next(m for m in report["metrics"] if m["id"] == "M3a")["status"] == "pending"
    assert len(report["metrics"]) == 18 and report["model_requests"] == 0


def test_development_manifest_cannot_masquerade_as_evaluation(frozen_pair, tmp_path):
    pair, collector, rows, source, bundle, directories = frozen_pair
    for row in rows:
        row["split"] = "development"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec = json.loads(bundle.read_text())["datasets"][pair["task"]]
    spec["sha256"] = all_metrics.digest(source)
    save(bundle, {"schema_version": 1, "datasets": {pair["task"]: spec}})
    spec = dataset_spec(pair["task"], bundle)
    for directory in directories:
        manifest = json.loads((directory / "run.json").read_text())
        manifest["dataset"] = spec
        save(directory / "run.json", manifest)
    output = tmp_path / "export"
    output.mkdir()
    with pytest.raises(ValueError, match="Frozen collector validation failed"):
        all_metrics.export_pair(pair, collector, tmp_path, output)
    assert "study phase" in (output / "collector_export.log").read_text()


def test_frozen_collector_source_mismatch_rejected(frozen_pair, tmp_path):
    pair, collector, _, _, _, directories = frozen_pair
    manifest = json.loads((directories[1] / "run.json").read_text())
    manifest["code_identity"] = {"other": "b" * 64}
    save(directories[1] / "run.json", manifest)
    output = tmp_path / "export"
    output.mkdir()
    with pytest.raises(ValueError, match="Frozen collector validation failed"):
        all_metrics.export_pair(pair, collector, tmp_path, output)
    assert "collector code differs" in (output / "collector_export.log").read_text()


@pytest.fixture
def exposure_files(tmp_path):
    manifest = tmp_path / "data_manifest.json"
    save(
        manifest,
        {
            "reviewed_for_gradients": True,
            "human_validated": False,
            "permitted_gradient_ids": ["train"],
        },
    )
    streams = {}
    for name in ("forget_train", "correct_train", "retain_train"):
        path = tmp_path / (name + ".jsonl")
        save(path, {"id": "view", "source_item_id": "train", "family_id": "family"})
        streams[name] = {"filename": path.name, "sha256": all_metrics.digest(path)}
    binding = tmp_path / "training_binding.json"
    save(
        binding,
        {
            "language": "uk",
            "deployment_binding": {
                "source_review_manifest_sha256": all_metrics.digest(manifest)
            },
            "streams": streams,
        },
    )
    recipe = tmp_path / "recipe.json"
    save(
        recipe,
        {"method": "wga_kl", "data_manifest_sha256": all_metrics.digest(binding)},
    )
    execution = tmp_path / "execution.json"
    save(
        execution,
        {
            "methods": {
                "wga_kl": {
                    "status": "trained_checkpoint_saved",
                    "returncode": 0,
                    "steps": 1,
                }
            }
        },
    )
    directory = tmp_path / "wga_kl"
    directory.mkdir()
    event = {
        "step": 0,
        "role": "forget",
        "id": "view",
        "source_item_id": "train",
        "family_id": "family",
        "weight": 1.0,
        "valid_loss_tokens": 2,
        "effective_input": {"input_ids": [12, 34], "completion_token_indices": [0, 1]},
    }
    log = directory / "exposure.jsonl"
    save(log, event)
    receipt_path = directory / "intervention.json"
    receipt = {
        "config_sha256": all_metrics.digest(recipe),
        "exposure_log_sha256": all_metrics.digest(log),
        "exposure_valid_tokens": {"forget": 2},
    }
    save(receipt_path, receipt)
    evidence = tmp_path / "evidence.json"
    save(
        evidence,
        {
            "files": {
                "wga_kl/exposure.jsonl": all_metrics.digest(log),
                "wga_kl/intervention.json": all_metrics.digest(receipt_path),
            }
        },
    )
    config = {
        "execution": str(execution),
        "data_manifest": str(manifest),
        "training_binding": str(binding),
        "method_directory": str(directory),
        "recipe": str(recipe),
        "evidence_manifest": str(evidence),
    }
    return config, event, log, receipt_path, evidence


def test_current_exposure_preserves_human_review_status(exposure_files, tmp_path):
    config, _, _, _, _ = exposure_files
    result = all_metrics.exposure_summary(config, {"method": "wga_kl"}, tmp_path)
    assert result["status"] == "available" and result["human_validated"] is False
    assert result["summary"]["forget"]["valid_loss_tokens"] == 2
    assert result["outcome_correlation"] is None


@pytest.mark.parametrize("change", ["mask", "evaluation_id"])
def test_exposure_rejects_wrong_masks_and_nontraining_ids(
    exposure_files, tmp_path, change
):
    config, event, log, receipt_path, evidence = exposure_files
    if change == "mask":
        event["effective_input"]["completion_token_indices"] = [0]
    else:
        event["source_item_id"] = "evaluation"
    save(log, event)
    receipt = json.loads(receipt_path.read_text())
    receipt["exposure_log_sha256"] = all_metrics.digest(log)
    save(receipt_path, receipt)
    save(
        evidence,
        {
            "files": {
                "wga_kl/exposure.jsonl": all_metrics.digest(log),
                "wga_kl/intervention.json": all_metrics.digest(receipt_path),
            }
        },
    )
    with pytest.raises(ValueError, match="completion mask|non-training"):
        all_metrics.exposure_summary(config, {"method": "wga_kl"}, tmp_path)


@pytest.mark.parametrize("change", [None, "wrong_arm", "post_bytes", "base_bytes"])
def test_pair_is_bound_to_actual_training_receipt(exposure_files, tmp_path, change):
    config, _, _, receipt_path, _ = exposure_files
    recipe = json.loads(Path(config["recipe"]).read_text())
    recipe["model_path"] = "/models/frozen-base"
    save(Path(config["recipe"]), recipe)
    execution = json.loads(Path(config["execution"]).read_text())
    execution["methods"]["wga_kl"]["checkpoint"] = "/training/wga-checkpoint"
    save(Path(config["execution"]), execution)
    receipt = json.loads(receipt_path.read_text())
    receipt["merged_checkpoint_files"] = [
        {"path": "model.safetensors", "sha256": "b" * 64}
    ]
    receipt["source_model_files"] = [
        {"path": "model.safetensors", "sha256": "a" * 64},
        {"path": ".cache/unneeded", "sha256": "c" * 64},
    ]
    save(receipt_path, receipt)
    before = {
        "model_identity": {
            "parameters": {"pretrained": "/models/frozen-base"},
            "local_files_sha256": {"model.safetensors": "a" * 64},
        }
    }
    after = {
        "model_identity": {
            "parameters": {"pretrained": "/training/wga-checkpoint"},
            "local_files_sha256": {"model.safetensors": "b" * 64},
        }
    }
    if change == "wrong_arm":
        after["model_identity"]["parameters"]["pretrained"] = "/training/npo-checkpoint"
    elif change == "post_bytes":
        after["model_identity"]["local_files_sha256"]["model.safetensors"] = "c" * 64
    elif change == "base_bytes":
        before["model_identity"]["local_files_sha256"]["model.safetensors"] = "c" * 64
    pair = {"provenance": {"before": {"run": before}, "after": {"run": after}}}
    if change:
        with pytest.raises(ValueError, match="checkpoint (path|bytes)"):
            all_metrics.bind_training_pair(pair, config, {"method": "wga_kl"}, tmp_path)
    else:
        bound = all_metrics.bind_training_pair(
            pair, config, {"method": "wga_kl"}, tmp_path
        )
        assert bound["status"] == "bound_to_actual_training_arm"
        assert (
            bound["base_binding"]
            == "all_native_identity_file_hashes_match_trainer_source"
        )
