"""Saved-only reporting of all 18 study metric commitments.

Frozen collectors validate their own saved runs in isolated subprocesses. This
analysis layer records a separate identity and never rewrites collector evidence.
"""

import hashlib
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from lm_eval.fair_uk.analysis_inputs import (
    enrich_rows,
    load_annotations,
    severity_report,
)
from lm_eval.fair_uk.contracts import write_json
from lm_eval.fair_uk.data import family
from lm_eval.fair_uk.intervention_metrics import compare_predictions
from lm_eval.fair_uk.metrics import score_all
from lm_eval.fair_uk.provenance import code_identity
from lm_eval.fair_uk.transfer_metrics import transfer_report


SOURCE_PIN = "710ae774a327e948e8957879dcc1ce071bbc5eae"
METRICS = (
    "forget_efficacy",
    "NNR",
    "retain_utility",
    "LMS",
    "ICAT",
    "M1",
    "M1b",
    "M2",
    "M3a",
    "M3b",
    "M3c",
    "M3d",
    "RI",
    "M4a",
    "M4b",
    "BBQ",
    "training_imbalance",
    "severity",
)
DEPENDENCIES = {
    "forget_efficacy": "Complete matched triplet predictions; native weighting and target scope.",
    "NNR": "Frozen model/task/language neutrality calibration and eligible baseline cohort.",
    "retain_utility": "Matched in-scope correctness controls; benign judgments and practical utility limits remain separate.",
    "LMS": "Complete matched triplet scores, including unrelated candidates.",
    "ICAT": "Same native scope's SS and LMS; secondary composite only.",
    "M1": "One quality outcome, matched common content and an explicit independent-support rule.",
    "M1b": "Qualified M1 panel and strictly positive maximum group quality.",
    "M2": "Qualified common quality panel and paired group losses.",
    "M3a": "Declared actor target and matched actor cells; no target is inferred.",
    "M3b": "Declared reviewed topic target and reviewed near/far mapping.",
    "M3c": "Declared gender/age target and matched profile support.",
    "M3d": "Single training language and aligned opposite-language before/after outputs.",
    "RI": "Declared target/harm axis, practical improvement minimum and positive paired target CI.",
    "M4a": "Benign raw responses, response-bound judgments and human calibration; no judge is selected implicitly.",
    "M4b": "Sufficient-evidence QA, factual favorable/adverse assignments and unknown/accuracy/invalid outcomes.",
    "BBQ": "Native BBQ outputs and separately labelled WarBias QA with validated semantic roles/denominators.",
    "training_imbalance": "Actual source-bound valid-token events; outcome association needs matched outcomes/groups.",
    "severity": "Reviewed high/standard annotations bound to adapted dataset bytes; descriptive only.",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(root, value):
    return str((root / value).resolve()) if value is not None else None


def rules(value):
    result = {
        "tau_by_task": {},
        "min_independent_support": None,
        "min_target_improvement": None,
        "bootstrap": 1000,
        "seed": 42,
        **value,
    }
    for field in ("bootstrap", "seed"):
        require(
            type(result[field]) is int and result[field] >= 0,
            field + " must be nonnegative",
        )
    minimum = result["min_independent_support"]
    require(
        minimum is None or type(minimum) is int and minimum >= 2,
        "Support must be explicit and >=2",
    )
    practical = result["min_target_improvement"]
    require(
        practical is None
        or type(practical) in (int, float)
        and math.isfinite(practical)
        and practical >= 0,
        "Invalid practical improvement minimum",
    )
    require(isinstance(result["tau_by_task"], dict), "tau_by_task must be a mapping")
    for threshold in result["tau_by_task"].values():
        require(
            threshold is None
            or type(threshold) in (int, float)
            and math.isfinite(threshold)
            and threshold >= 0,
            "Invalid neutrality threshold",
        )
    return result


def export_pair(pair, collector, root, output):
    """Run only frozen validation/scoring code, with inference and network disabled."""
    request = {**pair, "evaluator_root": _path(root, collector["evaluator_root"])}
    for key in (
        "before_run",
        "after_run",
        "dataset_bundle",
        "before_judgments",
        "after_judgments",
    ):
        request[key] = _path(root, pair.get(key))
    request_path = output / "export_request.json"
    export_path = output / "validated_pair.json"
    write_json(request_path, request)
    environment = {
        **os.environ,
        "PYTHONPATH": request["evaluator_root"],
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    command = [
        _path(root, collector["python"]),
        str(Path(__file__).with_name("collector_export.py")),
        "--request",
        str(request_path),
        "--output",
        str(export_path),
    ]
    with (output / "collector_export.log").open("x") as stream:
        # The explicit study manifest selects the local interpreter, without a shell.
        result = subprocess.run(  # noqa: S603
            command,
            cwd=request["evaluator_root"],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    require(
        result.returncode == 0,
        "Frozen collector validation failed; see "
        + str(output / "collector_export.log"),
    )
    exported = read(export_path)
    require(
        exported["kind"] == "collector_validated_saved_pair"
        and exported["task"] == pair["task"],
        "Unexpected frozen export identity",
    )
    return exported, {"path": str(export_path), "sha256": digest(export_path)}


def analyze_pair(exported, pair, analysis_rules, root):
    rows = exported["rows"]
    metadata, annotations = {}, {}
    if pair.get("annotations"):
        annotations = load_annotations(
            _path(root, pair["annotations"]),
            exported["provenance"]["before"]["run"]["dataset"],
            rows,
        )
        metadata = {k: v for k, v in annotations.items() if v["reviewed"]}
    rows = enrich_rows(rows, annotations)
    threshold = analysis_rules["tau_by_task"].get(pair["task"])
    calibration = analysis_rules.get("neutrality_calibration", {}).get(pair["task"])
    # The preserved kernel accepts a number; the orchestration also requires its authority.
    if threshold is not None and not calibration:
        threshold = None
    elif threshold is not None:
        require(
            isinstance(calibration, dict) and calibration.get("calibration_id"),
            "NNR requires an identified frozen calibration record",
        )
        require(
            calibration.get("baseline_predictions_sha256")
            == exported["provenance"]["before"]["predictions_sha256"],
            "NNR calibration baseline predictions differ",
        )
        expected = {
            row["id"]
            for row in exported["before_scored"]
            if row["candidate_means"][0] - row["candidate_means"][1] > threshold
        }
        eligible = calibration.get("baseline_eligible_ids")
        require(
            isinstance(eligible, list)
            and len(eligible) == len(set(eligible))
            and set(eligible) == expected,
            "NNR frozen eligible cohort differs",
        )
    before, after = exported["before_predictions"], exported["after_predictions"]
    comparison = compare_predictions(
        rows,
        before,
        after,
        tau=threshold,
        min_independent_support=analysis_rules["min_independent_support"],
        source_metadata=metadata,
        bootstrap=analysis_rules["bootstrap"],
        seed=analysis_rules["seed"],
        answer_policy=exported["answer_policy"],
    )
    pre, post = [
        score_all(rows, predictions, exported["answer_policy"])
        for predictions in (before, after)
    ]
    return {
        "task": pair["task"],
        "phase": pair.get("phase", "evaluation"),
        "comparison": comparison,
        "severity": severity_report(rows, pre, post),
        "annotations_sha256": digest(_path(root, pair["annotations"]))
        if pair.get("annotations")
        else None,
        "neutrality_calibration": calibration if threshold is not None else None,
        "answer_policy": exported["answer_policy"],
    }, {"task": pair["task"], "rows": rows, "before_scored": pre, "after_scored": post}


def exposure_summary(config, intervention, root):
    """Validate current trainer events against the new reviewed source binding."""
    if not config:
        return {
            "status": "pending",
            "reason": "missing_actual_training_events",
            "summary": None,
        }
    paths = {
        key: Path(_path(root, config[key]))
        for key in (
            "execution",
            "data_manifest",
            "training_binding",
            "method_directory",
            "recipe",
            "evidence_manifest",
        )
    }
    manifest, binding, execution = [
        read(paths[key]) for key in ("data_manifest", "training_binding", "execution")
    ]
    method = intervention["method"]
    arm = execution.get("methods", {}).get(method, {})
    require(
        arm.get("status") == "trained_checkpoint_saved" and arm.get("returncode") == 0,
        "Exposure requires a completed actual training arm",
    )
    require(
        binding.get("deployment_binding", {}).get("source_review_manifest_sha256")
        == digest(paths["data_manifest"]),
        "Trainer binding differs from original reviewed manifest",
    )
    require(
        manifest.get("reviewed_for_gradients") is True,
        "Training selections were not reviewed",
    )
    language = binding.get("language")
    require(
        language in ("uk", "en")
        and language == intervention.get("training_language", language),
        "Actual exposure language differs from the intervention",
    )
    directory = paths["method_directory"]
    log = directory / "exposure.jsonl"
    receipt_path = directory / ("run.json" if method == "rmu" else "intervention.json")
    receipt = read(receipt_path)
    recipe = read(paths["recipe"])
    evidence = read(paths["evidence_manifest"])
    require(
        recipe.get("method") == method
        and recipe.get("data_manifest_sha256") == digest(paths["training_binding"]),
        "Recipe differs from the method or reviewed trainer binding",
    )
    recorded_config = receipt.get(
        "config_sha256", receipt.get("provenance", {}).get("config_sha256")
    )
    require(
        recorded_config == digest(paths["recipe"]),
        "Training receipt differs from the frozen recipe",
    )
    for filename in ("exposure.jsonl", receipt_path.name):
        require(
            evidence.get("files", {}).get(method + "/" + filename)
            == digest(directory / filename),
            "Actual training evidence hash differs: " + filename,
        )
    if method != "rmu":
        require(
            receipt.get("exposure_log_sha256") == digest(log),
            "Exposure log differs from training receipt",
        )
    else:
        require(
            receipt["provenance"]["data_manifest"]["sha256"]
            == digest(paths["training_binding"]),
            "RMU source binding differs",
        )
    stream_names = {
        "forget": "forget_train",
        "positive_acquisition": "forget_train",
        "correction": "correct_train",
        "preservation": "retain_train",
    }
    inventories = {}
    for name in set(stream_names.values()):
        entry = binding["streams"][name]
        source = paths["training_binding"].parent / entry["filename"]
        require(digest(source) == entry["sha256"], "Reviewed stream bytes changed")
        inventories[name] = {
            v["id"]: v
            for v in [json.loads(line) for line in source.read_text().splitlines()]
        }
    events = [json.loads(line) for line in log.read_text().splitlines() if line]
    cells = defaultdict(
        lambda: {
            "events": 0,
            "valid_loss_tokens": 0,
            "weighted_loss_tokens": 0.0,
            "ids": set(),
            "source_ids": set(),
            "families": set(),
            "steps": set(),
        }
    )
    seen, derived_views = set(), {}
    for event in events:
        role = event["role"]
        require(role in stream_names, "Unknown actual exposure role")
        count = event["valid_loss_tokens"]
        require(
            type(count) is int and count > 0 and type(event["step"]) is int,
            "Exposure requires actual valid-token counts and step IDs",
        )
        effective = event["effective_input"]
        indices = effective["completion_token_indices"]
        require(
            len(indices) == len(set(indices)) == count
            and all(
                type(i) is int and 0 <= i < len(effective["input_ids"]) for i in indices
            ),
            "Valid-token count differs from effective completion mask",
        )
        key = (event["step"], role, event["id"])
        require(key not in seen, "Duplicate exposure event")
        seen.add(key)
        source_id = event["source_item_id"]
        require(
            source_id in manifest["permitted_gradient_ids"],
            "Exposure includes a non-training source ID",
        )
        inventory = inventories[stream_names[role]]
        # Adversarial views may add a suffix; source identity and family remain fixed.
        matching = [
            v
            for v in inventory.values()
            if v["source_item_id"] == source_id and v["family_id"] == event["family_id"]
        ]
        require(matching, "Exposure is outside the reviewed stream/family")
        if event["id"] not in inventory:
            alternatives = inventories["correct_train"]
            require(
                method == "biasunlearn"
                and role == "forget"
                and event["id"] in alternatives
                and alternatives[event["id"]]["source_item_id"] == source_id,
                "Exposure view is not an approved primary or corrective-derived view",
            )
            derived_views[event["id"]] = {
                "declared_role": role,
                "source_stream": "correct_train",
                "source_item_id": source_id,
                "family_id": event["family_id"],
            }
        weight = event["weight"]
        require(
            type(weight) in (int, float) and math.isfinite(weight) and weight >= 0,
            "Invalid exposure weight",
        )
        cell = cells[role]
        cell["events"] += 1
        cell["valid_loss_tokens"] += count
        cell["weighted_loss_tokens"] += weight * count
        for field, value in (
            ("ids", event["id"]),
            ("source_ids", source_id),
            ("families", event["family_id"]),
            ("steps", event["step"]),
        ):
            cell[field].add(value)
    summary = {}
    for role, cell in sorted(cells.items()):
        summary[role] = {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in cell.items()
        }
    require(
        not receipt.get("exposure_valid_tokens")
        or receipt["exposure_valid_tokens"]
        == {role: cell["valid_loss_tokens"] for role, cell in cells.items()},
        "Exposure totals differ from receipt",
    )
    return {
        "status": "available",
        "language": language,
        "summary": summary,
        "derived_views": derived_views,
        "optimizer_steps": arm.get("steps"),
        "source_binding": {
            key: {"path": str(path), "sha256": digest(path)}
            for key, path in paths.items()
            if key != "method_directory"
        },
        "log": {"path": str(log), "sha256": digest(log)},
        "receipt": {"path": str(receipt_path), "sha256": digest(receipt_path)},
        "human_validated": manifest.get("human_validated"),
        "outcome_correlation": None,
        "limitations": "Actual exposure summaries only; no group causality or exposure-outcome correlation is inferred. Weighted tokens use logged scalar stream coefficients, not per-token confidence weights or gradient magnitudes. Positive acquisition is separate from forgetting and null task-vector selection. Corrective-derived adversarial views keep their original source identity.",
    }


def bind_training_pair(exported, exposure_config, intervention, root):
    """Reject a valid pair from a different trained arm or base checkpoint."""
    if not exposure_config:
        return {
            "status": "not_supplied",
            "reason": "No actual training evidence supplied",
        }
    execution = read(_path(root, exposure_config["execution"]))
    recipe = read(_path(root, exposure_config["recipe"]))
    method = intervention["method"]
    arm = execution["methods"][method]
    receipt_path = Path(_path(root, exposure_config["method_directory"])) / (
        "run.json" if method == "rmu" else "intervention.json"
    )
    receipt = read(receipt_path)
    before, after = [
        exported["provenance"][phase]["run"]["model_identity"]
        for phase in ("before", "after")
    ]
    require(
        before.get("parameters", {}).get("pretrained") == recipe["model_path"],
        "Before checkpoint path differs from the frozen training base",
    )
    require(
        after.get("parameters", {}).get("pretrained") == arm["checkpoint"],
        "After checkpoint path differs from the actual trained arm",
    )
    expected = {
        item["path"]: item["sha256"]
        for item in receipt[
            "checkpoint_identity" if method == "rmu" else "merged_checkpoint_files"
        ]
    }
    require(
        bool(expected) and after.get("local_files_sha256") == expected,
        "After checkpoint bytes differ from the actual training receipt",
    )
    if method == "rmu":
        base_status = "path_and_frozen_recipe_pin_only_source_base_hashes_not_recorded"
    else:
        source = {
            item["path"]: item["sha256"] for item in receipt["source_model_files"]
        }
        observed = before.get("local_files_sha256")
        require(
            isinstance(observed, dict)
            and bool(observed)
            and all(
                source.get(name) == checksum for name, checksum in observed.items()
            ),
            "Before checkpoint bytes differ from the actual trainer source",
        )
        base_status = "all_native_identity_file_hashes_match_trainer_source"
    return {
        "status": "bound_to_actual_training_arm",
        "method": method,
        "receipt_sha256": digest(receipt_path),
        "recipe_sha256": digest(_path(root, exposure_config["recipe"])),
        "post_checkpoint_files": len(expected),
        "base_binding": base_status,
        "base_model_id": recipe.get("model_id"),
        "base_revision": recipe.get("model_revision"),
        "null_task_vector_allowed": method == "task_vector",
    }


def metric_inventory(
    comparisons, transfer, exposure, required_tasks, tasks, rules_record
):
    """Keep every commitment visible; missing dependencies are never zero effects."""
    kinds = {family(task) for task in tasks}
    triplets = bool(kinds & {"stereoset_uk", "warbias_triplets"})
    quality = [
        row for pair in comparisons for row in pair["comparison"]["group_quality"]
    ]
    qualified = any(row["status"] == "qualified_common_panel" for row in quality)
    entries = []
    for metric in METRICS:
        status, reason, references = "pending", DEPENDENCIES[metric], []
        if metric in ("forget_efficacy", "LMS", "ICAT") and triplets:
            status, reason = "available", None
            references = [
                p["task"] + "/analysis.json#/comparison/association"
                for p in comparisons
                if family(p["task"]) in ("stereoset_uk", "warbias_triplets")
            ]
        elif metric == "retain_utility" and comparisons:
            status, reason = (
                "partial",
                "In-scope native outcomes only; general utility and practical tolerances are not qualified.",
            )
            references = [
                p["task"] + "/analysis.json#/comparison/native_changes"
                for p in comparisons
            ]
        elif metric == "NNR" and triplets:
            scopes = [
                scope["nnr"]
                for p in comparisons
                for scope in p["comparison"]["association"].get("scopes", [])
            ]
            eligible = [
                scope
                for scope in scopes
                if scope.get("status") == "available" and scope.get("nnr") is not None
            ]
            if eligible:
                status = "available" if len(eligible) == len(scopes) else "partial"
                reason = (
                    None
                    if status == "available"
                    else "Only scopes with a frozen calibrated tau and nonempty eligible baseline cohort are available."
                )
                references = [
                    p["task"] + "/analysis.json#/comparison/association/scopes"
                    for p in comparisons
                    if p["neutrality_calibration"]
                ]
            elif scopes and all(
                scope.get("status") == "undefined_no_baseline_eligible_items"
                for scope in scopes
            ):
                reason = "No baseline-eligible items; NNR is undefined, not zero."
        elif metric in ("M1", "M1b", "M2") and qualified:
            complete = all(
                row["status"] == "qualified_common_panel"
                and (metric != "M1b" or row["m1b_post"] is not None)
                for row in quality
            )
            status = "available" if complete else "partial"
            reason = (
                None
                if complete
                else "Only explicitly qualified common panels are available; sparse, unmatched or zero-maximum ratio scopes remain null."
            )
            references = [
                p["task"] + "/analysis.json#/comparison/group_quality"
                for p in comparisons
            ]
        elif metric in ("M3a", "M3b", "M3c", "M3d"):
            section = transfer.get(metric, [])
            records = (
                [row for values in section.values() for row in values]
                if isinstance(section, dict)
                else section
            )
            eligible = [
                row
                for row in records
                if row.get("status") == "available" and row.get("target_values")
            ]
            if eligible:
                status = "available" if len(eligible) == len(records) else "partial"
                reason = (
                    None
                    if status == "available"
                    else "Only supported target-qualified contexts are available; every other context retains its reason."
                )
                references = ["all_metrics.json#/transfer/" + metric]
        elif metric == "RI":
            records = [
                *transfer.get("M3a", []),
                *transfer.get("M3b", []),
                *transfer.get("M3d", []),
            ]
            records += [
                row for values in transfer.get("M3c", {}).values() for row in values
            ]
            eligible = [
                row
                for row in records
                if row.get("redistribution", {}).get("ratio") is not None
            ]
            if eligible:
                status = "available" if len(eligible) == len(records) else "partial"
                reason = (
                    None
                    if status == "available"
                    else "Ratios are available only on declared stable axes; other axes retain null ratios and their raw changes."
                )
                references = ["all_metrics.json#/transfer"]
        elif metric == "M4a" and "warbias_benign" in kinds:
            status = "partial"
            reason = "Responses collected; calibrated judgment coverage must be established separately."
            references = [
                p["task"] + "/analysis.json#/comparison/checkpoint_pre/benign"
                for p in comparisons
                if family(p["task"]) == "warbias_benign"
            ]
        elif metric == "M4b" and kinds & {"warbias", "bbq_uk"}:
            status, reason = (
                "partial",
                "QA caution is calculable; favorable/adverse factual slices require source-bound reviewed assignments.",
            )
            references = [
                p["task"] + "/analysis.json#/comparison/paired_profiles"
                for p in comparisons
                if family(p["task"]) in ("warbias", "bbq_uk")
            ]
        elif metric == "BBQ" and kinds & {"warbias", "bbq_uk"}:
            status, reason = "available", None
            undefined = [
                row
                for p in comparisons
                for row in p["comparison"]["native_changes"]
                if row.get("zero_denominator_pre") or row.get("zero_denominator_post")
            ]
            adaptation_undefined = [
                scope
                for p in comparisons
                for phase in ("checkpoint_pre", "checkpoint_post")
                for scope in p["comparison"][phase]["warbias_bbq"].get("scopes", [])
                if scope.get("status") != "available"
            ]
            if undefined or adaptation_undefined:
                status, reason = (
                    "partial",
                    "Scopes without non-unknown answer mass have undefined bias changes; native zero conventions are not neutrality.",
                )
            references = [
                p["task"] + "/analysis.json#/comparison/native_changes"
                for p in comparisons
                if family(p["task"]) in ("warbias", "bbq_uk")
            ]
            references += [
                p["task"] + "/analysis.json#/comparison/paired_profiles/warbias_bbq"
                for p in comparisons
                if family(p["task"]) == "warbias"
            ]
        elif metric == "training_imbalance" and exposure["status"] == "available":
            status, reason, references = (
                "partial",
                "Actual stream/family/token exposure only; outcome correlation and causal imbalance remain unavailable.",
                ["all_metrics.json#/training_exposure"],
            )
        elif metric == "severity" and any(
            p["severity"].get("coverage", {}).get("reviewed_severity_rows", 0)
            for p in comparisons
        ):
            status, reason = (
                "partial",
                "Reviewed strata are descriptive; no severity uncertainty is supplied.",
            )
            references = [
                p["task"] + "/analysis.json#/severity"
                for p in comparisons
                if p["severity"].get("coverage", {}).get("reviewed_severity_rows", 0)
            ]
        entries.append(
            {
                "id": metric,
                "status": status,
                "reason": reason,
                "references": references,
                "missing_tasks": sorted(set(required_tasks) - set(tasks)),
            }
        )
    return entries


def analyze(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config, root = read(config_path), config_path.parent
    require(
        config.get("schema_version") == 1
        and config.get("kind") == "fair_uk_all_metrics",
        "Unsupported all-metrics config",
    )
    require(not output.exists(), "Use a new analysis output directory")
    rules_record = rules(config.get("rules", {}))
    intervention = config["intervention"]
    require(
        isinstance(intervention.get("target_values", {}), dict),
        "target_values must be explicit",
    )
    required = config.get("required_tasks", [])
    require(
        isinstance(required, list) and len(required) == len(set(required)),
        "Duplicate required tasks",
    )
    require(
        all(
            isinstance(task, str) and re.fullmatch(r"[a-z0-9_]+", task)
            for task in required
        ),
        "Invalid task identifier",
    )
    pairs = config.get("pairs", [])
    tasks = [pair["task"] for pair in pairs]
    require(len(tasks) == len(set(tasks)), "Duplicate task pair")
    require(set(tasks) <= set(required), "Pair task is absent from required_tasks")
    inputs = [config_path]
    if pairs:
        inputs.append(Path(_path(root, config["collector"]["evaluator_root"])))
    for value in (config.get("exposure") or {}).values():
        inputs.append(Path(_path(root, value)))
    for pair in pairs:
        inputs.extend(
            Path(_path(root, pair[name]))
            for name in ("before_run", "after_run", "dataset_bundle")
        )
        inputs.extend(
            Path(_path(root, pair[name]))
            for name in ("annotations", "before_judgments", "after_judgments")
            if pair.get(name)
        )
    for source in inputs:
        require(
            source != output and source not in output.parents,
            "Output must remain outside saved inputs",
        )
    output.mkdir(parents=True)
    exposure = exposure_summary(config.get("exposure"), intervention, root)
    comparisons, transfer_pairs, task_status = [], [], {}
    for task in required:
        pair = next((pair for pair in pairs if pair["task"] == task), None)
        if task in config.get("task_blocks", {}):
            require(
                pair is None, "A source-policy-blocked task cannot have an active pair"
            )
            task_status[task] = {
                "status": "blocked",
                "reason": config["task_blocks"][task],
            }
            continue
        if pair is None:
            task_status[task] = {
                "status": "pending",
                "reason": "required_pair_not_supplied",
            }
            continue
        if not all(
            Path(_path(root, pair[name])).exists()
            for name in ("before_run", "after_run", "dataset_bundle")
        ):
            task_status[task] = {
                "status": "pending",
                "reason": "saved_native_input_missing",
            }
            continue
        directory = output / task
        directory.mkdir()
        # Invalid evidence is an error, not a silently missing result.
        exported, source = export_pair(pair, config["collector"], root, directory)
        training_binding = bind_training_pair(
            exported, config.get("exposure"), intervention, root
        )
        comparison, transfer_pair = analyze_pair(exported, pair, rules_record, root)
        comparison["training_binding"] = training_binding
        write_json(directory / "analysis.json", comparison)
        comparisons.append(comparison)
        if pair.get("phase", "evaluation") == "evaluation":
            transfer_pairs.append(transfer_pair)
        task_status[task] = {
            "status": "available",
            "phase": pair.get("phase", "evaluation"),
            "validated_export": source,
            "analysis_sha256": digest(directory / "analysis.json"),
            "collector_code_identity": exported["collector_code_identity"],
        }
    transfer = transfer_report(
        transfer_pairs,
        intervention,
        bootstrap=rules_record["bootstrap"],
        seed=rules_record["seed"],
        min_target_improvement=rules_record["min_target_improvement"],
    )
    completed = [
        task for task, record in task_status.items() if record["status"] == "available"
    ]
    report = {
        "schema_version": 1,
        "kind": "all_18_saved_study_metrics",
        "model_requests": 0,
        "config": {"path": str(config_path), "sha256": digest(config_path)},
        "historical_calculation_source_pin": SOURCE_PIN,
        "analysis_code_identity": code_identity(),
        "analysis_source_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in (
                "all_metrics.py",
                "collector_export.py",
                "intervention_metrics.py",
                "transfer_metrics.py",
                "analysis_inputs.py",
                "baseline.py",
                "study_metric_sources.json",
            )
        },
        "intervention": intervention,
        "rules": rules_record,
        "task_status": task_status,
        "pairs": comparisons,
        "transfer": transfer,
        "training_exposure": exposure,
        "metrics": metric_inventory(
            comparisons, transfer, exposure, required, completed, rules_record
        ),
        "limitations": "One integration arm; no benchmark pooling, invented calibration, judge selection, external capability scope or causal exposure claim. Missing metrics remain pending or partial, not zero.",
    }
    write_json(output / "all_metrics.json", report)
    return report
