"""Frozen source, severity and checkpoint verification for saved baselines."""

import hashlib
import json
import math
from pathlib import Path

from lm_eval.fair_uk.warbias.baseline_summary import case_id


SCORING_SETTINGS = {
    "batch_size": "8",
    "chunk_size": 128,
    "max_output_tokens": 512,
    "generation_prompt_format": "native-chat",
    "answer_policy": "abc_option_text_v2",
    "bootstrap": 1000,
    "seed": 42,
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_baseline_run(
    directory,
    task,
    bundle,
    parameters,
    expected_identity,
    *,
    smoke=False,
    collection_settings=None,
):
    """Verify complete native outputs against their frozen source and checkpoint.

    This checks exact checkpoint files, model options, evaluator/scorer identity,
    selected IDs, answer policy and decoding settings. ``smoke`` selects one
    component per source stratum; full baseline analysis requires all rows.
    """
    from lm_eval.fair_uk.data import dataset_spec, family, load, select_clusters
    from lm_eval.fair_uk.provenance import code_identity
    from lm_eval.fair_uk.runs import load_verified_run
    from lm_eval.fair_uk.suite import verify_result

    settings = SCORING_SETTINGS if collection_settings is None else collection_settings
    if set(settings) != set(SCORING_SETTINGS):
        raise ValueError("Declare all collection settings")
    directory, bundle = Path(directory), Path(bundle)
    run, report = read(directory / "run.json"), read(directory / "report.json")
    identity = run.get("model_identity") or {}
    hashes = expected_identity.get("local_files_sha256")
    if (
        not isinstance(hashes, dict)
        or not hashes
        or identity.get("local_files_sha256") != hashes
    ):
        raise ValueError("Run checkpoint files differ from pinned weights or assets")
    if identity.get("parameters") != parameters:
        raise ValueError("Run model parameters differ")
    if run.get("code_identity") != code_identity():
        raise ValueError("Run evaluator identity differs")
    limit = 1 if smoke else None
    if (
        run.get("model_backend") != "vllm"
        or run.get("seed") != settings["seed"]
        or run.get("batch_size") != settings["batch_size"]
        or run.get("limit_clusters_per_stratum") != limit
    ):
        raise ValueError("Run inference settings or source selection differ")
    generation = (
        {
            "max_output_tokens": settings["max_output_tokens"],
            "temperature": 0.0,
            "do_sample": False,
            "prompt_format": settings["generation_prompt_format"],
        }
        if family(task) == "warbias"
        else None
    )
    if run.get("generation_config") != generation:
        raise ValueError("Run generation settings differ")
    if family(task) == "warbias" and any(
        record.get("scoring_policy") != settings["answer_policy"]
        for record in (run, report)
    ):
        raise ValueError("QA answer scoring policy differs")
    if any(
        report.get("bootstrap", {}).get(key) != value
        for key, value in {
            "replicates": settings["bootstrap"],
            "seed": settings["seed"],
        }.items()
    ):
        raise ValueError("Native report bootstrap settings differ")
    selected = select_clusters(load(task, dataset_bundle=bundle), limit)
    verify_result(
        directory,
        task,
        dataset_spec(task, bundle),
        selected,
        identity,
        settings["generation_prompt_format"],
    )
    return load_verified_run(
        directory,
        answer_policy=settings["answer_policy"],
        dataset_bundle=bundle,
    )


def _release_inputs(release):
    from lm_eval.fair_uk.data import family, load

    release = Path(release).resolve()
    bundle = release / "inputs/manifest.json"
    manifest, profiles = read(bundle), read(release / "models.json")
    tasks = manifest["datasets"]
    expected = {
        (kind, language)
        for kind in ("warbias_triplets", "warbias")
        for language in ("uk", "en")
    }
    if (
        len(tasks) != 4
        or {(family(task), spec["language"]) for task, spec in tasks.items()}
        != expected
    ):
        raise ValueError("Select exactly the bilingual triplet and QA tasks")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Declare the model profiles")
    if any(
        not key or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in key)
        for key in profiles
    ):
        raise ValueError("Model IDs must be safe directory names")
    for record in (manifest.get("source") or {}).get("files", {}).values():
        path = (release / "inputs" / record["path"]).resolve()
        if not path.is_relative_to(release) or sha(path) != record["sha256"]:
            raise ValueError("Released source file differs from its binding")
    rows = {task: load(task, dataset_bundle=bundle) for task in tasks}
    severity_path = release / "inputs/severity/case_scores.jsonl"
    binding = read(release / "inputs/severity/binding.json")
    if sha(severity_path) != binding.get("case_scores_sha256"):
        raise ValueError("Severity case scores differ from their binding")
    if binding.get("protocol_path"):
        protocol = (severity_path.parent / binding["protocol_path"]).resolve()
        if not protocol.is_relative_to(severity_path.parent) or sha(
            protocol
        ) != binding.get("protocol_sha256"):
            raise ValueError("Severity protocol differs from its binding")
    severity = {}
    for row in map(json.loads, severity_path.read_text().splitlines()):
        score, level = row["severity_mean_score"], row["severity_level"]
        if (
            row["case_id"] in severity
            or type(level) is not int
            or level not in range(1, 6)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 100
            or level != min(5, int(score / 20) + 1)
        ):
            raise ValueError(
                "Severity cases require unique IDs and valid mean/level values"
            )
        severity[row["case_id"]] = row
    if set(severity) != {case_id(row) for values in rows.values() for row in values}:
        raise ValueError("Severity records must cover exactly the selected cases")
    return release, manifest, profiles, rows, severity, binding


def release_overview(release):
    """Validate frozen inputs and describe all model/task cells without inference."""
    from lm_eval.fair_uk.data import family

    release, manifest, profiles, rows, severity, binding = _release_inputs(release)
    cells = [
        {
            "model": model,
            "task": task,
            "language": manifest["datasets"][task]["language"],
            "format": "triplets" if family(task) == "warbias_triplets" else "qa",
            "rows": len(values),
            "cases": len({case_id(row) for row in values}),
        }
        for model in profiles
        for task, values in rows.items()
    ]
    counts = []
    for level in range(1, 6):
        row = {
            "severity_level": level,
            "cases": sum(
                record["severity_level"] == level for record in severity.values()
            ),
        }
        for task, values in rows.items():
            row[task] = sum(
                severity[case_id(value)]["severity_level"] == level for value in values
            )
        counts.append(row)
    return {
        "models": list(profiles),
        "cells": cells,
        "severity": {"counts": counts, "binding": binding},
        "source": manifest.get("source"),
        "validation": {
            "status": "verified",
            "bundle_sha256": sha(release / "inputs/manifest.json"),
            "models_sha256": sha(release / "models.json"),
            "severity_scores_sha256": sha(
                release / "inputs/severity/case_scores.jsonl"
            ),
        },
    }
