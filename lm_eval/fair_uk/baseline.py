"""Preserved pure source-policy helpers; no study collector or capability CLI.

Functions are copied from Fair-UK 710ae774a327e948e8957879dcc1ce071bbc5eae.
"""

import re
from pathlib import Path

from lm_eval.fair_uk.contracts import file_sha256 as digest


def resolve(config_path, value):
    return (Path(config_path).resolve().parent / value).resolve()


def measurement_policy(config, specification, split):
    """Keep unpublished preparation visible without authorizing inference."""
    eligibility = specification.get("measurement_eligibility", "active")
    if eligibility not in ("active", "preparation_only"):
        raise ValueError("Unknown baseline measurement eligibility")
    reason = specification.get("measurement_eligibility_reason")
    if eligibility == "preparation_only" and (
        not isinstance(reason, str) or not reason
    ):
        raise ValueError("Preparation-only measurements require a blocking reason")
    scope = config.get("approved_dataset_scope")
    if scope is None:
        return eligibility, reason
    if config.get("capability_manifest"):
        raise ValueError(
            "External capability manifests are outside the approved dataset scope"
        )
    if (
        scope.get("namespace") != "FairForget"
        or scope.get("published_only", True) is not True
    ):
        raise ValueError(
            "The approved dataset scope requires published FairForget sources"
        )
    provenance = specification.get("source_provenance")
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("Scoped baseline measurements require source provenance")
    role = specification.get("source_role")
    if role not in ("training_signal", "development", "evaluation"):
        raise ValueError("Unknown baseline source role")
    expected_role = {
        "train": "training_signal",
        "train_signal": "training_signal",
        "development": "development",
        "evaluation": "evaluation",
        "deferred": "evaluation",
    }.get(split)
    if role != expected_role:
        raise ValueError("Baseline source role differs from its measurement phase")
    if (
        not isinstance(specification.get("measurement_purpose"), str)
        or not specification["measurement_purpose"]
    ):
        raise ValueError("Scoped baseline measurements require a measurement purpose")
    training = set(scope.get("training_repositories", []))
    evaluation = set(scope.get("evaluation_only_repositories", []))
    preparation = set(scope.get("preparation_only_repositories", []))
    excluded = set(scope.get("excluded_experimental_repositories", [])) | {
        "FairForget/LENS-WarBias"
    }
    revisions = scope.get("source_revisions", {})
    permitted = (
        training
        if role in ("training_signal", "development")
        else training | evaluation
    )
    for source in provenance:
        if not isinstance(source, dict):
            raise TypeError("Invalid baseline source provenance")
        repository = source.get("repo_id", "")
        if (
            not repository.startswith("FairForget/")
            or repository in excluded
            or "crows" in repository.lower()
        ):
            raise ValueError("Baseline source is outside the approved dataset scope")
        if (
            not isinstance(source.get("path"), str)
            or not source["path"]
            or not re.fullmatch(r"[0-9a-f]{64}", source.get("sha256", ""))
            or not isinstance(source.get("published"), bool)
        ):
            raise ValueError("Invalid baseline source provenance")
        if source["published"]:
            revision = source.get("revision", "")
            if repository not in permitted:
                raise ValueError(
                    "Baseline source role violates the approved dataset scope"
                )
            if (
                not re.fullmatch(r"[0-9a-f]{40}", revision or "")
                or revisions.get(repository) != revision
            ):
                raise ValueError(
                    "Published baseline source revision differs from the approved pin"
                )
            if repository in preparation:
                raise ValueError(
                    "Preparation-only repositories cannot become active releases"
                )
        elif (
            eligibility != "preparation_only"
            or repository not in preparation
            or source.get("revision") is not None
        ):
            raise ValueError(
                "Unpublished baseline sources must remain preparation-only"
            )
        else:
            local_source = Path(source["path"])
            if not local_source.is_absolute():
                raise ValueError(
                    "Unpublished preparation sources require absolute local paths"
                )
            if digest(local_source) != source["sha256"]:
                raise ValueError("Unpublished preparation source bytes changed")
    return eligibility, reason
