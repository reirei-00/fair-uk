"""Bind evaluated checkpoint identities to separately verified trainer evidence."""

from lm_eval.fair_uk.contracts import (
    canonical_sha256,
    validate_exposure,
    validate_intervention,
)


def checkpoint_matches(reference, identity):
    """Match immutable contents; checkpoint storage location can be relocated."""
    if reference.get("files_sha256") is not None:
        return reference["files_sha256"] == identity.get("local_files_sha256")
    return reference["uri"].removeprefix("hf://") == identity.get(
        "repository"
    ) and reference.get("revision") == identity.get("revision")


def bind_training_evidence(report, intervention, exposure, release):
    """Validate a fresh comparison before attaching method/exposure claims."""
    if report.get("kind") not in {
        "paired_native_checkpoint_evaluation",
        "paired_before_after_primary_warbias",
    }:
        raise ValueError("Trainer lineage requires a supported paired evaluation report")
    validate_intervention(intervention)
    validate_exposure(exposure, release=release, intervention=intervention)
    if intervention["training"].get("exposure_canonical_sha256") != canonical_sha256(
        exposure
    ):
        raise ValueError("Intervention must bind the exact exposure record")
    for state in ("before", "after"):
        if report["kind"] == "paired_native_checkpoint_evaluation":
            identities = [report["provenance"][state]["run"]["model_identity"]]
        else:
            identities = [
                entry["model_identity"]
                for language in report["sources"][state].values()
                for stream in language.values()
                for entry in stream
            ]
        if not identities or not all(
            checkpoint_matches(intervention[state], identity) for identity in identities
        ):
            raise ValueError(
                f"Evaluated {state} checkpoint differs from intervention receipt"
            )
    return {
        "status": "matched_trainer_artifacts",
        "intervention_id": intervention["intervention_id"],
        "method": intervention["method"],
        "purpose": intervention["training"].get("purpose", "not_declared"),
        "intervention_sha256": canonical_sha256(intervention),
        "exposure_sha256": canonical_sha256(exposure),
        "data_release_sha256": canonical_sha256(release),
        "actual_exposure_events": len(exposure["events"]),
        "valid_loss_tokens": sum(
            event["valid_loss_tokens"] for event in exposure["events"]
        ),
        "qualification": "Artifact matching verifies recorded lineage; model-specific training validation and human calibration remain separate evidence.",
    }
