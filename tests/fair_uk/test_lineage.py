"""An observational checkpoint contrast needs exact trainer lineage for attribution."""

from copy import deepcopy

import pytest

from lm_eval.fair_uk.contracts import canonical_sha256
from lm_eval.fair_uk.lineage import bind_training_evidence


def artifacts():
    release = {
        "kind": "data_release",
        "schema_version": 1,
        "release_id": "fixture",
        "languages": ["lg1"],
        "item_ids": ["a"],
        "family_by_item": {"a": "family"},
        "split_by_family": {"family": "train"},
        "qualification": {"human_validated": False},
        "sources": [{"repo": "fixture", "revision": "a" * 40}],
        "files": [{"path": "rows.jsonl", "sha256": "b" * 64, "role": "source"}],
    }
    exposure = {
        "kind": "exposure",
        "schema_version": 1,
        "intervention_id": "test",
        "data_release_sha256": canonical_sha256(release),
        "events": [],
    }
    intervention = {
        "kind": "intervention",
        "schema_version": 1,
        "intervention_id": "test",
        "method": "fixture",
        "before": {"uri": "hf://fixture/base", "revision": "a" * 40},
        "after": {
            "uri": "relocated/checkpoint",
            "files_sha256": {"weights.bin": "d" * 64},
        },
        "data_release_sha256": canonical_sha256(release),
        "training": {
            "seed": 1,
            "language": "lg1",
            "purpose": "software_verification",
            "streams": [{"name": "forget", "role": "suppression", "source_ids": ["a"]}],
            "exposure_canonical_sha256": canonical_sha256(exposure),
        },
    }
    intervention["training"]["starting_checkpoint_verification"] = {
        "schema_version": 1,
        "method": "loaded_state_from_verified_checkpoint_bytes",
        "checkpoint_reference_sha256": canonical_sha256(intervention["before"]),
        "state_sha256": "e" * 64,
        "files_sha256": {"weights.bin": "f" * 64},
    }
    report = {
        "kind": "paired_native_checkpoint_evaluation",
        "provenance": {
            "before": {
                "run": {
                    "model_identity": {
                        "repository": "fixture/base",
                        "revision": "a" * 40,
                    }
                }
            },
            "after": {
                "run": {
                    "model_identity": {"local_files_sha256": {"weights.bin": "d" * 64}}
                }
            },
        },
    }
    return report, intervention, exposure, release


def test_matches_relocated_checkpoint_contents_and_keeps_fixture_status():
    bound = bind_training_evidence(*artifacts())
    assert bound["actual_exposure_events"] == 0
    assert bound["purpose"] == "software_verification"


@pytest.mark.parametrize("damage", ["checkpoint", "exposure", "release"])
def test_different_checkpoint_or_trainer_evidence_rejects(damage):
    report, intervention, exposure, release = deepcopy(artifacts())
    if damage == "checkpoint":
        report["provenance"]["after"]["run"]["model_identity"]["local_files_sha256"][
            "weights.bin"
        ] = "e" * 64
    elif damage == "exposure":
        exposure["extra"] = "changed"
    else:
        release["release_id"] = "changed"
    with pytest.raises(ValueError):
        bind_training_evidence(report, intervention, exposure, release)


def primary_artifacts():
    native, intervention, exposure, release = artifacts()
    release["languages"] = ["en"]
    intervention["training"]["language"] = "en"
    exposure["data_release_sha256"] = canonical_sha256(release)
    intervention["data_release_sha256"] = canonical_sha256(release)
    intervention["training"]["exposure_canonical_sha256"] = canonical_sha256(exposure)
    report = {
        "kind": "paired_before_after_primary_warbias",
        "sources": {
            state: {
                language: {
                    stream: [
                        {
                            "model_identity": native["provenance"][state]["run"][
                                "model_identity"
                            ]
                        }
                    ]
                    for stream in ("original", "neutral")
                }
                for language in ("uk", "en")
            }
            for state in ("before", "after")
        },
        "language_transfer": {
            "status": "available",
            "training_language": "en",
            "language_pair": {"lg1": "uk", "lg2": "en"},
            "training_slot": "lg2",
            "opposite_slot": "lg1",
        },
    }
    return report, intervention, exposure, release


def test_verified_receipt_binds_the_declared_transfer_arm():
    assert bind_training_evidence(*primary_artifacts())["training_language"] == "en"


@pytest.mark.parametrize(
    "damage", ["language", "training_slot", "opposite_slot", "missing_language"]
)
def test_mislabeled_language_transfer_cannot_receive_matched_attribution(damage):
    report, intervention, exposure, release = primary_artifacts()
    transfer = report["language_transfer"]
    if damage == "language":
        transfer["training_language"] = "uk"
    elif damage == "missing_language":
        transfer["training_language"] = None
    elif damage == "training_slot":
        transfer["training_slot"] = "lg1"
    else:
        transfer["opposite_slot"] = "lg2"
    with pytest.raises(ValueError, match="language"):
        bind_training_evidence(report, intervention, exposure, release)


def test_missing_transfer_language_keeps_the_effect_unavailable():
    report, intervention, exposure, release = primary_artifacts()
    report["language_transfer"] = {
        "status": "unavailable",
        "training_language": None,
        "reason": "missing_training_language",
    }
    bound = bind_training_evidence(report, intervention, exposure, release)
    assert bound["training_language"] == "en"
    assert report["language_transfer"]["status"] == "unavailable"


@pytest.mark.parametrize("damage", ["missing", "reference", "state", "files"])
def test_unverified_starting_weights_cannot_receive_method_attribution(damage):
    report, intervention, exposure, release = artifacts()
    evidence = intervention["training"]["starting_checkpoint_verification"]
    if damage == "missing":
        del intervention["training"]["starting_checkpoint_verification"]
    elif damage == "reference":
        evidence["checkpoint_reference_sha256"] = "0" * 64
    elif damage == "state":
        evidence["state_sha256"] = "not-a-fingerprint"
    else:
        evidence["files_sha256"] = {}
    with pytest.raises(ValueError):
        bind_training_evidence(report, intervention, exposure, release)
