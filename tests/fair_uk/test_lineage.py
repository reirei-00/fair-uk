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
