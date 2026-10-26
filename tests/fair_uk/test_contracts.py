"""Cross-repository receipts reject changed sources and held-out gradients."""

from copy import deepcopy

import pytest

from lm_eval.fair_uk.contracts import (
    canonical_sha256,
    file_sha256,
    validate_checkpoint,
    validate_data_release,
    validate_exposure,
    validate_intervention,
)


@pytest.fixture
def release(tmp_path):
    (tmp_path / "rows.jsonl").write_text('{"id":"a"}\n')
    return {
        "schema_version": 1,
        "kind": "data_release",
        "release_id": "fixture",
        "languages": ["lg1", "lg2"],
        "item_ids": ["a", "b"],
        "family_by_item": {"a": "family-a", "b": "family-b"},
        "split_by_family": {"family-a": "train", "family-b": "evaluation"},
        "item_sha256": {"lg1:a": "a" * 64},
        "sources": [{"repo": "fixture/data", "revision": "a" * 40}],
        "qualification": {"human_validated": False},
        "files": [
            {
                "path": "rows.jsonl",
                "sha256": file_sha256(tmp_path / "rows.jsonl"),
                "role": "source_rows",
            }
        ],
    }


def receipt(release):
    return {
        "schema_version": 1,
        "kind": "intervention",
        "intervention_id": "tiny",
        "method": "fixture",
        "before": {"uri": "fixture/base", "revision": "b" * 40},
        "after": {"uri": "after", "files_sha256": {"weights.bin": "c" * 64}},
        "data_release_sha256": canonical_sha256(release),
        "training": {
            "seed": 1,
            "language": "lg1",
            "streams": [{"name": "forget", "role": "suppression", "source_ids": ["a"]}],
        },
    }


def exposure(release):
    return {
        "schema_version": 1,
        "kind": "exposure",
        "intervention_id": "tiny",
        "data_release_sha256": canonical_sha256(release),
        "events": [
            {
                "event_id": "0:forget:a",
                "step": 0,
                "stream": "forget",
                "item_id": "a",
                "language": "lg1",
                "valid_loss_tokens": 2,
                "source_sha256": "a" * 64,
            }
        ],
    }


def test_release_checks_exact_bytes_and_qualification(release, tmp_path):
    assert validate_data_release(release, root=tmp_path) is release
    (tmp_path / "rows.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="digest"):
        validate_data_release(release, root=tmp_path)
    del release["qualification"]["human_validated"]
    with pytest.raises(ValueError, match="human-validation"):
        validate_data_release(release, verify_files=False)


@pytest.mark.parametrize(
    "path", ["../outside", "/absolute", "C:\\absolute", "a\\b", ".", "./"]
)
def test_portable_paths_cannot_escape(release, path):
    release["files"][0]["path"] = path
    with pytest.raises(ValueError, match="inside"):
        validate_data_release(release, verify_files=False)


def test_symlink_is_not_a_release_file(release, tmp_path):
    (tmp_path / "rows.jsonl").rename(tmp_path / "real.jsonl")
    (tmp_path / "rows.jsonl").symlink_to(tmp_path / "real.jsonl")
    with pytest.raises(ValueError, match="absent or escapes"):
        validate_data_release(release, root=tmp_path)


def test_family_allocation_and_deferred_exception(release):
    release["split_by_item"] = {"a": "deferred", "b": "evaluation"}
    validate_data_release(release, verify_files=False)
    release["split_by_item"]["b"] = "train"
    with pytest.raises(ValueError, match="family allocation"):
        validate_data_release(release, verify_files=False)


def test_mutable_checkpoint_rejected():
    with pytest.raises(ValueError, match="immutable Git"):
        validate_checkpoint({"uri": "model", "revision": "main"})


def test_receipt_and_real_event_bindings(release):
    intervention = receipt(release)
    events = exposure(release)
    validate_intervention(intervention)
    validate_exposure(events, release=release, intervention=intervention)
    events["events"][0]["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="content differs"):
        validate_exposure(events, release=release, intervention=intervention)


@pytest.mark.parametrize("split", ["evaluation", "development", "deferred"])
def test_held_out_items_cannot_receive_gradients(release, split):
    release["split_by_family"]["family-a"] = split
    with pytest.raises(ValueError, match="cannot receive gradients"):
        validate_exposure(exposure(release), release=release)


def test_changed_release_cannot_reuse_receipt(release):
    events = exposure(release)
    changed = deepcopy(release)
    changed["release_id"] = "other"
    with pytest.raises(ValueError, match="release identity"):
        validate_exposure(events, release=changed)


@pytest.mark.parametrize("tokens", [0, -1, True, 2.5])
def test_exposure_counts_are_actual_positive_integers(release, tokens):
    events = exposure(release)
    events["events"][0]["valid_loss_tokens"] = tokens
    with pytest.raises(ValueError, match="token counts"):
        validate_exposure(events, release=release)
