"""Reviewed labels and actual-event arithmetic, without models or downloads."""

import json
from copy import deepcopy

import pytest

from lm_eval.fair_uk import analysis_inputs as inputs
from lm_eval.fair_uk.data import PROTOCOLS, adapt, digest
from lm_eval.fair_uk.metrics import score_all


@pytest.fixture
def dataset(tmp_path):
    rows = [
        {
            "id": str(i),
            "task": "stereoset_uk",
            "language": "uk",
            "protocol": PROTOCOLS["stereoset_uk"],
            "cluster": f"c{i}",
            "stratum": "target",
            "group": "a" if i < 3 else "b",
            "group_kind": "lexical_target",
            "condition": "intrasentence",
            "panel": str(i),
            "category": "gender",
            "eligible": True,
            "sentences": ["S", "C", "U"],
            "paired_eligible": False,
            "pairing_exclusion_reason": "UK-only fixture",
            "source": {"topic": "provisional topic", "final_text_severity": "high"},
        }
        for i in range(4)
    ]
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    specification = {
        "filename": path.name,
        "format": "adapted_jsonl_v1",
        "sha256": digest(path),
        "rows": len(rows),
        "language": "uk",
        "protocol": PROTOCOLS["stereoset_uk"],
        "counterpart": None,
        "pairing": {"mode": "unpaired", "reason": "UK-only fixture"},
    }
    return rows, specification, path


def save_annotations(tmp_path, specification, records):
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_sha256": specification["sha256"],
                "rows": records,
            }
        )
    )
    return path


def reviewed(identifier, **fields):
    return {"id": identifier, "reviewed": True, "reviewer": "Reviewer One", **fields}


def scored(rows, reverse=False):
    return score_all(
        rows,
        [
            {
                "id": row["id"],
                "scores": [2, 3, 1] if (row["group"] == "b") ^ reverse else [3, 2, 1],
            }
            for row in rows
        ],
    )


def point(report, metric, scope):
    return next(
        p
        for p in report["metric_points"]
        if p["metric"] == metric and p["scope"].get("scope") == scope
    )


def test_template_never_borrows_provisional_source_labels(dataset):
    rows, spec, _ = dataset
    template = inputs.annotation_template(spec, rows)
    assert template["dataset_sha256"] == spec["sha256"]
    assert [r["id"] for r in template["rows"]] == [r["id"] for r in rows]
    assert all(not r["reviewed"] and r["reviewer"] is None for r in template["rows"])
    assert all(
        r[field] is None for r in template["rows"] for field in inputs.SEMANTIC_FIELDS
    )


def test_reviewed_labels_are_dedicated_and_do_not_mutate_source(dataset, tmp_path):
    rows, spec, _ = dataset
    original = deepcopy(rows)
    path = save_annotations(
        tmp_path,
        spec,
        [
            reviewed(
                "0", topic="reviewed topic", topic_distance="near", severity="standard"
            )
        ],
    )
    mapping = inputs.load_annotations(path, spec, rows)
    enriched = inputs.enrich_rows(rows, mapping)
    assert rows == original
    assert enriched[0]["source"] == original[0]["source"]
    assert enriched[0]["analysis_topic"] == "reviewed topic"
    assert enriched[0]["analysis_severity"] == "standard"
    assert enriched[0]["analysis_annotations_reviewed"] is True
    assert enriched[0]["analysis_reviewer"] == "Reviewer One"
    assert enriched[1]["analysis_severity"] is None


def test_unreviewed_values_are_pending_and_never_used(dataset, tmp_path):
    rows, spec, _ = dataset
    path = save_annotations(
        tmp_path,
        spec,
        [
            {
                "id": "0",
                "severity": "high",
                "topic_distance": "target",
                "reviewed": False,
            }
        ],
    )
    mapping = inputs.load_annotations(path, spec, rows)
    assert mapping["0"]["severity"] is None
    assert mapping["0"]["supplied_annotations"]["severity"] == "high"
    assert mapping["0"]["status"] == "pending_review"
    enriched = inputs.enrich_rows(rows, mapping)
    report = inputs.severity_report(enriched, scored(rows))
    assert report["status"] == "pending_reviewed_labels"
    assert report["coverage"]["reviewed_severity_rows"] == 0
    assert len(report["coverage"]["missing_label_ids"]) == 4
    assert all(s["before"] is None for s in report["slices"])


@pytest.mark.parametrize(
    "record",
    [
        {"id": "0", "reviewed": True, "severity": "high"},
        {"id": "0", "severity": "high"},
        {"id": "0", "reviewed": 1, "reviewer": "R"},
        reviewed("0", severity="medium"),
        reviewed("0", topic_distance="automatic"),
        reviewed("0", evidence_assignment="aligned"),
        reviewed("0", topic=5),
        {**reviewed("0"), "reviewer": " "},
        {**reviewed("0"), "sevverity": "high"},
    ],
)
def test_annotation_schema_and_review_requirements(dataset, tmp_path, record):
    rows, spec, _ = dataset
    with pytest.raises(ValueError):
        inputs.load_annotations(save_annotations(tmp_path, spec, [record]), spec, rows)


@pytest.mark.parametrize(
    "change", ["hash", "extra_id", "duplicate_id", "partial_dataset"]
)
def test_annotations_cannot_bind_to_wrong_or_partial_data(dataset, tmp_path, change):
    rows, spec, _ = dataset
    records = [reviewed("0", severity="high")]
    if change == "extra_id":
        records.append(reviewed("other", severity="high"))
    if change == "duplicate_id":
        records.append(reviewed("0", severity="standard"))
    path = save_annotations(tmp_path, spec, records)
    if change == "hash":
        document = json.loads(path.read_text())
        document["dataset_sha256"] = "0" * 64
        path.write_text(json.dumps(document))
    if change == "partial_dataset":
        rows = rows[:1]
    with pytest.raises(ValueError):
        inputs.load_annotations(path, spec, rows)


def test_evidence_assignment_conflict_cannot_override_source(dataset, tmp_path):
    rows, spec, _ = dataset
    rows[0]["source"]["evidence_assignment"] = "target_adverse"
    path = save_annotations(
        tmp_path, spec, [reviewed("0", evidence_assignment="comparison_adverse")]
    )
    with pytest.raises(ValueError, match="conflicts"):
        inputs.load_annotations(path, spec, rows)
    path = save_annotations(
        tmp_path, spec, [reviewed("0", evidence_assignment="target_adverse")]
    )
    enriched = inputs.enrich_rows(rows, inputs.load_annotations(path, spec, rows))
    assert enriched[0]["source"]["evidence_assignment"] == "target_adverse"
    assert enriched[0]["evidence_assignment"] == "target_adverse"
    path = save_annotations(
        tmp_path,
        spec,
        [{"id": "0", "reviewed": False, "evidence_assignment": "comparison_adverse"}],
    )
    enriched = inputs.enrich_rows(rows, inputs.load_annotations(path, spec, rows))
    assert enriched[0]["source"]["evidence_assignment"] == "target_adverse"


@pytest.mark.parametrize("change", ["mapping_id", "reviewer"])
def test_direct_enrichment_cannot_bypass_review_or_mapping_identity(dataset, change):
    rows, _, _ = dataset
    record = reviewed("0", severity="high")
    if change == "mapping_id":
        record["id"] = "other"
    else:
        record.pop("reviewer")
    with pytest.raises(ValueError):
        inputs.enrich_rows(rows, {"0": record})


def test_template_export_reads_only_local_adapted_bundle(
    dataset, tmp_path, monkeypatch
):
    rows, spec, _ = dataset
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "datasets": {"stereoset_uk": spec}})
    )
    import huggingface_hub

    def forbidden_download(*args, **kwargs):
        raise AssertionError("Local export attempted a dataset download")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden_download)
    output = tmp_path / "template.json"
    document = inputs.export_annotation_template(manifest, "stereoset_uk", output)
    assert json.loads(output.read_text()) == document
    assert len(document["rows"]) == len(rows)
    with pytest.raises(ValueError, match="local bundle"):
        inputs.export_annotation_template(manifest, "warbias_en", output)


def test_severity_uses_registered_target_weights_and_fixed_paired_slices(
    dataset, tmp_path
):
    rows, spec, _ = dataset
    annotations = inputs.load_annotations(
        save_annotations(
            tmp_path, spec, [reviewed(r["id"], severity="high") for r in rows]
        ),
        spec,
        rows,
    )
    enriched = inputs.enrich_rows(rows, annotations)
    before = scored(rows)
    report = inputs.severity_report(enriched, before, scored(rows, reverse=True))
    assert report["status"] == "available"
    high = report["slices"][0]
    # Three stereotypical items from target a and one counter item from target b
    # still produce the native equal-target SS of 50 rather than item mean 75.
    assert point(high["before"], "ss", "target_macro")["value"] == 50
    assert point(high["after"], "ss", "target_macro")["value"] == 50
    delta = next(
        p
        for p in high["paired_metric_points"]
        if p["metric"] == "ss" and p["scope"].get("scope") == "target_macro"
    )
    assert delta["delta_after_minus_before"] == 0
    assert report["slices"][1]["before"] is None
    assert report["uncertainty"]["intervals"] is None


def test_partial_severity_coverage_remains_explicit_in_baseline_mode(dataset, tmp_path):
    rows, spec, _ = dataset
    annotations = inputs.load_annotations(
        save_annotations(tmp_path, spec, [reviewed("0", severity="standard")]),
        spec,
        rows,
    )
    report = inputs.severity_report(inputs.enrich_rows(rows, annotations), scored(rows))
    assert report["status"] == "incomplete_coverage"
    assert report["coverage"]["fraction"] == 0.25
    assert report["coverage"]["missing_label_ids"] == ["1", "2", "3"]
    assert report["mode"] == "baseline_only"
    assert all(
        s["after"] is None and s["paired_metric_points"] is None
        for s in report["slices"]
    )


def test_complete_severity_labels_do_not_claim_complete_native_pairing_panels():
    rows = []
    for role in ("occupation", "participant"):
        for first in ("occupation", "participant"):
            row_id = f"p1:{role}:double:male"
            source = {
                "prompt_id": f"{row_id}:{first}_first",
                "row_id": row_id,
                "pair_id": "p1",
                "context": "double",
                "pronoun_type": "nominative",
                "pronoun_set": "male",
                "option_order": f"{first}_first",
                "option_a_role": first,
                "option_b_role": "participant"
                if first == "occupation"
                else "occupation",
                "gold_role": role,
                "gold_option": "A" if role == first else "B",
                "prompt_uk": "A. технік\nB. клієнт\nВідповідь:",
            }
            rows.append(adapt("winopron_uk", source, language="uk"))
    enriched = inputs.enrich_rows(
        rows,
        {
            row["id"]: reviewed(row["id"], severity="high" if i == 0 else "standard")
            for i, row in enumerate(rows)
        },
    )
    outcomes = score_all(
        rows, [{"id": row["id"], "response": "AB"[row["gold"]]} for row in rows]
    )
    report = inputs.severity_report(enriched, outcomes, outcomes)
    assert report["coverage"]["fraction"] == 1
    assert report["status"] == "incomplete_native_panels"
    assert all(s["status"] == "incomplete_native_panel" for s in report["slices"])
    assert all(s["before"]["metric_points"] is None for s in report["slices"])
    assert all(s["paired_metric_points"] is None for s in report["slices"])


@pytest.mark.parametrize("change", ["missing", "duplicate", "source", "nonfinite"])
def test_severity_rejects_unmatched_or_invalid_saved_outcomes(dataset, change):
    rows, _, _ = dataset
    before = scored(rows)
    if change == "missing":
        before.pop()
    elif change == "duplicate":
        before.append(deepcopy(before[0]))
    elif change == "source":
        before[0]["group"] = "different"
    else:
        before[0]["lms"] = float("nan")
    with pytest.raises(ValueError):
        inputs.severity_report(rows, before)


def event(identifier, tokens=10, group="a", **extra):
    return {
        "event_id": identifier,
        "step": 0,
        "item_id": identifier,
        "stream": "forget",
        "language": "uk",
        "group": group,
        "tokens": tokens,
        **extra,
    }


def test_actual_exposure_counts_repetitions_tokens_and_observed_group_ratios():
    events = [
        event("e1", 10, item_id="same"),
        event("e2", 20, item_id="same", step=1),
        event("e3", 10, group="b", step=1),
        event("e4", 0, group="b", stream="general_preservation", language="en"),
    ]
    report = inputs.exposure_report(events)
    assert report["state"] == "actual_logged_exposure"
    assert report["totals"]["tokens"] == 40
    assert report["totals"]["items"] == 3
    assert report["totals"]["item_exposures"] == 4
    assert report["totals"]["steps"] == 2
    forget = next(c for c in report["by_stream"] if c["stream"] == "forget")
    assert forget["token_proportion"] == 1
    assert forget["item_exposure_proportion"] == 0.75
    ratio = next(c for c in report["group_token_imbalance"] if c["stream"] == "forget")
    assert ratio["max_min_token_ratio"] == 3
    zero = next(
        c
        for c in report["group_token_imbalance"]
        if c["stream"] == "general_preservation"
    )
    assert zero["max_min_token_ratio"] is None and zero["zero_denominator"]
    assert report == inputs.exposure_report(list(reversed(events)))


def test_no_logged_events_do_not_become_zero_measured_imbalance():
    report = inputs.exposure_report([])
    assert report["state"] == "pending_actual_training_events"
    assert report["totals"]["token_proportion"] is None
    assert report["group_token_imbalance"] == []


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "negative_step",
        "bool_step",
        "float_step",
        "nan_tokens",
        "negative_tokens",
        "stream",
        "language",
        "missing_group",
        "unknown_field",
    ],
)
def test_exposure_rejects_malformed_or_replayed_events(change):
    events = [event("e1")]
    edits = {
        "negative_step": ("step", -1),
        "bool_step": ("step", True),
        "float_step": ("step", 1.0),
        "nan_tokens": ("tokens", float("nan")),
        "negative_tokens": ("tokens", -1),
        "stream": ("stream", "evaluation"),
        "language": ("language", "fr"),
        "missing_group": ("group", " "),
        "unknown_field": ("evaluation_rows", 100),
    }
    if change == "duplicate":
        events.append(deepcopy(events[0]))
    else:
        key, value = edits[change]
        events[0][key] = value
    with pytest.raises(ValueError):
        inputs.exposure_report(events)


@pytest.fixture
def training_inventory(dataset, tmp_path):
    rows, specification, data_path = dataset
    for row in rows:
        row["split"] = "train"
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    specification.update(
        sha256=digest(data_path),
        source_role="training_signal",
        measurement_purpose="pre_intervention_training_source_signal",
        measurement_eligibility="active",
        source_provenance=[
            {
                "repo_id": "FairForget/StereoSet-UK-Unlearning",
                "revision": "a" * 40,
                "path": "data/train.csv",
                "sha256": "b" * 64,
                "published": True,
            }
        ],
    )
    bundle = tmp_path / "manifest.json"
    bundle.write_text(
        json.dumps({"schema_version": 1, "datasets": {"stereoset_uk": specification}})
    )
    config = {
        "approved_dataset_scope": {
            "namespace": "FairForget",
            "published_only": True,
            "training_repositories": ["FairForget/StereoSet-UK-Unlearning"],
            "evaluation_only_repositories": ["FairForget/BBQ-UK"],
            "preparation_only_repositories": ["FairForget/StereoSet-EN-Unlearning"],
            "source_revisions": {"FairForget/StereoSet-UK-Unlearning": "a" * 40},
        },
        "phases": {
            "train_signal": {
                "split": "train",
                "dataset_bundle": bundle.name,
                "tasks": ["stereoset_uk"],
            }
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    source_manifest = {
        "training_sources": deepcopy(specification["source_provenance"]),
        "samples": [
            {
                "task": "stereoset_uk",
                "dataset_sha256": specification["sha256"],
                "item_id": rows[0]["id"],
                "stream": "forget",
                "language": "uk",
                "group": "a",
                "selection_reviewed": True,
                "reviewer": "Reviewer One",
                "loss_role": "stereotype",
            }
        ],
    }
    return source_manifest, config_path, config, bundle, specification, data_path


def test_exposure_selection_uses_local_frozen_reviewed_training_rows(
    training_inventory, monkeypatch
):
    manifest, config_path, config, _, _, _ = training_inventory
    import huggingface_hub

    def forbidden_download(*args, **kwargs):
        raise AssertionError("Exposure verification attempted a dataset download")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden_download)
    manifest["samples"].append(
        {
            **manifest["samples"][0],
            "stream": "anti_stereotype",
            "loss_role": "antistereotype",
        }
    )
    selected = inputs.validate_exposure_samples(manifest, config_path, config)
    assert selected == {
        ("0", "forget", "uk"): "a",
        ("0", "anti_stereotype", "uk"): "a",
    }


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "duplicate",
        "unselected_task",
        "dataset_sha",
        "invented_id",
        "language",
        "group",
        "unreviewed",
        "missing_reviewer",
        "loss_role",
        "source_sha",
        "source_pin",
        "unpublished_source",
        "no_scope",
    ],
)
def test_exposure_selection_rejects_unbound_or_unreviewed_samples(
    training_inventory, change
):
    manifest, config_path, config, _, _, _ = training_inventory
    sample = manifest["samples"][0]
    edits = {
        "unselected_task": ("task", "bbq_uk"),
        "dataset_sha": ("dataset_sha256", "0" * 64),
        "invented_id": ("item_id", "evaluation-only-item"),
        "language": ("language", "en"),
        "group": ("group", "other"),
        "unreviewed": ("selection_reviewed", False),
        "missing_reviewer": ("reviewer", None),
        "loss_role": ("loss_role", "general"),
    }
    if change in edits:
        key, value = edits[change]
        sample[key] = value
    elif change == "empty":
        manifest["samples"] = []
    elif change == "duplicate":
        manifest["samples"].append(deepcopy(sample))
    elif change == "source_sha":
        manifest["training_sources"][0]["sha256"] = "0" * 64
    elif change == "source_pin":
        manifest["training_sources"][0]["revision"] = "c" * 40
    elif change == "unpublished_source":
        manifest["training_sources"][0]["published"] = False
    else:
        config.pop("approved_dataset_scope")
    with pytest.raises(ValueError):
        inputs.validate_exposure_samples(manifest, config_path, config)


@pytest.mark.parametrize("change", ["split", "source_role", "blocked", "edited_data"])
def test_exposure_selection_rejects_nontraining_blocked_or_edited_inventory(
    training_inventory, change
):
    manifest, config_path, config, bundle, specification, data_path = training_inventory
    if change == "split":
        rows = [json.loads(line) for line in data_path.read_text().splitlines()]
        rows[0]["split"] = "evaluation"
        data_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        specification["sha256"] = digest(data_path)
        manifest["samples"][0]["dataset_sha256"] = specification["sha256"]
    elif change == "source_role":
        specification["source_role"] = "evaluation"
    elif change == "blocked":
        specification["measurement_eligibility"] = "preparation_only"
        specification["measurement_eligibility_reason"] = "Unpublished staged source"
    else:
        data_path.write_text(data_path.read_text() + "\n")
    bundle.write_text(
        json.dumps({"schema_version": 1, "datasets": {"stereoset_uk": specification}})
    )
    with pytest.raises(ValueError):
        inputs.validate_exposure_samples(manifest, config_path, config)
