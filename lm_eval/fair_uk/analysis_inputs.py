"""Offline reviewed annotations, severity slices, and actual exposure summaries.

The caller binds saved model runs and actual training-log headers to immutable
sources. This module never infers semantic labels or treats evaluation rows as
training exposure. Native metrics keep their registered benchmark denominators.
"""

import json
import math
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from lm_eval.fair_uk.data import bundle_registry, load
from lm_eval.fair_uk.metrics import native_summaries
from lm_eval.fair_uk.reporting import native_entries


SEMANTIC_FIELDS = ("topic", "topic_distance", "severity", "evidence_assignment")
ENUMS = {
    "topic_distance": {"target", "near", "far"},
    "severity": {"high", "standard"},
    "evidence_assignment": {"target_adverse", "comparison_adverse"},
}
STREAMS = ("forget", "anti_stereotype", "general_preservation")
LOSS_ROLES = {
    "forget": "stereotype",
    "anti_stereotype": "antistereotype",
    "general_preservation": "general",
}
EVENT_FIELDS = {"event_id", "step", "item_id", "stream", "language", "group", "tokens"}
SOURCE_FIELDS = (
    "task",
    "language",
    "protocol",
    "cluster",
    "stratum",
    "group",
    "group_kind",
    "condition",
    "panel",
    "category",
    "score_group",
    "eligible",
)
SCOPE_FIELDS = (
    "scope",
    "stratum",
    "condition",
    "score_group",
    "category",
    "group",
    "pronoun_type",
    "pronoun_set",
    "context",
    "actor",
    "gender",
    "age",
)


def _index_rows(rows):
    indexed = {}
    for row in rows:
        identifier = row.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in indexed
        ):
            raise ValueError("Dataset rows require unique nonempty IDs")
        indexed[identifier] = row
    if not indexed:
        raise ValueError("Provide the nonempty full dataset")
    return indexed


def _dataset_binding(specification, rows):
    if specification.get("format") != "adapted_jsonl_v1" or not re.fullmatch(
        r"[0-9a-f]{64}", specification.get("sha256", "")
    ):
        raise ValueError("Annotations require a hash-bound adapted local dataset")
    indexed = _index_rows(rows)
    if specification.get("rows") != len(indexed):
        raise ValueError("Annotation binding requires the full dataset row inventory")
    return indexed


def _annotation(record, source_row):
    if type(record.get("reviewed")) is not bool:
        raise ValueError("Every annotation requires an explicit reviewed boolean")
    reviewer = record.get("reviewer")
    if reviewer is not None and (not isinstance(reviewer, str) or not reviewer.strip()):
        raise ValueError("Annotation reviewer must be a nonempty string or null")
    if record["reviewed"] and reviewer is None:
        raise ValueError("Reviewed annotations require a recorded reviewer")
    supplied = {field: record.get(field) for field in SEMANTIC_FIELDS}
    for field, value in supplied.items():
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid annotation field: {field}")
        if field in ENUMS and value not in ENUMS[field]:
            raise ValueError(f"Unknown annotation value: {field}")
    effective = supplied if record["reviewed"] else dict.fromkeys(SEMANTIC_FIELDS)
    assignment = effective["evidence_assignment"]
    source = source_row.get("source", {})
    if not isinstance(source, dict):
        raise TypeError("Raw source metadata must be an object")
    existing = {
        value
        for value in (
            source_row.get("evidence_assignment"),
            source.get("evidence_assignment"),
        )
        if value is not None
    }
    if len(existing) > 1 or (
        assignment is not None and existing and assignment not in existing
    ):
        raise ValueError(
            "Reviewed evidence assignment conflicts with existing source metadata"
        )
    return {
        "id": record["id"],
        "reviewed": record["reviewed"],
        "reviewer": reviewer,
        **effective,
        "supplied_annotations": supplied,
        "pending_fields": [
            field
            for field, value in supplied.items()
            if value is not None and not record["reviewed"]
        ],
        "status": "reviewed" if record["reviewed"] else "pending_review",
    }


def load_annotations(path, dataset_spec, rows):
    """Return ID-keyed reviewed values, nulling all unreviewed semantic fields.

    The bound SHA is the adapted data-file SHA, not a manifest or original CSV SHA.
    Missing records are permitted and remain explicitly unlabeled on enrichment.
    """
    indexed = _dataset_binding(dataset_spec, rows)
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "dataset_sha256", "rows"}
        or document.get("schema_version") != 1
        or document.get("dataset_sha256") != dataset_spec["sha256"]
        or not isinstance(document.get("rows"), list)
    ):
        raise ValueError("Annotation schema or adapted dataset SHA differs")
    result = {}
    allowed = {"id", "reviewed", "reviewer", *SEMANTIC_FIELDS}
    for record in document["rows"]:
        if not isinstance(record, dict) or set(record) - allowed:
            raise ValueError("Unknown or malformed annotation fields")
        identifier = record.get("id")
        if (
            not isinstance(identifier, str)
            or identifier not in indexed
            or identifier in result
        ):
            raise ValueError(
                "Annotation IDs must be unique and belong to the full dataset"
            )
        result[identifier] = _annotation(record, indexed[identifier])
    return result


def enrich_rows(rows, annotations):
    """Copy rows with dedicated effective labels; preserve provisional source labels."""
    indexed = _index_rows(rows)
    if set(annotations) - set(indexed):
        raise ValueError("Annotation IDs differ from the dataset")
    result = []
    for row in rows:
        original = annotations.get(row["id"])
        record = (
            _annotation(
                {
                    key: original.get(key)
                    for key in ("id", "reviewed", "reviewer", *SEMANTIC_FIELDS)
                },
                row,
            )
            if original is not None
            else None
        )
        if record is not None and record["id"] != row["id"]:
            raise ValueError("Annotation mapping key differs from its ID")
        enriched = deepcopy(row)
        for field in ("topic", "topic_distance", "severity"):
            enriched["analysis_" + field] = record[field] if record else None
        enriched["analysis_annotations_reviewed"] = (
            record["reviewed"] if record else False
        )
        enriched["analysis_reviewer"] = (
            record["reviewer"] if record and record["reviewed"] else None
        )
        enriched["analysis_pending_annotation_fields"] = (
            original.get("pending_fields", record["pending_fields"])
            if record
            else list(SEMANTIC_FIELDS)
        )
        if record and record["evidence_assignment"] is not None:
            enriched["evidence_assignment"] = record["evidence_assignment"]
            enriched.setdefault("source", {})["evidence_assignment"] = record[
                "evidence_assignment"
            ]
        result.append(enriched)
    return result


def annotation_template(dataset_spec, rows):
    """Return a SHA-bound blank review template without borrowing source labels."""
    _dataset_binding(dataset_spec, rows)
    return {
        "schema_version": 1,
        "dataset_sha256": dataset_spec["sha256"],
        "rows": [
            {
                "id": row["id"],
                **dict.fromkeys(SEMANTIC_FIELDS),
                "reviewed": False,
                "reviewer": None,
            }
            for row in rows
        ],
    }


def export_annotation_template(bundle_path, task, output_path):
    """Export only an explicitly present adapted local bundle task; never download."""
    specification = bundle_registry(bundle_path).get(task)
    if not specification or specification.get("format") != "adapted_jsonl_v1":
        raise ValueError(
            "Annotation export requires an adapted task in the local bundle"
        )
    rows = load(task, dataset_bundle=bundle_path)
    document = annotation_template(specification, rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return document


def _exposure_counts(events, total_tokens, total_events):
    tokens = sum(event["tokens"] for event in events)
    steps = {event["step"] for event in events}
    return {
        "events": len(events),
        "item_exposures": len(events),
        "items": len({event["item_id"] for event in events}),
        "tokens": tokens,
        "steps": len(steps),
        "first_step": min(steps) if steps else None,
        "last_step": max(steps) if steps else None,
        "token_proportion": tokens / total_tokens if total_tokens else None,
        "item_exposure_proportion": len(events) / total_events
        if total_events
        else None,
    }


def exposure_report(events):
    """Summarize validated actual events; callers bind run/source/log provenance."""
    events = list(events)
    identifiers = set()
    for event in events:
        if not isinstance(event, dict) or set(event) != EVENT_FIELDS:
            raise ValueError("Training exposure events must match the event schema")
        if any(
            not isinstance(event[key], str) or not event[key].strip()
            for key in ("event_id", "item_id", "group")
        ):
            raise ValueError(
                "Training exposure event IDs, item IDs and groups must be nonempty strings"
            )
        if event["event_id"] in identifiers:
            raise ValueError("Duplicate training exposure event ID")
        identifiers.add(event["event_id"])
        if event["stream"] not in STREAMS or event["language"] not in ("uk", "en"):
            raise ValueError("Unknown training exposure stream or language")
        if any(
            type(event[key]) is not int or event[key] < 0 for key in ("step", "tokens")
        ):
            raise ValueError(
                "Exposure steps and token counts require nonnegative finite integers"
            )
    total_tokens = sum(event["tokens"] for event in events)
    total_events = len(events)
    aggregations = {}
    for name, fields in {
        "by_stream": ("stream",),
        "by_language": ("language",),
        "by_group": ("group",),
        "by_stream_language_group": ("stream", "language", "group"),
    }.items():
        blocks = defaultdict(list)
        for event in events:
            blocks[tuple(event[field] for field in fields)].append(event)
        aggregations[name] = [
            {
                **dict(zip(fields, key, strict=True)),
                **_exposure_counts(blocks[key], total_tokens, total_events),
            }
            for key in sorted(blocks)
        ]
    panels = defaultdict(list)
    for cell in aggregations["by_stream_language_group"]:
        panels[cell["stream"], cell["language"]].append(cell)
    ratios = []
    for (stream, language), cells in sorted(panels.items()):
        tokens = [cell["tokens"] for cell in cells]
        ratios.append(
            {
                "stream": stream,
                "language": language,
                "observed_groups": len(cells),
                "maximum_tokens": max(tokens),
                "minimum_tokens": min(tokens),
                "max_min_token_ratio": max(tokens) / min(tokens)
                if min(tokens)
                else None,
                "zero_denominator": min(tokens) == 0,
            }
        )
    return {
        "schema_version": 1,
        "kind": "training_exposure_summary",
        "state": "actual_logged_exposure"
        if events
        else "pending_actual_training_events",
        "totals": _exposure_counts(events, total_tokens, total_events),
        **aggregations,
        "group_token_imbalance": ratios,
        "definitions": {
            "items": "Unique logged item IDs; counts can overlap across languages and streams.",
            "item_exposures": "Logged item events, including repeated presentations.",
            "steps": "Distinct logged optimizer-step identifiers; not a claimed number of updates.",
            "proportions": "Shares of total logged tokens or item events; zero denominators are null.",
            "ratio": "Maximum/minimum group token counts among observed groups in each stream/language; absent groups are not inferred.",
        },
        "qualification": "Validated event structure alone does not authenticate actual execution, source-role permissions or applied gradients. Caller must verify the actual-run/source-bound log header. No evaluation counts are used.",
    }


def validate_exposure_samples(source_manifest, config_path, config):
    """Bind reviewed loss-role selections to approved local training inventories.

    The caller verifies the source-manifest/config envelope hashes and intervention
    identity. This helper verifies membership and declared reviewed selection roles,
    without inferring either semantic suitability or whether gradients were applied.
    """
    from lm_eval.fair_uk.baseline import measurement_policy, resolve

    scope = config.get("approved_dataset_scope")
    phase = config.get("phases", {}).get("train_signal")
    if (
        not isinstance(scope, dict)
        or scope.get("namespace") != "FairForget"
        or scope.get("published_only", True) is not True
        or not scope.get("training_repositories")
        or not isinstance(phase, dict)
        or phase.get("split") != "train"
    ):
        raise ValueError("Exposure selections require the approved local train phase")
    samples = source_manifest.get("samples")
    sources = source_manifest.get("training_sources")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Exposure requires nonempty reviewed training selections")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Exposure selections require training-source provenance")
    source_keys = set()
    for source in sources:
        if not isinstance(source, dict):
            raise TypeError("Exposure training-source provenance must be an object")
        repository = source.get("repo_id")
        revision = source.get("revision")
        checksum = source.get("sha256")
        if (
            repository not in scope["training_repositories"]
            or repository in scope.get("preparation_only_repositories", [])
            or source.get("published") is not True
            or not re.fullmatch(r"[0-9a-f]{40}", str(revision))
            or scope.get("source_revisions", {}).get(repository) != revision
            or not re.fullmatch(r"[0-9a-f]{64}", str(checksum))
        ):
            raise ValueError(
                "Exposure provenance is not an approved published training source"
            )
        source_keys.add((repository, revision, checksum))
    bundle = resolve(config_path, phase["dataset_bundle"])
    registry = bundle_registry(bundle)
    selected_tasks = phase.get("tasks", [])
    inventories = {}
    selected = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise TypeError("Exposure training selections must be objects")
        task = sample.get("task")
        stream = sample.get("stream")
        if (
            not isinstance(task, str)
            or task not in selected_tasks
            or task not in registry
            or stream not in LOSS_ROLES
            or sample.get("selection_reviewed") is not True
            or not isinstance(sample.get("reviewer"), str)
            or not sample["reviewer"].strip()
            or sample.get("loss_role") != LOSS_ROLES[stream]
        ):
            raise ValueError(
                "Exposure selections require a configured task and reviewed matching loss role"
            )
        specification = registry[task]
        eligibility, _ = measurement_policy(config, specification, phase["split"])
        if (
            eligibility != "active"
            or specification.get("format") != "adapted_jsonl_v1"
            or sample.get("dataset_sha256") != specification["sha256"]
        ):
            raise ValueError(
                "Exposure selections require an active hash-bound local training dataset"
            )
        provenance = specification["source_provenance"]
        if any(
            (source["repo_id"], source["revision"], source["sha256"]) not in source_keys
            for source in provenance
        ):
            raise ValueError(
                "Exposure header omits the selected dataset's frozen training sources"
            )
        if task not in inventories:
            rows = load(task, dataset_bundle=bundle)
            if {row.get("split") for row in rows} != {"train"}:
                raise ValueError(
                    "Exposure selections contain development or evaluation rows"
                )
            inventories[task] = _dataset_binding(specification, rows)
        identifier = sample.get("item_id")
        if not isinstance(identifier, str) or identifier not in inventories[task]:
            raise ValueError(
                "Exposure item is outside the full frozen training inventory"
            )
        row = inventories[task][identifier]
        if (
            sample.get("language") not in ("uk", "en")
            or sample["language"] != row["language"]
            or not isinstance(sample.get("group"), str)
            or not sample["group"].strip()
            or sample["group"] != row["group"]
        ):
            raise ValueError(
                "Exposure selection language or group differs from its source row"
            )
        key = (identifier, stream, sample["language"])
        if key in selected:
            raise ValueError("Duplicate exposure item/stream/language selection")
        selected[key] = row["group"]
    return selected


def _scored_index(rows, scored):
    original = _index_rows(rows)
    indexed = _index_rows(scored)
    if original.keys() != indexed.keys():
        raise ValueError("Severity scoring requires exactly the full dataset IDs")
    for identifier, row in original.items():
        item = indexed[identifier]
        if any(
            field in row and row[field] != item.get(field) for field in SOURCE_FIELDS
        ):
            raise ValueError("Severity scored source metadata differs")
        for key in (
            "accuracy",
            "lms",
            "stereotype_preference",
            "stereotype_tie",
            "unknown_rate",
            "invalid",
            "task_success",
            "full_refusal_rate",
            "any_refusal_rate",
            "judgment_coverage",
        ):
            value = item.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("Severity outcomes must be finite rates")
        for value in item.get("candidate_means", []):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Severity candidate means must be finite")
    return indexed


def metric_points(scored, native=None):
    """Flatten native scores with their registered units and explicit scope labels."""
    native = native_summaries(scored) if native is None else native
    result = []
    for record in native:
        labels = {key: record[key] for key in SCOPE_FIELDS if key in record}
        for _, metric, value, unit in native_entries(
            {"task": scored[0]["task"], "native": [record]}
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError("Native severity metric is not finite")
            result.append(
                {"scope": labels, "metric": metric, "value": value, "unit": unit}
            )
    return result


def _slice_report(scored):
    if not scored:
        return None
    try:
        native = native_summaries(scored)
    except ValueError as error:
        return {
            "status": "incomplete_native_panel",
            "reason": str(error),
            "native_summaries": None,
            "metric_points": None,
        }
    return {
        "status": "available",
        "native_summaries": native,
        "metric_points": metric_points(scored, native),
    }


def _deltas(before, after):
    if (
        before is None
        or after is None
        or before.get("metric_points") is None
        or after.get("metric_points") is None
    ):
        return None
    indexed = []
    for report in (before, after):
        points = {}
        for point in report["metric_points"]:
            key = (
                json.dumps(point["scope"], sort_keys=True),
                point["metric"],
                point["unit"],
            )
            if key in points:
                raise ValueError("Ambiguous native severity metric scope")
            points[key] = point
        indexed.append(points)
    output = []
    for key in sorted(indexed[0].keys() | indexed[1].keys()):
        point = indexed[0].get(key, indexed[1].get(key))
        before_value = indexed[0].get(key, {}).get("value")
        after_value = indexed[1].get(key, {}).get("value")
        difference = (
            after_value - before_value
            if before_value is not None and after_value is not None
            else None
        )
        output.append(
            {
                **point,
                "before": before_value,
                "after": after_value,
                "delta_after_minus_before": difference,
            }
        )
    return output


def severity_report(rows, before_scored, after_scored=None):
    """Describe fixed reviewed severity slices, preserving native metric weights.

    Labels come only from enrich_rows' dedicated reviewed fields. Point differences
    require both saved outcomes on exactly the same dataset; this helper supplies
    no confidence intervals or causal claims. Incomplete native panels stay pending.
    """
    before = _scored_index(rows, before_scored)
    after = _scored_index(rows, after_scored) if after_scored is not None else None
    if len({row["task"] for row in rows}) != 1:
        raise ValueError("Severity reporting requires exactly one task")
    labels = {}
    missing, unreviewed = [], []
    for row in rows:
        reviewer = row.get("analysis_reviewer")
        reviewed = (
            row.get("analysis_annotations_reviewed") is True
            and isinstance(reviewer, str)
            and bool(reviewer.strip())
        )
        severity = row.get("analysis_severity") if reviewed else None
        if severity is not None and severity not in ENUMS["severity"]:
            raise ValueError("Unknown reviewed severity label")
        if severity is None:
            missing.append(row["id"])
            if not reviewed:
                unreviewed.append(row["id"])
        else:
            labels[row["id"]] = severity
    slices = []
    for severity in ("high", "standard"):
        identifiers = [row["id"] for row in rows if labels.get(row["id"]) == severity]
        before_report = _slice_report(
            [before[identifier] for identifier in identifiers]
        )
        after_report = (
            _slice_report([after[identifier] for identifier in identifiers])
            if after is not None
            else None
        )
        panel_incomplete = any(
            report is not None and report["status"] != "available"
            for report in (before_report, after_report)
        )
        slices.append(
            {
                "severity": severity,
                "rows": len(identifiers),
                "independent_components": len(
                    {before[identifier]["cluster"] for identifier in identifiers}
                ),
                "item_ids": identifiers,
                "status": "no_reviewed_rows"
                if not identifiers
                else "incomplete_native_panel"
                if panel_incomplete
                else "available",
                "before": before_report,
                "after": after_report,
                "paired_metric_points": _deltas(before_report, after_report)
                if after is not None
                else None,
            }
        )
    return {
        "schema_version": 1,
        "kind": "reviewed_severity_analysis",
        "mode": "paired_before_after" if after is not None else "baseline_only",
        "status": "pending_reviewed_labels"
        if not labels
        else "incomplete_coverage"
        if missing
        else "incomplete_native_panels"
        if any(s["status"] == "incomplete_native_panel" for s in slices)
        else "available",
        "coverage": {
            "dataset_rows": len(rows),
            "reviewed_severity_rows": len(labels),
            "fraction": len(labels) / len(rows),
            "missing_label_ids": missing,
            "unreviewed_ids": unreviewed,
        },
        "slices": slices,
        "weighting": "Each slice independently uses native_summaries' registered task-specific denominators and units; identical slice IDs and source metadata are shared before/after. No custom cross-slice pooled headline score is inferred.",
        "uncertainty": {"status": "point_estimates_only", "intervals": None},
        "qualification": "Partial annotation coverage is descriptive. Complete labels do not establish content validity, precision or complete native pairing panels. Source-provisional severity is never used automatically.",
    }
