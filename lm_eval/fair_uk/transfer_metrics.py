"""Offline paired intervention effects and redistribution on explicit metadata.

The caller supplies already scored, identical pre/post item inventories. This
module never loads a task, dataset, model, or annotation. Ratios are descriptive
and remain unavailable without an explicit practical threshold and a positive
paired target-improvement interval.
"""

import json
import math
from collections import Counter, defaultdict
from statistics import fmean

import numpy as np

from lm_eval.fair_uk.capabilities import benchmark
from lm_eval.fair_uk.data import family


_VARIANT_FIELDS = (
    "condition",
    "polarity",
    "target_mention_position",
    "option_order",
    "context",
    "variant_signature",
    "actor",
    "gender",
    "age",
    "language",
    "analysis_topic",
    "analysis_topic_distance",
)
_SUFFICIENT_EVIDENCE = {
    "stereotype_aligned",
    "stereotype_conflicting",
    "disambiguated",
    "sufficient_evidence",
}
_PRECISION = (
    "Intervals resample declared source components within each stratum jointly "
    "across profiles and languages. Few components give imprecise intervals; "
    "these descriptive effects do not establish causal demographic effects."
)


def _indexed(records, label):
    indexed = {}
    for row in records:
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{label}: every item needs a nonempty string id")
        if item_id in indexed:
            raise ValueError(f"{label}: duplicate id {item_id}")
        indexed[item_id] = row
    return indexed


def _prepare(pairs):
    grouped = defaultdict(list)
    tasks = set()
    for pair in pairs:
        task = pair["task"]
        if task in tasks:
            raise ValueError(f"Duplicate task pair: {task}")
        tasks.add(task)
        name = benchmark(task)
        rows = _indexed(pair["rows"], f"{task} rows")
        before = _indexed(pair["before_scored"], f"{task} before")
        after = _indexed(pair["after_scored"], f"{task} after")
        if rows.keys() != before.keys() or rows.keys() != after.keys():
            raise ValueError(f"{task}: row, before, and after id inventories differ")
        for item_id in sorted(rows):
            row = rows[item_id]
            for source in (row, before[item_id], after[item_id]):
                if source.get("task", task) != task:
                    raise ValueError(f"{task}/{item_id}: task identity mismatch")
            for scored in (before[item_id], after[item_id]):
                for key in (
                    "protocol",
                    "language",
                    "condition",
                    "cluster",
                    "stratum",
                    "comparison_unit",
                    "actor",
                    "gender",
                    "age",
                ):
                    if key in scored and scored[key] != row.get(key):
                        raise ValueError(
                            f"{task}/{item_id}: scored {key} identity mismatch"
                        )
            record = {
                **row,
                "task": task,
                "_before": before[item_id],
                "_after": after[item_id],
            }
            grouped[name, row.get("protocol")].append(record)
    return grouped


def _targets(intervention, axis):
    value = intervention.get("target_values", {}).get(axis)
    if axis == "language":
        value = intervention.get("training_language")
        declared = intervention.get("target_values", {}).get("language")
        if declared is not None and declared != value:
            raise ValueError("language target must equal training_language")
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{axis}: target values must be nonempty strings")
    return list(dict.fromkeys(values))


def _metric(rows, intervention):
    explicit = intervention.get("harm_metric")
    if explicit is not None:
        metric = explicit.get("id")
        condition = explicit.get("condition")
        unit = explicit.get("unit", "rate")
        if not isinstance(metric, str) or not metric or not condition:
            return None, "QA harm requires an explicit metric id and condition"
        if unit != "rate":
            return None, "bounded QA harm must declare rate units in [0, 1]"
        if any(family(r["task"]) not in ("warbias", "bbq_uk") for r in rows):
            return None, "explicit QA harm cannot be mixed with another outcome family"
        if metric == "unknown_rate" and condition not in _SUFFICIENT_EVIDENCE:
            return None, "unknown_rate is harm only on explicitly sufficient evidence"
        return {
            "id": metric,
            "score_field": metric,
            "unit": unit,
            "scale": 1,
            "harm": "explicit_bounded_QA_rate",
            "condition": condition,
        }, None
    if all(family(r["task"]) in ("stereoset_uk", "warbias_triplets") for r in rows):
        return {
            "id": "ss",
            "score_field": "stereotype_preference",
            "unit": "percentage_points",
            "scale": 100,
            "harm": "abs(native_ss - 50)",
            "condition": None,
        }, None
    return None, "no association outcome or explicitly named bounded QA harm"


def _value(row, axis, targets):
    if axis != "topic":
        return row.get(axis)
    if row.get("analysis_annotations_reviewed") is not True:
        return None
    topic = row.get("analysis_topic")
    if not isinstance(topic, str) or not topic:
        return None
    if topic in targets:
        return "target"
    distance = row.get("analysis_topic_distance")
    return distance if distance in ("near", "far") else None


def _unit(row):
    return row["stratum"], row["cluster"]


def _signature(row, axis):
    return json.dumps(
        {
            key: row.get(key)
            for key in _VARIANT_FIELDS
            if key != axis
            and (
                not key.startswith("analysis_")
                or row.get("analysis_annotations_reviewed") is True
            )
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _matched(rows, axis, values):
    """Require all values on the same content and complete variant multiset."""
    blocks = defaultdict(lambda: defaultdict(list))
    unmatched = []
    for row in rows:
        if not row.get("comparison_unit"):
            unmatched.append(
                {
                    "task": row["task"],
                    "id": row["id"],
                    "reason": "missing comparison_unit",
                }
            )
        else:
            blocks[str(row["comparison_unit"])][row[axis]].append(row)
    matched = []
    for unit, cells in sorted(blocks.items()):
        reason = None
        if set(cells) != set(values):
            reason = "incomplete axis-value coverage"
        else:
            signatures = [
                Counter((_signature(r, axis), _unit(r)) for r in cells[v])
                for v in values
            ]
            if any(signature != signatures[0] for signature in signatures[1:]):
                reason = "nonmatching variant conditions or remaining attributes"
            sampling = [{_unit(r) for r in cells[v]} for v in values]
            if any(units != sampling[0] for units in sampling[1:]):
                reason = "nonmatching stratum/cluster source components"
        block_rows = [r for records in cells.values() for r in records]
        if reason:
            unmatched.append(
                {
                    "comparison_unit": unit,
                    "reason": reason,
                    "present_values": sorted(cells),
                    "rows": [{"task": r["task"], "id": r["id"]} for r in block_rows],
                }
            )
        else:
            matched.extend(block_rows)
    return matched, unmatched


def _weighting(rows):
    if family(rows[0]["task"]) == "stereoset_uk":
        return "target_macro"
    if family(rows[0]["task"]) == "warbias_triplets" or rows[0].get(
        "expansion_version"
    ):
        return "component_macro"
    return "row_pooled"


def _point(rows, time, metric, weighting, weights=None):
    field = metric["score_field"]
    if weighting == "target_macro":
        groups = defaultdict(list)
        for row in rows:
            weight = weights.get(_unit(row), 0) if weights is not None else 1
            groups[row["group"]].append((row[time][field], weight))
        estimates = []
        for values in groups.values():
            denominator = sum(w for _, w in values)
            if not denominator:
                return None
            estimates.append(sum(v * w for v, w in values) / denominator)
        estimate = fmean(estimates)
    elif weighting == "component_macro":
        units = defaultdict(list)
        for row in rows:
            units[_unit(row)].append(row[time][field])
        denominator = sum(
            weights.get(k, 0) if weights is not None else 1 for k in units
        )
        if not denominator:
            return None
        estimate = (
            sum(
                fmean(v) * (weights.get(k, 0) if weights is not None else 1)
                for k, v in units.items()
            )
            / denominator
        )
    else:
        denominator = sum(
            weights.get(_unit(r), 0) if weights is not None else 1 for r in rows
        )
        if not denominator:
            return None
        estimate = (
            sum(
                r[time][field]
                * (weights.get(_unit(r), 0) if weights is not None else 1)
                for r in rows
            )
            / denominator
        )
    return metric["scale"] * estimate


def _harm(point, metric):
    return abs(point - 50) if metric["harm"] == "abs(native_ss - 50)" else point


def _ci(values):
    return np.quantile(values, [0.025, 0.975]).tolist() if values else None


def _draws(rows, cells, bootstrap, rng):
    reasons = []
    for value, selected in cells.items():
        strata = defaultdict(set)
        for row in selected:
            strata[row["stratum"]].add(row["cluster"])
        if any(len(units) < 2 for units in strata.values()):
            reasons.append(f"cell {value} has fewer than two components in a stratum")
    if not bootstrap:
        reasons.append("bootstrap was not requested")
    if reasons:
        return [], reasons
    strata = defaultdict(set)
    for row in rows:
        strata[row["stratum"]].add(row["cluster"])
    draws = [{} for _ in range(bootstrap)]
    for stratum, clusters in sorted(strata.items()):
        units = sorted(clusters)
        sampled = rng.integers(len(units), size=(bootstrap, len(units)))
        for weights, sample in zip(draws, sampled, strict=True):
            weights.update(
                {(stratum, units[i]): int(n) for i, n in Counter(sample).items()}
            )
    return draws, []


def _unavailable(base, reason, **details):
    return {
        **base,
        "status": "unavailable",
        "unavailable_reasons": [reason],
        "cells": {},
        "redistribution": {
            "numerator": None,
            "denominator": None,
            "ratio": None,
            "ratio_unavailable_reasons": [reason],
        },
        **details,
    }


def _axis_report(rows, axis, targets, metric, bootstrap, rng, threshold, base):
    if not targets:
        return _unavailable(base, f"no declared target for {axis}")
    missing = [
        r
        for r in rows
        if not isinstance(_value(r, axis, targets), str) or not _value(r, axis, targets)
    ]
    if missing:
        return _unavailable(
            base,
            f"missing explicit {'reviewed ' if axis == 'topic' else ''}{axis} metadata",
            missing_metadata_rows=[{"task": r["task"], "id": r["id"]} for r in missing],
        )
    original = rows
    effective_targets = ["target"] if axis == "topic" else targets
    values = sorted({_value(r, axis, targets) for r in rows})
    if not set(effective_targets) <= set(values):
        return _unavailable(base, "declared target values have no complete support")
    unmatched = []
    if axis != "topic":
        rows, unmatched = _matched(rows, axis, values)
    if not rows:
        return _unavailable(
            base, "no complete matched-content blocks", unmatched=unmatched
        )
    cells = {
        value: [r for r in rows if _value(r, axis, targets) == value]
        for value in values
    }
    field = metric["score_field"]
    missing_scores = []
    for row in rows:
        for time in ("_before", "_after"):
            value = row[time].get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                missing_scores.append(
                    {"task": row["task"], "id": row["id"], "time": time[1:]}
                )
    if missing_scores:
        return _unavailable(
            base,
            "bounded outcome lacks complete valid pre/post scoring",
            unscored_rows=missing_scores,
            unmatched=unmatched,
        )
    weighting = _weighting(rows)
    if weighting == "target_macro" and any(not r.get("group") for r in rows):
        return _unavailable(
            base, "StereoSet native target weighting requires group metadata"
        )
    draws, bootstrap_reasons = _draws(rows, cells, bootstrap, rng)
    output_cells = {}
    cell_draws = {}
    for value, selected in cells.items():
        before = _point(selected, "_before", metric, weighting)
        after = _point(selected, "_after", metric, weighting)
        estimates = [
            (
                _point(selected, "_before", metric, weighting, weights),
                _point(selected, "_after", metric, weighting, weights),
            )
            for weights in draws
        ]
        complete = all(a is not None and b is not None for a, b in estimates)
        if not complete:
            bootstrap_reasons.append(f"bootstrap draw omitted support in cell {value}")
        cell_draws[value] = estimates if complete else []
        output_cells[value] = {
            "role": "target" if value in effective_targets else "non_target",
            "effect_kind": "opposite_language_transfer"
            if axis == "language" and value not in effective_targets
            else "training_language_change"
            if axis == "language"
            else "paired_change",
            "before": before,
            "after": after,
            "delta": after - before,
            "ci95": _ci([b - a for a, b in estimates]) if complete else None,
            "harm_before": _harm(before, metric),
            "harm_after": _harm(after, metric),
            "harm_change": _harm(after, metric) - _harm(before, metric),
            "harm_change_ci95": _ci(
                [_harm(b, metric) - _harm(a, metric) for a, b in estimates]
            )
            if complete
            else None,
            "coverage": {
                "rows": len(selected),
                "sampling_units": len({_unit(r) for r in selected}),
                "comparison_units": len({r.get("comparison_unit") for r in selected})
                if axis != "topic"
                else None,
            },
        }
    target_rows = [r for r in rows if _value(r, axis, targets) in effective_targets]
    target_before = _point(target_rows, "_before", metric, weighting)
    target_after = _point(target_rows, "_after", metric, weighting)
    denominator = _harm(target_before, metric) - _harm(target_after, metric)
    non_targets = [value for value in values if value not in effective_targets]
    numerator = sum(max(0, output_cells[v]["harm_change"]) for v in non_targets)
    improvements = []
    ratios = []
    if draws and not bootstrap_reasons:
        for index, weights in enumerate(draws):
            a = _point(target_rows, "_before", metric, weighting, weights)
            b = _point(target_rows, "_after", metric, weighting, weights)
            improvement = _harm(a, metric) - _harm(b, metric)
            improvements.append(improvement)
            growth = sum(
                max(
                    0,
                    _harm(cell_draws[v][index][1], metric)
                    - _harm(cell_draws[v][index][0], metric),
                )
                for v in non_targets
            )
            ratios.append(growth / improvement if improvement > 0 else None)
    interval = _ci(improvements)
    reasons = []
    if not non_targets:
        reasons.append("no non-target comparison cells")
    if threshold is None:
        reasons.append("minimum practical target improvement was not declared")
    elif denominator < threshold:
        reasons.append("target improvement is below the declared practical minimum")
    if denominator <= 0:
        reasons.append("target harm did not decrease")
    if interval is None or interval[0] <= 0:
        reasons.append(
            "paired confidence interval does not support positive target improvement"
        )
    return {
        **base,
        "status": "available",
        "unavailable_reasons": [],
        "comparison_unavailable_reasons": []
        if non_targets
        else [
            "opposite language view absent"
            if axis == "language"
            else "no non-target comparison cells"
        ],
        "metric": metric["id"],
        "unit": metric["unit"],
        "harm_definition": metric["harm"],
        "native_weighting": weighting,
        "cells": output_cells,
        "coverage": {
            "input_rows": len(original),
            "matched_rows": len(rows),
            "excluded_rows": len(original) - len(rows),
        },
        "unmatched": unmatched,
        "bootstrap": {
            "requested": bootstrap,
            "completed": len(draws) if not bootstrap_reasons else 0,
            "unavailable_reasons": list(dict.fromkeys(bootstrap_reasons)),
            "caveat": _PRECISION,
        },
        "redistribution": {
            "numerator": numerator,
            "denominator": denominator,
            "denominator_ci95": interval,
            "minimum_target_improvement": threshold,
            "target_harm_before": _harm(target_before, metric),
            "target_harm_after": _harm(target_after, metric),
            "ratio": numerator / denominator if not reasons else None,
            "ratio_ci95": _ci(ratios)
            if not reasons and all(v is not None for v in ratios)
            else None,
            "ratio_ci95_unavailable_reason": None
            if not reasons and all(v is not None for v in ratios)
            else "ratio eligibility failed or a paired bootstrap denominator was nonpositive",
            "ratio_unavailable_reasons": reasons,
            "numerator_definition": "sum of positive non-target cell harm changes",
        },
    }


def transfer_report(
    pairs, intervention, *, bootstrap=0, seed=42, min_target_improvement=None
):
    """Return M3a actor, M3b reviewed topic, M3c demographics, M3d language.

    ``target_values`` declares actor/topic/gender/age targets; topic values are
    reviewed topic names, with non-target cells supplied as reviewed near/far
    distances. ``training_language`` declares the language target. Each other
    axis is reported per benchmark, protocol, condition, and language; language
    effects join the two language views only on explicit matched content.

    Default harm is abs(native SS - 50), in percentage points. Explicit QA
    ``harm_metric={id, condition, unit: 'rate'}`` treats the named bounded rate as
    higher-is-worse and never pools it with association scores. An unset
    practical threshold, missing paired CI, or nonpositive improvement leaves
    RI null while retaining the paired effects and its numerator/denominator.
    Neither positive effects nor a computable ratio constitute an acceptance
    decision. Inputs are not mutated.
    """
    if not isinstance(bootstrap, int) or isinstance(bootstrap, bool) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    if min_target_improvement is not None and (
        not isinstance(min_target_improvement, (int, float))
        or isinstance(min_target_improvement, bool)
        or not math.isfinite(min_target_improvement)
        or min_target_improvement < 0
    ):
        raise ValueError(
            "minimum practical improvement must be a finite nonnegative value"
        )
    language = intervention.get("training_language")
    if language is not None and language not in ("uk", "en"):
        raise ValueError("training_language must be uk or en")
    result = {
        "schema": "fair_uk_transfer_report_v1",
        "M3a": [],
        "M3b": [],
        "M3c": {"gender": [], "age": []},
        "M3d": [],
        "interpretation": "Paired pre/post effects; language transfer is the opposite-language change, never a baseline language gap.",
        "acceptance_decision": None,
    }
    rng = np.random.default_rng(seed)
    for (name, protocol), rows in sorted(
        _prepare(pairs).items(), key=lambda item: str(item[0])
    ):
        metric, reason = _metric(rows, intervention)
        for axis, destination in (
            ("actor", result["M3a"]),
            ("topic", result["M3b"]),
            ("gender", result["M3c"]["gender"]),
            ("age", result["M3c"]["age"]),
            ("language", result["M3d"]),
        ):
            targets = _targets(intervention, axis)
            contexts = defaultdict(list)
            for row in rows:
                if (
                    metric
                    and metric["condition"]
                    and row.get("condition") != metric["condition"]
                ):
                    continue
                contexts[
                    row.get("condition"),
                    None if axis == "language" else row.get("language"),
                ].append(row)
            if not contexts:
                contexts[(metric["condition"] if metric else None, None)] = []
            for (condition, context_language), selected in sorted(
                contexts.items(), key=lambda item: str(item[0])
            ):
                base = {
                    "benchmark": name,
                    "protocol": protocol,
                    "axis": axis,
                    "condition": condition,
                    "language": context_language,
                    "target_values": targets,
                    "tasks": sorted({r["task"] for r in selected}),
                }
                if reason or not selected:
                    destination.append(
                        _unavailable(
                            base, reason or "no rows in the declared QA condition"
                        )
                    )
                elif any(
                    not r.get("stratum") or not r.get("cluster") for r in selected
                ):
                    destination.append(
                        _unavailable(
                            base, "missing explicit stratum/cluster sampling metadata"
                        )
                    )
                else:
                    destination.append(
                        _axis_report(
                            selected,
                            axis,
                            targets,
                            metric,
                            bootstrap,
                            rng,
                            min_target_improvement,
                            base,
                        )
                    )
    return result
