"""Readable reports and experiment tables; benchmark scales stay separate."""

import csv
import hashlib
import json


def fmt(value):
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).replace("|", " / ").replace("\n", " ")


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *(
                "| " + " | ".join(fmt(v).replace("|", " / ") for v in row) + " |"
                for row in rows
            ),
        ]
    )


def model_label(report):
    identity = report["provenance"].get("model_identity")
    if isinstance(identity, dict):
        return (
            identity.get("repository")
            or identity.get("model_snapshot")
            or identity.get("parameters", {}).get("pretrained", "local checkpoint")
        )
    return "external predictions (model unverified)"


def native_entries(report):
    for record in report["native"]:
        scope = " / ".join(
            str(record[k])
            for k in ("scope", "stratum", "condition", "status", "group")
            if k in record
        )
        for metric, value in record.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                # The native adapters retain their reference score scales.
                unit = (
                    "rate"
                    if report["task"].startswith("warbias_")
                    else "percent / percentage points"
                )
                if metric in ("n", "source_cases", "non_unknown_mass"):
                    unit = "count / selection mass"
                elif metric.endswith("tie_rate"):
                    unit = "rate"
                yield scope, metric, value, unit


def markdown(report):
    lines = [
        f"# Fair-UK — {report['task']}",
        "",
        f"Model: {model_label(report)}. Language: `{report['language']}`. Protocol: `{report['protocol']}`.",
        "",
        f"**Pilot, human-unvalidated.** {report['rows']} rows; {report['source_cases']} source cases. Partial run: {report['partial_run']}.",
        "",
        "Native scores retain their benchmark-specific scales. Group comparisons use equal source-case weights within each group.",
        "",
        table(["Native scope", "Metric", "Value", "Unit"], native_entries(report)),
        "",
    ]
    if "scoring_policy" in report:
        diagnostics = report["answer_diagnostics"]
        lines[4:4] = [
            f"Answer scoring: `{report['scoring_policy']}`. Strict-format accuracy: {fmt(diagnostics['strict_accuracy'])}; format-violation rate: {fmt(diagnostics['format_violation_rate'])}.",
            "Formatting diagnostics are separate from semantic invalid answers and fairness outcomes.",
            "",
        ]
    rows = []
    for c in report["comparisons"]:
        scope = " / ".join(
            str(c[k])
            for k in ("condition", "stratum", "score_group")
            if c[k] is not None
        )
        rows.append(
            (
                scope,
                c["metric"],
                c["worst"],
                (c["ci95"] or {}).get("worst"),
                ", ".join(c["worst_groups"]),
                c["gap"],
                c["minmax_ratio"],
                ", ".join(c["missing_groups"]),
            )
        )
    lines += [
        table(
            [
                "Scope",
                "Metric",
                "Worst available rate",
                "95% interval",
                "Group(s)",
                "Gap",
                "Min/max",
                "Missing groups",
            ],
            rows,
        ),
        "",
        "All subgroup rates, numerators, denominators and source-case counts are in `groups.csv` and `report.json`. Semantic item results are in `records.jsonl`.",
        "",
        "A parity ratio of one does not imply low harm. Zero/zero ratios are undefined. Intervals are exploratory source-case bootstrap intervals, not simultaneous coverage guarantees. Translations and demographic variants are not independent source cases.",
        "",
    ]
    if report["excluded_design_groups"]:
        lines += [
            "Design exclusions: " + ", ".join(report["excluded_design_groups"]),
            "",
        ]
    return "\n".join(lines)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def group_rows(report):
    for c in report["comparisons"]:
        for group, values in c["groups"].items():
            yield {
                "model": model_label(report),
                "task": report["task"],
                "language": report["language"],
                "condition": c["condition"],
                "stratum": c["stratum"],
                "score_group": c["score_group"],
                "metric": c["metric"],
                "direction": c["direction"],
                "group": group,
                **{
                    k: json.dumps(v) if isinstance(v, list) else v
                    for k, v in values.items()
                },
            }


def paired_markdown(report):
    rows = []
    for c in report["comparisons"]:
        for group, value in c["groups"].items():
            rows.append(
                (
                    c["condition"],
                    c["metric"],
                    group,
                    value["source_cases"],
                    value["uk"],
                    value["en"],
                    value["delta_en_minus_uk"],
                    value["ci95"],
                )
            )
    worst = [
        (
            c["condition"],
            c["metric"],
            c["worst_available"]["uk"],
            c["worst_available"]["en"],
            c["worst_delta_en_minus_uk"],
            c["worst_delta_ci95"],
        )
        for c in report["comparisons"]
    ]
    return "\n".join(
        [
            "# Fair-UK — paired WarBias language comparison",
            "",
            f"**EN minus UK**, in rate units. {report['paired_rows']} matched items; {report['source_cases']} source cases. Partial run: {report['partial_run']}.",
            "",
            f"Answer scoring: `{report.get('scoring_policy', 'legacy / unspecified')}`. Both languages are rescored under this policy.",
            "",
            "Positive deltas mean a higher English rate; improvement depends on the metric. Worst groups can differ between languages. Their extrema are recomputed in each paired bootstrap replicate.",
            "",
            table(
                [
                    "Condition",
                    "Metric",
                    "Worst UK",
                    "Worst EN",
                    "Delta",
                    "95% interval",
                ],
                worst,
            ),
            "",
            table(
                [
                    "Condition",
                    "Metric",
                    "Group",
                    "Cases",
                    "UK",
                    "EN",
                    "Delta",
                    "95% interval",
                ],
                rows,
            ),
            "",
            report["bootstrap"]["limitations"],
            "",
            "Full provenance, coverage, worst-group identities and paired results: `comparison.json`. Pilot data; human validation pending.",
            "",
        ]
    )


def summarize(paths, output):
    reports, native, worst, coverage = [], [], [], []
    seen = set()
    for path in paths:
        report = json.loads(path.read_text())
        provenance = report["provenance"]
        scoring_policy = report.get("scoring_policy", "legacy / unspecified")
        key = json.dumps(
            {
                "generation": provenance,
                "scoring_policy": scoring_policy,
                "scorer_code_identity": report.get("scorer_code_identity"),
            },
            sort_keys=True,
        )
        if key in seen:
            raise ValueError("Duplicate experiment identity; select one report per run")
        seen.add(key)
        reports.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "report": report,
            }
        )
        common = {
            "model": model_label(report),
            "run_id": hashlib.sha256(key.encode()).hexdigest()[:16],
            "task": report["task"],
            "language": report["language"],
            "protocol": report["protocol"],
            "scoring_policy": scoring_policy,
            "dataset_revision": provenance["dataset"]["revision"],
            "partial_run": report["partial_run"],
        }
        coverage.append(
            {
                **common,
                "rows": report["rows"],
                "source_cases": report["source_cases"],
                "full_dataset_rows": report["full_dataset_rows"],
                "status": report["status"],
                "excluded_design_groups": json.dumps(report["excluded_design_groups"]),
            }
        )
        native.extend(
            {**common, "scope": s, "metric": m, "value": v, "unit": u}
            for s, m, v, u in native_entries(report)
        )
        for c in report["comparisons"]:
            worst.append(
                {
                    **common,
                    **{
                        k: c[k]
                        for k in (
                            "condition",
                            "stratum",
                            "score_group",
                            "metric",
                            "direction",
                            "worst",
                            "gap",
                            "minmax_ratio",
                            "coverage_complete",
                        )
                    },
                    "worst_groups": json.dumps(c["worst_groups"]),
                    "ci95": json.dumps(c["ci95"]),
                    "missing_groups": json.dumps(c["missing_groups"]),
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("coverage", coverage),
        ("native", native),
        ("worst_groups", worst),
    ):
        write_csv(output / f"{name}.csv", values)
    (output / "summary.json").write_text(
        json.dumps(
            {"tool": "Fair-UK", "reports": reports},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    (output / "summary.md").write_text(
        "\n".join(
            [
                "# Fair-UK — experiment summary",
                "",
                "WarBias UK/EN form the core study; the other benchmarks provide complementary measurements. Each task, language, model and protocol is reported separately. No cross-benchmark composite is calculated.",
                "",
                table(
                    [
                        "Model",
                        "Run",
                        "Task",
                        "Language",
                        "Answer policy",
                        "Rows",
                        "Cases",
                        "Partial",
                    ],
                    [
                        (
                            r["model"],
                            r["run_id"],
                            r["task"],
                            r["language"],
                            r["scoring_policy"],
                            r["rows"],
                            r["source_cases"],
                            r["partial_run"],
                        )
                        for r in coverage
                    ],
                ),
                "",
                table(
                    ["Model", "Run", "Task", "Scope", "Native metric", "Value", "Unit"],
                    [
                        (
                            r["model"],
                            r["run_id"],
                            r["task"],
                            r["scope"],
                            r["metric"],
                            r["value"],
                            r["unit"],
                        )
                        for r in native
                    ],
                ),
                "",
                table(
                    [
                        "Model",
                        "Run",
                        "Task",
                        "Condition",
                        "Stratum / control",
                        "Metric",
                        "Worst rate",
                        "Groups",
                        "Gap",
                        "Min/max",
                        "Missing groups",
                    ],
                    [
                        (
                            r["model"],
                            r["run_id"],
                            r["task"],
                            r["condition"],
                            f"{r['stratum']} / {r['score_group']}",
                            r["metric"],
                            r["worst"],
                            r["worst_groups"],
                            r["gap"],
                            r["minmax_ratio"],
                            r["missing_groups"],
                        )
                        for r in worst
                    ],
                ),
                "",
                "CSV tables preserve coverage, native units, condition-specific worst-group scores and intervals. `summary.json` contains every source report and its file checksum. Pilot data; human validation pending.",
                "",
            ]
        )
    )
