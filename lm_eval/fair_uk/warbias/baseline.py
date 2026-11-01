"""Analyze saved WarBias triplet/QA baselines without model inference.

Use ``release_overview`` to verify and inspect frozen inputs,
``metric_definitions`` for definitions and units, ``analyze_baselines`` for
saved-run analysis, and ``write_baseline_results`` for JSON/CSV exports.
"""

from pathlib import Path

from lm_eval.fair_uk.warbias.baseline_inputs import (
    SCORING_SETTINGS,
    _release_inputs,
    read,
    release_overview,
    sha,
    verify_baseline_run,
)
from lm_eval.fair_uk.warbias.baseline_reporting import (
    metric_definitions,
    write_baseline_results,
)
from lm_eval.fair_uk.warbias.baseline_summary import (
    QA_FIELDS,
    TRIPLET_FIELDS,
    evidence_gap,
    group_diagnostics,
    paired_summary,
    qa_records,
    qa_summary,
    scope_selections,
    summary,
    triplet_items,
    verify_qa_formula,
)


__all__ = [
    "SCORING_SETTINGS",
    "analyze_baselines",
    "metric_definitions",
    "release_overview",
    "verify_baseline_run",
    "write_baseline_results",
]


def _analyse_model(release, results, model, profiles, severity, draws, seed):
    from lm_eval.fair_uk.data import family

    bundle = release / "inputs/manifest.json"
    tasks = read(bundle)["datasets"]
    root = results / model
    expected = (
        profiles[model].get("parameters")
        or profiles[model]["model_identity"]["parameters"]
    )
    missing = [
        task
        for task in tasks
        if not all(
            (root / "full" / task / name).is_file()
            for name in ("run.json", "predictions.jsonl", "report.json")
        )
    ]
    if missing:
        return {"status": "pending_collection", "missing_tasks": missing}
    binding, qualification = (
        read(root / "binding.json"),
        read(root / "qualification.json"),
    )
    if not qualification.get("passed") or qualification.get("binding_sha256") != sha(
        root / "binding.json"
    ):
        raise ValueError("Full collection lacks a matching repeated qualification")
    if (
        binding.get("bundle_sha256") != sha(bundle)
        or binding.get("models_sha256") != sha(release / "models.json")
        or binding.get("model") != model
        or binding.get("settings") != SCORING_SETTINGS
    ):
        raise ValueError("Collection binding differs from the selected release")
    runs = {
        task: verify_baseline_run(
            root / "full" / task,
            task,
            bundle,
            expected,
            profiles[model]["model_identity"],
            smoke=False,
        )
        for task in tasks
    }
    if any(len(run.selected_rows) != len(run.source_rows) for run in runs.values()):
        raise ValueError("A full-panel analysis received a source subset")
    provenance = {
        task: {
            "run_json_sha256": sha(root / "full" / task / "run.json"),
            "predictions_sha256": run.provenance["predictions_sha256"],
        }
        for task, run in runs.items()
    }
    indexed = {
        (family(task), tasks[task]["language"]): run for task, run in runs.items()
    }
    report = {
        "status": "complete",
        "runs": provenance,
        "triplets": {},
        "qa": {},
        "language_contrasts": [],
        "group_diagnostics": {},
    }
    for scoring in ("per_token", "per_byte", "sum"):
        languages = {
            language: triplet_items(
                indexed["warbias_triplets", language], severity, scoring
            )
            for language in ("uk", "en")
        }
        report["group_diagnostics"][scoring] = {
            language: group_diagnostics(items, scoring, draws, seed)
            for language, items in languages.items()
        }
        report["triplets"][scoring] = {
            language: [
                {
                    **scope,
                    **summary(
                        selected, TRIPLET_FIELDS, "triplets", draws=draws, seed=seed
                    ),
                }
                for scope, selected in scope_selections(items)
            ]
            for language, items in languages.items()
        }
        for scope, left in scope_selections(languages["uk"]):
            ids = {row["id"] for row in left}
            right = [row for row in languages["en"] if row["id"] in ids]
            report["language_contrasts"].append(
                {
                    "format": "triplets",
                    "scoring": scoring,
                    **scope,
                    **paired_summary(
                        left, right, TRIPLET_FIELDS, "triplets", None, draws, seed
                    ),
                }
            )
    languages = {
        language: qa_records(indexed["warbias", language], severity)
        for language in ("uk", "en")
    }
    for language, items in languages.items():
        scopes = []
        for condition in ("ambiguous", "stereotype_aligned", "stereotype_conflicting"):
            subset = [row for row in items if row["condition"] == condition]
            scopes += [
                {
                    **scope,
                    "condition": condition,
                    **qa_summary(selected, condition, draws, seed),
                }
                for scope, selected in scope_selections(subset)
            ]
        verify_qa_formula(indexed["warbias", language], scopes)
        report["qa"][language] = {
            "gender_age_contrasts": {
                "status": "unavailable",
                "reason": "QA scenarios do not provide matched explicit gender and age profiles.",
            },
            "scopes": scopes,
            "aligned_minus_conflicting_accuracy": evidence_gap(items, draws, seed),
        }
    for condition in ("ambiguous", "stereotype_aligned", "stereotype_conflicting"):
        for scope, left in scope_selections(
            [row for row in languages["uk"] if row["condition"] == condition]
        ):
            ids = {row["id"] for row in left}
            right = [row for row in languages["en"] if row["id"] in ids]
            report["language_contrasts"].append(
                {
                    "format": "qa",
                    "condition": condition,
                    **scope,
                    **paired_summary(
                        left, right, QA_FIELDS, "qa", condition, draws, seed
                    ),
                }
            )
    return report


def analyze_baselines(release, results, models=None, bootstrap=1000, seed=42):
    """Analyze complete saved baselines with native scoring and severity slices.

    ``release`` contains ``models.json`` and the frozen ``inputs/`` bundle.
    ``results`` contains ``MODEL/full/TASK`` plus runner qualification receipts.
    Missing task artifacts remain pending; inconsistent artifacts fail validation.
    All variants are averaged within their original claim component before
    between-component averaging and 95% percentile bootstrap resampling.
    """
    if type(bootstrap) is not int or bootstrap < 0 or type(seed) is not int:
        raise ValueError(
            "Use a nonnegative integer bootstrap count and an integer seed"
        )
    release, manifest, profiles, _rows, severity, binding = _release_inputs(release)
    results = Path(results).resolve()
    models = list(profiles) if models is None else list(models)
    if (
        not models
        or len(models) != len(set(models))
        or not set(models) <= profiles.keys()
    ):
        raise ValueError("Select unique known model IDs")
    report = {
        "kind": "warbias_final_baseline",
        "schema_version": 1,
        "scope": "Unchanged checkpoint on the full released panel; not held-out intervention evidence.",
        "units": {
            "rates": "proportion",
            "signed_bias": "[-1,1]",
            "association_margin": {
                "per_token": "nats/token",
                "per_byte": "nats/UTF-8 byte",
                "sum": "nats",
            },
        },
        "weighting": "Equal original claim components after within-component averaging; keep profile and language variants together.",
        "bootstrap": {"draws": bootstrap, "seed": seed, "level": 0.95},
        "language_pair": {"lg1": "uk", "lg2": "en"},
        "definitions": metric_definitions(),
        "source": manifest.get("source"),
        "inputs": {
            "bundle_sha256": sha(release / "inputs/manifest.json"),
            "models_sha256": sha(release / "models.json"),
        },
        "primary_prior_normalised_ss": {
            "status": "unavailable",
            "reason": "Final-text bilingual neutral controls have not been constructed and reviewed.",
        },
        "severity": {
            "case_scores_sha256": sha(release / "inputs/severity/case_scores.jsonl"),
            "binding_sha256": sha(release / "inputs/severity/binding.json"),
            "binding": binding,
        },
        "models": {},
    }
    for model in models:
        try:
            report["models"][model] = _analyse_model(
                release, results, model, profiles, severity, bootstrap, seed
            )
        except (ValueError, KeyError, FileNotFoundError) as error:
            report["models"][model] = {
                "status": "validation_failed",
                "reason": str(error),
            }
    return report
