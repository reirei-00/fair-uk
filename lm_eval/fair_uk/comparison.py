"""Paired WarBias language comparisons from fingerprinted runs and raw predictions."""

import hashlib
import json
from collections import defaultdict
from statistics import fmean

import numpy as np

from lm_eval.fair_uk import VERSION
from lm_eval.fair_uk.answers import NORMALIZED
from lm_eval.fair_uk.data import PROTOCOLS, REGISTRY, load, select_clusters
from lm_eval.fair_uk.metrics import make_report, score_all
from lm_eval.fair_uk.provenance import code_identity


def load_run(directory, input_path=None, answer_policy=NORMALIZED):
    manifest = json.loads((directory / "run.json").read_text())
    task = manifest["task"]
    if not task.startswith("warbias_") or task not in REGISTRY:
        raise ValueError("Paired language comparisons require registered WarBias runs")
    if manifest["dataset"] != REGISTRY[task]:
        raise ValueError("Run dataset does not match the pinned registry")
    if manifest["protocol"] != PROTOCOLS["warbias"]:
        raise ValueError("Run protocol does not match the registered protocol")
    identity = manifest.get("model_identity")
    if not isinstance(identity, dict) or not (
        identity.get("local_files_sha256")
        or (identity.get("repository") and identity.get("revision"))
        or (
            identity.get("provider") in ("openai", "google")
            and identity.get("model_snapshot")
        )
    ):
        raise ValueError("Comparison requires a fingerprinted model run")
    full = load(task, input_path)
    rows = select_clusters(full, manifest["limit_clusters_per_stratum"])
    checksum = hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest()
    if (
        checksum != manifest["selected_ids_sha256"]
        or len(rows) != manifest["selected_rows"]
    ):
        raise ValueError("Selected source IDs differ from the run manifest")
    prediction_path = directory / "predictions.jsonl"
    predictions = [
        json.loads(line) for line in prediction_path.read_text().splitlines()
    ]
    if identity.get("provider") == "openai":
        from lm_eval.fair_uk.openai_runner import (
            identity as openai_identity,
            validate_prediction,
        )

        if identity != openai_identity(identity["model_snapshot"], manifest["seed"]):
            raise ValueError("OpenAI run identity differs from the supported settings")
        indexed = {r["id"]: r for r in rows}
        for prediction in predictions:
            if prediction["id"] in indexed:
                validate_prediction(indexed[prediction["id"]], prediction, identity)
    if identity.get("provider") == "google":
        from lm_eval.fair_uk.gemini_runner import (
            identity as gemini_identity,
            validate_prediction,
        )

        if identity != gemini_identity(identity["model_snapshot"], manifest["seed"]):
            raise ValueError("Gemini identity differs from supported settings")
        indexed = {r["id"]: r for r in rows}
        for prediction in predictions:
            if prediction["id"] in indexed:
                validate_prediction(indexed[prediction["id"]], prediction, identity)
    scored = score_all(rows, predictions, answer_policy)
    provenance = {
        "run": manifest,
        "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
    }
    return scored, full, provenance


def align(uk, en):
    if not uk or not en:
        raise ValueError("Cannot compare empty runs")
    left = {r["id"]: r for r in uk}
    right = {r["id"]: r for r in en}
    if len(left) != len(uk) or len(right) != len(en) or left.keys() != right.keys():
        raise ValueError("UK and EN must contain exactly the same unique item IDs")
    fields = (
        "cluster",
        "stratum",
        "panel",
        "group",
        "group_kind",
        "condition",
        "polarity",
        "gold",
        "stereotype",
        "unknown",
        "eligible",
        "protocol",
    )
    for key, row in left.items():
        other = right[key]
        if row.get("scoring_policy") != other.get("scoring_policy"):
            raise ValueError("UK and EN must use the same answer scoring policy")
        if row["language"] != "uk" or other["language"] != "en":
            raise ValueError("Expected UK followed by EN")
        if row["task"].removesuffix("_uk") != other["task"].removesuffix("_en"):
            raise ValueError("Cannot compare base and intersectional tracks")
        if any(row[field] != other[field] for field in fields):
            raise ValueError(f"{key}: bilingual source metadata differs")
    return [left[key] for key in sorted(left)], [right[key] for key in sorted(left)]


def paired_rates(uk, en, metric, direction, registered, bootstrap, seed):
    # One weight per source case is shared by both languages and every profile.
    clusters = defaultdict(lambda: defaultdict(list))
    for a, b in zip(uk, en, strict=True):
        if a.get(metric) is not None and b.get(metric) is not None:
            clusters[a["group"]][(a["stratum"], a["cluster"])].append(
                (a[metric], b[metric])
            )
    if set(clusters) - set(registered):
        raise ValueError("Observed group absent from registry")
    keys = sorted({key for group in clusters.values() for key in group})
    groups = {}
    matrices = []
    for group in registered:
        cases = {
            key: np.mean(values, axis=0) for key, values in clusters[group].items()
        }
        rates = np.mean(list(cases.values()), axis=0) if cases else [None, None]
        groups[group] = {
            "uk": rates[0],
            "en": rates[1],
            "delta_en_minus_uk": float(rates[1] - rates[0]) if cases else None,
            "source_cases": len(cases),
            "paired_rows": sum(len(v) for v in clusters[group].values()),
            "ci95": None,
        }
        matrices.append(cases)
    complete = all(g["source_cases"] for g in groups.values())
    available = {g: v for g, v in groups.items() if v["source_cases"]}
    extreme = max if direction == "harm" else min
    worst = {
        lang: extreme(v[lang] for v in available.values()) if available else None
        for lang in ("uk", "en")
    }
    output = {
        "metric": metric,
        "direction": direction,
        "groups": groups,
        "coverage_complete": complete,
        "missing_groups": [g for g, v in groups.items() if not v["source_cases"]],
        "worst_available": worst,
        "worst_groups": {
            lang: [g for g, v in available.items() if v[lang] == worst[lang]]
            for lang in ("uk", "en")
        },
        "worst_delta_en_minus_uk": worst["en"] - worst["uk"] if available else None,
        "worst_delta_ci95": None,
        "bootstrap_valid_replicates": 0,
    }
    if (
        bootstrap <= 0
        or not complete
        or any(g["source_cases"] < 2 for g in groups.values())
    ):
        return output
    rng = np.random.default_rng(seed)
    weights = np.zeros((bootstrap, len(keys)), dtype=np.int32)
    strata = defaultdict(list)
    for i, key in enumerate(keys):
        strata[key[0]].append(i)
    for indices in strata.values():
        weights[:, indices] = rng.multinomial(
            len(indices), [1 / len(indices)] * len(indices), size=bootstrap
        )
    draws = []
    for group, cases in zip(registered, matrices, strict=True):
        mask = np.array([key in cases for key in keys], dtype=float)
        values = np.array([cases.get(key, [0.0, 0.0]) for key in keys])
        denominator = (weights @ mask)[:, None]
        rates = np.divide(
            weights @ values,
            denominator,
            out=np.full((bootstrap, 2), np.nan),
            where=denominator > 0,
        )
        finite = np.isfinite(rates).all(axis=1)
        if finite.any():
            groups[group]["ci95"] = np.quantile(
                rates[finite, 1] - rates[finite, 0], [0.025, 0.975]
            ).tolist()
        draws.append(rates)
    matrix = np.array(draws)
    valid = np.isfinite(matrix).all(axis=(0, 2))
    output["bootstrap_valid_replicates"] = int(valid.sum())
    if valid.any():
        worst_draws = (
            matrix[:, valid, :].max(axis=0)
            if direction == "harm"
            else matrix[:, valid, :].min(axis=0)
        )
        output["worst_delta_ci95"] = np.quantile(
            worst_draws[:, 1] - worst_draws[:, 0], [0.025, 0.975]
        ).tolist()
    return output


def compare_rows(uk, en, registry_uk, registry_en, bootstrap=1000, seed=42):
    if bootstrap < 0:
        raise ValueError("Bootstrap count cannot be negative")
    align(registry_uk, registry_en)
    uk, en = align(uk, en)
    report = make_report(uk, registry_uk, bootstrap=0)
    comparisons = []
    for spec in report["comparisons"]:
        a = [r for r in uk if r["condition"] == spec["condition"]]
        b = [r for r in en if r["condition"] == spec["condition"]]
        result = paired_rates(
            a,
            b,
            spec["metric"],
            spec["direction"],
            spec["registered_groups"],
            bootstrap,
            seed,
        )
        result["condition"] = spec["condition"]
        comparisons.append(result)
    return {
        "tool": "Fair-UK",
        "version": VERSION,
        "report_schema_version": 2,
        "scoring_policy": report["scoring_policy"],
        "kind": "paired_warbias_language_comparison",
        "scorer_code_identity": code_identity(),
        "tasks": [uk[0]["task"], en[0]["task"]],
        "status": "pilot_human_unvalidated",
        "delta": "EN minus UK",
        "paired_rows": len(uk),
        "source_cases": len({r["cluster"] for r in uk}),
        "partial_run": len(uk) != len(registry_uk),
        "excluded_design_groups": report["excluded_design_groups"],
        "choice_disagreement_rate": fmean(
            a["choice_mass"] != b["choice_mass"] for a, b in zip(uk, en, strict=True)
        ),
        "bootstrap": {
            "replicates": bootstrap,
            "seed": seed,
            "method": "paired stratified source-case percentile; language extrema recomputed per replicate",
            "limitations": "Exploratory, not simultaneous coverage. Zero-event samples can give degenerate intervals. Two cases per registered group required. Language differences do not establish a causal language effect.",
        },
        "comparisons": comparisons,
    }


def compare_runs(
    uk_dir,
    en_dir,
    uk_input=None,
    en_input=None,
    bootstrap=1000,
    seed=42,
    answer_policy=NORMALIZED,
):
    uk, full_uk, provenance_uk = load_run(uk_dir, uk_input, answer_policy)
    en, full_en, provenance_en = load_run(en_dir, en_input, answer_policy)
    a, b = provenance_uk["run"], provenance_en["run"]
    for key in (
        "model_identity",
        "model_backend",
        "code_identity",
        "batch_size",
        "protocol",
        "seed",
        "version",
        "harness_version",
        "runtime_versions",
    ):
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f"Paired runs must have identical {key}")
    result = compare_rows(uk, en, full_uk, full_en, bootstrap, seed)
    result["provenance"] = {"uk": provenance_uk, "en": provenance_en}
    return result
