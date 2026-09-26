"""Paired language comparisons with native metrics and separate group analyses."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np

from lm_eval.fair_uk import VERSION
from lm_eval.fair_uk.answers import NORMALIZED
from lm_eval.fair_uk.capabilities import benchmark
from lm_eval.fair_uk.data import (
    PROTOCOLS,
    dataset_spec,
    family,
    load,
    merged_registry,
    select_clusters,
)
from lm_eval.fair_uk.metrics import make_report, native_summaries, score_all
from lm_eval.fair_uk.provenance import code_identity
from lm_eval.fair_uk.reporting import native_entries


def load_run(
    directory,
    input_path=None,
    answer_policy=NORMALIZED,
    dataset_bundle=None,
    judgments_path=None,
):
    manifest = json.loads((directory / "run.json").read_text())
    task = manifest["task"]
    if task not in merged_registry(dataset_bundle):
        raise ValueError(
            "Paired language comparisons require registered runs or a verified bundle"
        )
    if manifest["dataset"] != dataset_spec(task, dataset_bundle):
        raise ValueError("Run dataset does not match the pinned registry")
    if manifest["protocol"] != PROTOCOLS[family(task)]:
        raise ValueError("Run protocol does not match the registered protocol")
    identity = manifest.get("model_identity")
    if not isinstance(identity, dict) or not (
        identity.get("local_files_sha256")
        or (identity.get("repository") and identity.get("revision"))
        or (
            identity.get("provider")
            in ("openai", "google", "openai-compatible", "gemini")
            and identity.get("model_snapshot")
        )
    ):
        raise ValueError("Comparison requires a fingerprinted model run")
    full = load(task, input_path, dataset_bundle=dataset_bundle)
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
    if manifest.get("model_backend") == "hosted-api":
        from lm_eval.fair_uk.api_runner import (
            identity as hosted_identity,
            validate_prediction,
        )

        if identity != hosted_identity(
            manifest.get("hosted_config"), manifest["protocol"]
        ):
            raise ValueError("Hosted run identity differs from its configuration")
        source_fingerprint = hashlib.sha256(
            json.dumps(
                rows, ensure_ascii=False, sort_keys=True, allow_nan=False
            ).encode()
        ).hexdigest()
        if manifest.get("source_rows_sha256") != source_fingerprint:
            raise ValueError("Hosted source rows differ from the saved fingerprint")
        indexed = {r["id"]: r for r in rows}
        for prediction in predictions:
            if prediction["id"] in indexed:
                validate_prediction(indexed[prediction["id"]], prediction, identity)
    if judgments_path is not None:
        if family(task) != "warbias_benign":
            raise ValueError("Rubric judgments apply only to benign requests")
        from lm_eval.fair_uk.benign import attach_judgments

        predictions = attach_judgments(
            rows,
            predictions,
            [
                json.loads(line)
                for line in Path(judgments_path).read_text().splitlines()
            ],
        )
    scored = score_all(rows, predictions, answer_policy)
    provenance = {
        "run": manifest,
        "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
    }
    if judgments_path is not None:
        provenance["judgments_sha256"] = hashlib.sha256(
            Path(judgments_path).read_bytes()
        ).hexdigest()
    if manifest.get("model_backend") in (
        "hosted-api",
        "openai",
        "gemini",
    ) or identity.get("provider"):
        returned = [
            (p.get("api_response") or {}).get("model")
            or (p.get("api_response") or {}).get("modelVersion")
            for p in predictions
        ]
        provenance["returned_models"] = sorted(
            {version for version in returned if version}
        )
        provenance["responses_without_model_version"] = sum(
            not version for version in returned
        )
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
        "score_group",
        "category",
        "expansion_version",
        "comparison_unit",
        "actor",
        "gender",
        "age",
        "split",
        "split_group_id",
        "target_mention_position",
        "rubric_sha256",
    )
    for key, row in left.items():
        other = right[key]
        if row.get("scoring_policy") != other.get("scoring_policy"):
            raise ValueError("UK and EN must use the same answer scoring policy")
        if row["language"] != "uk" or other["language"] != "en":
            raise ValueError("Expected UK followed by EN")
        if benchmark(row["task"]) != benchmark(other["task"]):
            raise ValueError("Cannot compare different benchmark tracks")
        if any(row.get(field) != other.get(field) for field in fields):
            raise ValueError(f"{key}: bilingual source metadata differs")
    return [left[key] for key in sorted(left)], [right[key] for key in sorted(left)]


def paired_rates(uk, en, metric, direction, registered, bootstrap, seed):
    # One weight per source case is shared by both languages and every profile.
    clusters = defaultdict(lambda: defaultdict(list))
    source_units = defaultdict(set)
    for a, b in zip(uk, en, strict=True):
        if a.get(metric) is not None and b.get(metric) is not None:
            source_units[a["group"]].add(a.get("independent_case_unit", a["cluster"]))
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
            "source_cases": len(source_units[group]),
            "sampling_units": len(cases),
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
        or any(g["sampling_units"] < 2 for g in groups.values())
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


def paired_native(uk, en, bootstrap, seed):
    """Recompute native statistics on paired case draws, including nonlinear scores."""
    if family(uk[0]["task"]) == "warbias_benign":
        # Judge coverage can differ by language. Native utility/refusal deltas
        # must use the same scorable responses on both sides, like paired_rates.
        left, right = [], []
        for a, b in zip(uk, en, strict=True):
            if not a["judgment_coverage"] or not b["judgment_coverage"]:
                missing = {
                    key: None
                    for key in ("task_success", "full_refusal_rate", "any_refusal_rate")
                }
                a, b = {**a, **missing}, {**b, **missing}
            left.append(a)
            right.append(b)
        uk, en = left, right

    def values(rows):
        return {
            (scope, metric, unit): value
            for scope, metric, value, unit in native_entries(
                {"task": rows[0]["task"], "native": native_summaries(rows)}
            )
            if unit not in ("count", "count / selection mass")
        }

    a, b = values(uk), values(en)
    if a.keys() != b.keys():
        raise ValueError("Native metric scopes differ between matched languages")
    output = {
        key: {
            "scope": key[0],
            "metric": key[1],
            "unit": key[2],
            "uk": a[key],
            "en": b[key],
            "delta_en_minus_uk": b[key] - a[key]
            if a[key] is not None and b[key] is not None
            else None,
            "ci95": None,
            "bootstrap_valid_replicates": 0,
        }
        for key in a
    }
    clusters = defaultdict(list)
    for i, row in enumerate(uk):
        clusters[(row["stratum"], row["cluster"])].append(i)
    strata = defaultdict(list)
    for key in clusters:
        strata[key[0]].append(key)
    # Withhold intervals when a resampling stratum has no independent replication.
    if bootstrap <= 0 or any(len(keys) < 2 for keys in strata.values()):
        return list(output.values())
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in output}
    for _ in range(bootstrap):
        left, right = [], []
        for keys in strata.values():
            for draw, index in enumerate(rng.integers(len(keys), size=len(keys))):
                key = keys[int(index)]
                for row_index in clusters[key]:
                    for source, target in ((uk, left), (en, right)):
                        row = source[row_index]
                        # Duplicated source draws must preserve WinoBias's paired correctness.
                        # Distinct draw IDs do not change the reported independent support.
                        if row.get("expansion_version"):
                            row = {
                                **row,
                                "cluster": f"{key[0]}:{draw}:{row['cluster']}",
                            }
                        elif family(row["task"]).startswith("winobias"):
                            row = {**row, "panel": f"{key[0]}:{draw}:{row['panel']}"}
                        target.append(row)
        x, y = values(left), values(right)
        for key, distribution in draws.items():
            if x.get(key) is not None and y.get(key) is not None:
                distribution.append(y[key] - x[key])
    for key, distribution in draws.items():
        output[key]["bootstrap_valid_replicates"] = len(distribution)
        if distribution:
            output[key]["ci95"] = np.quantile(distribution, [0.025, 0.975]).tolist()
    return list(output.values())


def compare_rows(uk, en, registry_uk, registry_en, bootstrap=1000, seed=42):
    if bootstrap < 0:
        raise ValueError("Bootstrap count cannot be negative")

    def eligible(rows):
        return [r for r in rows if r.get("paired_eligible", True)]

    coverage = {
        "uk": {
            "evaluated_rows": len(uk),
            "full_dataset_rows": len(registry_uk),
            "excluded_unpaired_rows": len(uk) - len(eligible(uk)),
        },
        "en": {
            "evaluated_rows": len(en),
            "full_dataset_rows": len(registry_en),
            "excluded_unpaired_rows": len(en) - len(eligible(en)),
        },
        "policy": "Only explicitly declared unpaired rows are excluded; all remaining IDs and semantic metadata must match exactly.",
    }
    full_uk, _ = align(eligible(registry_uk), eligible(registry_en))
    uk, en = align(eligible(uk), eligible(en))
    report = make_report(uk, full_uk, bootstrap=0)
    if family(uk[0]["task"]) == "warbias_benign":
        other_report = make_report(en, en, bootstrap=0)
        if report.get("judge_identity") != other_report.get("judge_identity"):
            raise ValueError(
                "Paired benign reports require identical judge identities/settings"
            )
        validate_model_versions(
            report["judge_version_evidence"], other_report["judge_version_evidence"]
        )
    comparisons = []
    for spec in report["comparisons"]:

        def matches(row, spec=spec):
            return (
                row["condition"] == spec["condition"]
                and (spec["stratum"] is None or row["stratum"] == spec["stratum"])
                and (
                    spec["score_group"] is None
                    or row.get("score_group") == spec["score_group"]
                )
            )

        a, b = [r for r in uk if matches(r)], [r for r in en if matches(r)]
        result = paired_rates(
            a,
            b,
            spec["metric"],
            spec["direction"],
            spec["registered_groups"],
            bootstrap,
            seed,
        )
        result.update(
            condition=spec["condition"],
            stratum=spec["stratum"],
            score_group=spec["score_group"],
        )
        comparisons.append(result)
    choices = [
        (a, b)
        for a, b in zip(uk, en, strict=True)
        if "choice_mass" in a and "choice_mass" in b
    ]
    return {
        "tool": "Fair-UK",
        "version": VERSION,
        "report_schema_version": 2,
        "scoring_policy": report.get("scoring_policy"),
        "kind": "paired_benchmark_language_comparison",
        "scorer_code_identity": code_identity(),
        "metric_catalog": report.get("metric_catalog"),
        "tasks": [uk[0]["task"], en[0]["task"]],
        "status": "pilot_human_unvalidated",
        "delta": "EN minus UK",
        "paired_rows": len(uk),
        "source_cases": len({r.get("independent_case_unit", r["cluster"]) for r in uk}),
        "sampling_units": len({r["cluster"] for r in uk}),
        "partial_run": len(uk) != len(full_uk),
        "coverage": coverage,
        **(
            {
                "native_missingness_policy": "Utility/refusal metrics use only responses scorable in both languages; judgment coverage retains all paired responses."
            }
            if family(uk[0]["task"]) == "warbias_benign"
            else {}
        ),
        "excluded_design_groups": report["excluded_design_groups"],
        "choice_disagreement_rate": fmean(
            a["choice_mass"] != b["choice_mass"] for a, b in choices
        )
        if choices
        else None,
        "bootstrap": {
            "replicates": bootstrap,
            "seed": seed,
            "method": "paired stratified source-case percentile; native metrics and language extrema recomputed per replicate",
            "limitations": "Exploratory, not simultaneous coverage. Zero-event samples can give degenerate intervals. Native intervals require at least two cases per resampling stratum. Language differences do not establish a causal language effect; translation and adaptation may differ.",
        },
        "native_comparisons": paired_native(uk, en, bootstrap, seed),
        "comparisons": comparisons,
    }


def validate_model_versions(uk, en):
    if "returned_models" not in uk and "returned_models" not in en:
        return {"status": "checkpoint_identity_checked"}
    left, right = set(uk.get("returned_models", [])), set(en.get("returned_models", []))
    if len(left) > 1 or len(right) > 1 or (left and right and left != right):
        raise ValueError(
            "Paired hosted runs have different or mixed returned model versions"
        )
    complete = bool(left and right) and not (
        uk.get("responses_without_model_version", 0)
        or en.get("responses_without_model_version", 0)
    )
    return {
        "status": "matching_returned_identifiers"
        if complete
        else "version_evidence_incomplete",
        "uk": sorted(left),
        "en": sorted(right),
        "note": "Provider-returned identifiers are recorded evidence, not independent weight hashes; a mutable alias cannot guarantee identical weights.",
    }


def compare_runs(
    uk_dir,
    en_dir,
    uk_input=None,
    en_input=None,
    bootstrap=1000,
    seed=42,
    answer_policy=NORMALIZED,
    dataset_bundle=None,
    uk_judgments=None,
    en_judgments=None,
):
    uk, full_uk, provenance_uk = load_run(
        uk_dir, uk_input, answer_policy, dataset_bundle, uk_judgments
    )
    en, full_en, provenance_en = load_run(
        en_dir, en_input, answer_policy, dataset_bundle, en_judgments
    )
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
    if a.get("generation_config") != b.get("generation_config"):
        raise ValueError("Paired runs must have identical generation configuration")
    model_versions = validate_model_versions(provenance_uk, provenance_en)
    result = compare_rows(uk, en, full_uk, full_en, bootstrap, seed)
    result["model_version_comparison"] = model_versions
    result["pairing_notes"] = {
        "uk": a["dataset"].get("pairing"),
        "en": b["dataset"].get("pairing"),
    }
    result["provenance"] = {"uk": provenance_uk, "en": provenance_en}
    return result
