"""Synthetic WarBias panel inputs shared by the WarBias tests (no network, no models)."""

import hashlib
import json
import re

from lm_eval.fair_uk.data import PROTOCOLS, dataset_spec, digest, load
from lm_eval.fair_uk.warbias import rows as panel_rows
from lm_eval.fair_uk.warbias.neutral import collect
from lm_eval.fair_uk.warbias.prepare import prepare
from lm_eval.fair_uk.warbias.rows import FIELDS, task_name


# id, actor, gender, age, component, case, split
PROFILES = [
    ("WB-0001", "idp", "not_stated", "not_stated", "SG-a", "C1", "train"),
    ("WB-0002", "idp", "women", "not_stated", "SG-a", "C1", "train"),
    ("WB-0003", "idp", "men", "not_stated", "SG-a", "C1", "development"),
    ("WB-0004", "veteran", "not_stated", "not_stated", "SG-b", "C2", "evaluation"),
    ("WB-0005", "veteran", "not_stated", "young", "SG-b", "C2", "evaluation"),
    ("WB-0006", "veteran", "not_stated", "older", "SG-c", "C3", "deferred"),
]
PHASES = ("train", "development", "evaluation", "deferred")


def released(item, actor, lang):
    return {
        "id": item,
        "case_id": "WB-C" + item[-3:],
        "stereotype": f"{actor} {item} stereotype.",
        "counter_stereotype": f"{actor} {item} counter sentence that is longer.",
        "unrelated": f"Several {actor} {item} watch rain.",
        "actor": actor,
        "reporter": None,
        "context_scope": "cross_conflict_generic",
        "bias_tags": ["tag"],
        "bias_dimensions": [],
        "context_tags": [],
    }


def inventory_record(row, lang, gender, age, component, case, split):
    return {
        "record_key": f"release_triplets:{lang}:{row['id']}",
        "public_case_id": case,
        "source_case_id": "src_" + case,
        "claim_family_id": "family_" + case,
        "target_group": row["actor"],
        "gender": gender,
        "age": age,
        "attributes": {},
        "split": split,
        "split_group_id": component,
        "scope_eligible": split != "deferred",
        "exclusion_reasons": [] if split != "deferred" else ["deferred_local"],
        "source_severity_candidate": 1,
        "demographic_information": "explicit_metadata_or_verified_baseline",
        "text_sha256": "x",
        "payload": {k: row[k] for k in FIELDS},
    }


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


def panel_inputs(tmp_path, monkeypatch):
    """Released files and inventory; the pinned source hashes point at them."""
    files, inventory = {}, []
    for lang in ("uk", "en"):
        rows = []
        for item, actor, gender, age, component, case, split in PROFILES:
            row = released(item, actor, lang)
            rows.append(row)
            inventory.append(
                inventory_record(row, lang, gender, age, component, case, split)
            )
        files[lang] = write_jsonl(tmp_path / f"{lang}.jsonl", rows)
    inv = write_jsonl(tmp_path / "inventory.jsonl", inventory)
    monkeypatch.setitem(
        panel_rows.SOURCE,
        "files",
        {
            lang: (f"data/{lang}/unassigned.jsonl", digest(p))
            for lang, p in files.items()
        },
    )
    return files, inv, tmp_path


def scores_for(rows, shift=0.0):
    """Mean token log-probabilities: odd rows prefer the counter-stereotype."""
    out = []
    for i, r in enumerate(rows):
        base = -3.0 - 0.01 * i
        stereotype = base + (0.4 if i % 2 == 0 else -0.4) - shift
        out.append(
            {
                "id": r["id"],
                "scores": [stereotype, base, base - 1.0],
                "token_counts": [len(t.split()) + 1 for t in r["sentences"]],
            }
        )
    return out


def neutral_records(files, monkeypatch, invalid=("WB-0001",)):
    """Neutral variants replacing the actor word; ``invalid`` IDs keep it (rejected)."""
    from lm_eval.fair_uk.warbias import neutral

    for lang in ("uk", "en"):
        monkeypatch.setitem(neutral.STATUS, lang, re.compile(r"idp|veteran"))
    records = []
    for lang, path in files.items():
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        raw = []
        for r in rows:
            for k in FIELDS:
                text = (
                    r[k]
                    if (r["id"] in invalid and k == "stereotype")
                    else r[k].replace(r["actor"], "people")
                )
                raw.append(
                    {"id": r["id"], "language": lang, "field": k, "candidates": [text]}
                )
        records.extend(collect(rows, raw, lang))
    return records


def fake_run(directory, task, bundle, rows, predictions):
    """A saved run directory as the runner writes it (fingerprinted model)."""
    directory.mkdir(parents=True)
    write_jsonl(directory / "predictions.jsonl", predictions)
    (directory / "run.json").write_text(
        json.dumps(
            {
                "task": task,
                "dataset": dataset_spec(task, bundle),
                "protocol": PROTOCOLS["warbias_triplets"],
                "model_identity": {"repository": "org/model", "revision": "a" * 40},
                "model_backend": "vllm",
                "model_args": "pretrained=org/model",
                "code_identity": {"x": "y"},
                "seed": 42,
                "limit_clusters_per_stratum": None,
                "selected_rows": len(rows),
                "selected_ids_sha256": hashlib.sha256(
                    "\n".join(r["id"] for r in rows).encode()
                ).hexdigest(),
            }
        )
    )


def phase_runs(files, inv, tmp, records, neutral_shift=0.5):
    """Full bundle plus per-phase original and neutral runs for both languages."""
    full = prepare(files["uk"], files["en"], inv, tmp / "full", neutral=records)
    phases = {
        sp: prepare(files["uk"], files["en"], inv, tmp / sp, split=sp, neutral=records)
        for sp in PHASES
    }
    runs, neutral_runs = {"uk": [], "en": []}, {"uk": [], "en": []}
    for language in ("uk", "en"):
        for sp, bundle in phases.items():
            for is_neutral, target, shift in (
                (False, runs, 0.0),
                (True, neutral_runs, neutral_shift),
            ):
                task = task_name(language, is_neutral)
                rows = load(task, dataset_bundle=bundle)
                if not rows:
                    continue
                directory = (
                    tmp
                    / "runs"
                    / ("neutral" if is_neutral else "original")
                    / sp
                    / language
                )
                fake_run(directory, task, bundle, rows, scores_for(rows, shift))
                target[language].append((bundle, directory))
    return full, phases, runs, neutral_runs
