"""Offline, reproducible WarBias inventory and prospective experiment design.

Reads source data; writes only the specified preparation directory. Makes no
model calls, modifies no annotation tasks and never labels structural checks as
human/content validation. Python 3.10+, standard library only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import html
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ("stereotype", "counter_stereotype", "unrelated")
CRITERIA = ("S_ASSOCIATION", "C_RELEVANCE", "C_NO_INVERSION", "C_NATURAL", "C_NO_NEW_CLAIM", "U_INDEPENDENT", "PROFILE_MATCH", "SPEAKER_MATCH", "LANGUAGE")
NS = "not_stated"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def write_csv(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False, sort_keys=True) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in columns})


class Inputs:
    def __init__(self, root):
        self.root, self.hashes = root, {}

    def path(self, relative):
        p = self.root / relative
        self.hashes[str(p.relative_to(self.root))] = sha(p)
        return p

    def obj(self, relative):
        return json.loads(self.path(relative).read_text())

    def rows(self, relative):
        return [json.loads(s) for s in self.path(relative).read_text().splitlines() if s.strip()]

    def csv(self, relative):
        with self.path(relative).open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def latest(self, directory):
        """Follow the original archive's attempt/revision/filename precedence.

        Retain per-ID attempt/content counts and detect ambiguous same-rank rows.
        File mtimes are deliberately irrelevant.
        """
        result, ranking = {}, {}
        stats = collections.defaultdict(lambda: {"records": 0, "hashes": set(), "same_rank_conflict": False})
        for p in sorted((self.root / directory).glob("*.jsonl")):
            rel = str(p.relative_to(self.root))
            for line, row in enumerate(self.rows(rel), 1):
                rid = row["id"]
                key = (row.get("attempt", 1), row.get("review_revision", 0), p.name)
                h = digest({k: row.get(k) for k in FIELDS}) if all(k in row for k in FIELDS) else digest(row)
                stats[rid]["records"] += 1
                stats[rid]["hashes"].add(h)
                if rid not in ranking or key > ranking[rid]:
                    result[rid] = (row, rel, line)
                    ranking[rid] = key
                elif key == ranking[rid] and digest(row) != digest(result[rid][0]):
                    stats[rid]["same_rank_conflict"] = True
        return result, stats


class Union:
    def __init__(self, values):
        self.parent = {v: v for v in values}

    def find(self, value):
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def join(self, values):
        roots = sorted({self.find(v) for v in values if v in self.parent})
        for r in roots[1:]:
            self.parent[r] = roots[0]


def normalize_gender(value):
    mapping = {None: NS, "": NS, NS: NS, "neutral": NS, "generic_plural": NS, "unspecified": NS, "masculine": "men", "feminine": "women", "men": "men", "women": "women"}
    if value not in mapping:
        raise ValueError(f"Unknown gender code: {value}")
    return mapping[value]


def normalize_age(value):
    mapping = {None: NS, "": NS, NS: NS, "unspecified": NS, "young_adult": "young", "young": "young", "middle_age": "middle_aged", "middle_aged": "middle_aged", "older_adult": "older", "older": "older"}
    if value not in mapping:
        raise ValueError(f"Unknown age code: {value}")
    return mapping[value]


def coverage(profiles):
    profiles = set(profiles)
    if not profiles:
        return "none"
    gender = any(g != NS for g, a in profiles)
    age = any(a != NS for g, a in profiles)
    if any(g != NS and a != NS for g, a in profiles):
        return "joint_gender_age"
    if gender and age:
        return "both_separately"
    return "gender_only" if gender else "age_only" if age else "baseline"


def contrast_coverage(profiles):
    """Observed explicit attributes are not necessarily demographic contrasts."""
    profiles = set(profiles)
    if not profiles:
        return "none"
    gender = any({("men", a), ("women", a)} <= profiles for a in (NS, "young", "middle_aged", "older"))
    age = any(len({a for gg, a in profiles if gg == g and a != NS}) >= 2 for g in (NS, "men", "women"))
    common_ages = {a for a in ("young", "middle_aged", "older") if {("men", a), ("women", a)} <= profiles}
    if len(common_ages) >= 2:
        return "joint_gender_age"
    if gender and age:
        return "both_separately"
    return "gender_only" if gender else "age_only" if age else "baseline"


def expected_profiles(case):
    g = case["intrinsic_attributes"].get("gender")
    a = case["intrinsic_attributes"].get("age")
    genders = [normalize_gender(g)] if g else [NS, "men", "women"]
    ages = [normalize_age(a)] if a else [NS, "young", "middle_aged", "older"]
    return [(g, a) for g, a in itertools.product(genders, ages) if not (case["target_group"] == "veteran" and g == "women" and a == "older")]


def structural_issues(row):
    values = [row.get(k) for k in FIELDS]
    if any(not isinstance(v, str) or not v.strip() for v in values):
        return ["missing_text"]
    issues = ["duplicate_fields"] if len(set(values)) != 3 else []
    for v in values:
        if v != v.strip() or "\n" in v or "[TARGET]" in v or "{target}" in v:
            issues.append("unrendered_or_whitespace")
        if len(v.split()) > 45 or not v.endswith((".", "!", "?")):
            issues.append("sentence_contract")
    return sorted(set(issues))


def archive_acceptance(row, review):
    return (not structural_issues(row) and not row.get("contract_issues") and review.get("text_hash") == digest({k: row[k] for k in FIELDS}) and review.get("decision") == "accepted" and all(review.get("criteria", {}).get(c) == "pass" for c in CRITERIA) and (row.get("language") != "uk" or review.get("criteria", {}).get("TRANSLATION_FIDELITY") == "pass"))


def build(root, out, decisions_path):
    inputs = Inputs(root)
    decisions = json.loads(decisions_path.read_text())
    inventory, sources, comparisons, unresolved = [], [], [], []
    v1 = "fair_forget_tranlsation/outputs/warbias_en_uk_v1_20260909"
    v2 = "fair_forget_tranlsation/outputs/warbias_en_uk_v2_20260910"
    lens = "fair_forget_tranlsation/outputs/lens_warbias_uk_en_v1"
    release = "outputs/warbias_exclude_older_women_veterans_20260911/hf_update"
    base_qa = "outputs/warbias_eval240_16_20260911/hf_verified"
    inter_qa = "outputs/warbias_intersectional_qa_20260911/hf_verified"
    legacy_dir = "fair_forget_tranlsation/data/stereoset_uk_extension"

    specs = {r["case_id"]: r for r in inputs.rows(v1 + "/case_specifications.jsonl")}
    legacy = {r["case_id"]: r for r in inputs.csv(legacy_dir + "/cases.csv")}
    old_matrix = inputs.csv(legacy_dir + "/matrix.csv")
    matrix1 = {r["id"]: r for r in inputs.rows(v1 + "/matrix.jsonl")}
    matrix2 = {r["id"]: r for r in inputs.rows(v2 + "/matrix.jsonl")}
    public_map = inputs.obj("outputs/warbias_hf_pr_20260910/private_id_mapping.json")
    source_to_public = {v["source_case_id"]: v["public_case_id"] for v in public_map.values()}
    assert len(source_to_public) == 127
    public_to_source = {v: k for k, v in source_to_public.items()}
    assert len(public_to_source) == len(source_to_public)
    assert set(source_to_public) == {r["case_id"] for r in matrix1.values()} == {r["case_id"] for r in matrix2.values()}
    public_cases = {r["case_id"]: r for r in inputs.rows(release + "/metadata/cases.jsonl")}
    review_meta = {(r["language"], r["id"]): r for r in inputs.rows(release + "/metadata/review_metadata.jsonl")}
    lineage = {r["id"]: r for r in inputs.rows(release + "/metadata/lineage.jsonl")}
    blocks = inputs.rows(release + "/metadata/comparison_blocks.jsonl")
    exclusions = inputs.obj(release + "/metadata/profile_exclusions.json")
    released_manifest = inputs.obj(inter_qa + "/metadata/release_manifest.json")
    for directory, config in [(release, "uk"), (release, "en"), (base_qa, "eval_uk"), (base_qa, "eval_en"), (inter_qa, "eval_intersectional_uk"), (inter_qa, "eval_intersectional_en")]:
        relative = f"data/{config}/unassigned.jsonl"
        assert sha(inputs.path(directory + "/" + relative)) == released_manifest["checksums"][relative], relative
    review_authority = "assistant_source_review_20260925_not_independent_validation"

    family_map = {}
    for group in decisions["claim_families"]:
        for n in group["cases"]:
            cid = f"WB-C{n:03d}"
            assert cid in public_to_source and cid not in family_map
            family_map[cid] = group
    severity = {}
    for level, values in decisions["severity_assignments"].items():
        for n in values:
            cid = f"WB-C{n:03d}"
            assert cid not in severity
            severity[cid] = int(level)
    assert set(severity) == set(public_to_source), (set(public_to_source) - set(severity), set(severity) - set(public_to_source))

    cases = {}
    for source_id in sorted(set(specs) | set(legacy)):
        spec, old = specs.get(source_id, {}), legacy.get(source_id, {})
        public = source_to_public.get(source_id)
        parents = [p for p in spec.get("source_case_ids", []) if p != source_id]
        inherited = [legacy[p] for p in parents if p in legacy]
        claim = spec.get("source_claim_uk", old.get("stereotype_claim_uk"))
        group = family_map.get(public, {})
        reason = decisions["scope"]["deferred_public_cases"].get(public)
        state = "eligible" if public and not reason else "deferred_local" if public else "superseded_compound" if spec.get("status") == "superseded_compound" else "legacy_outside_current_scope"
        cases[source_id] = {
            "source_case_id": source_id, "public_case_id": public,
            "claim_family_id": group.get("id", "claim_" + (public or source_id)),
            "family_relation": group.get("relation", "single_source_claim"),
            "family_note": group.get("note", "No cross-actor equivalence assigned."),
            "source_claim_uk": claim, "source_claim_sha256": digest(claim),
            "source_parent_case_ids": parents, "target_group": spec.get("target_group", old.get("bias_type_code")),
            "original_category": spec.get("bias_category", old.get("category_code")),
            "intrinsic_attributes": spec.get("intrinsic_attributes", {}),
            "harm_mechanisms_uk": sorted({r["harm_mechanism_uk"] for r in ([old] if old else inherited) if r.get("harm_mechanism_uk")}),
            "scope_status": state, "scope_reason": reason or ("Current veteran/IDP case; row-level scope checks also apply." if public else "Retained for lineage and leakage prevention; not scheduled for generation."),
            "source_severity_candidate": severity.get(public),
            "severity_rationale": decisions["severity_rubric"]["levels"].get(str(severity.get(public))),
            "severity_review_status": "assistant_provisional" if public else "not_assigned_outside_current_scope",
            "final_text_severity": None, "final_text_severity_status": "review_required",
            "mapping_review": review_authority, "human_validated": False,
        }

    def row_exclusions(case_id, attrs, scope=None):
        case = cases[case_id]
        reasons = []
        if case["scope_status"] != "eligible":
            reasons.append(case["scope_status"])
        if scope == "ukraine_context" or any(k in attrs for k in ("service_period", "displacement_wave", "language", "origin_region", "region")):
            reasons.append("deferred_local_profile")
        if attrs.get("occupation") or attrs.get("actor_occupation") or attrs.get("prior_occupation"):
            reasons.append("excluded_occupation_qualified_profile")
        if case["target_group"] == "veteran" and normalize_gender(attrs.get("gender")) == "women" and normalize_age(attrs.get("age")) == "older":
            reasons.append("excluded_older_women_veteran")
        return sorted(set(reasons))

    def add(source, rid, language, case_id, payload, attrs, **extra):
        scope = extra.pop("context_scope", None)
        kind = extra.pop("kind", "triplet")
        entry = {"record_key": f"{source}:{language}:{rid}", "source": source, "item_id": rid, "language": language, "kind": kind,
                 "source_case_id": case_id, "public_case_id": source_to_public.get(case_id), "target_group": cases[case_id]["target_group"],
                 "gender": normalize_gender(attrs.get("gender")), "age": normalize_age(attrs.get("age")),
                 "demographic_information": "explicit_metadata_or_verified_baseline", "attributes": attrs,
                 "context_scope": scope, "exclusion_reasons": row_exclusions(case_id, attrs, scope),
                 "text_sha256": digest(payload), "leakage_sha256": digest({k: payload[k] for k in FIELDS[:2]}) if kind == "triplet" else digest(payload),
                 "payload": payload, "human_validated": False, **extra}
        entry["scope_eligible"] = not entry["exclusion_reasons"]
        inventory.append(entry)
        return entry

    current = {}
    for lang in decisions["languages"]:
        rel = release + f"/data/{lang}/unassigned.jsonl"
        for i, row in enumerate(inputs.rows(rel), 1):
            rid = row["id"]
            cid = public_to_source[row["case_id"]]
            original = matrix2.get(public_map.get(rid, {}).get("source_id"))
            if original:
                attrs = {**original["intrinsic_attributes"], **original["additional_tags"]}
            else:
                assert rid in lineage, rid
                attrs = {**lineage[rid].get("intrinsic_attributes", {}), **lineage[rid]["target_attributes"]}
            meta = review_meta[lang, rid]
            # Explicit demographic fields in the released review sidecar are authoritative.
            for k, value in [("gender", meta["actor_gender"]), ("age", meta["actor_age_group"])]:
                if value is None:
                    attrs.pop(k, None)
                else:
                    attrs[k] = value
            if meta["actor_occupation"]:
                attrs["actor_occupation"] = meta["actor_occupation"]
            # Local history codes were deliberately generalized in the release.
            if row["context_scope"] == "cross_conflict_generic":
                for local in ("service_period", "displacement_wave"):
                    if local in attrs:
                        attrs[local + "_generalized"] = attrs.pop(local)
            entry = add("release_triplets", rid, lang, cid, {f: row[f] for f in FIELDS}, attrs,
                        context_scope=row["context_scope"], source_path=rel, source_line=i,
                        actor_gender_form=meta["actor_gender_form"], reporter=row["reporter"],
                        review_status="released_model_reviewed_human_validation_pending", reuse_status="preferred_released_text",
                        original_source_item_id=public_map.get(rid, {}).get("source_id"),
                        variant_parent_id=lineage.get(rid, {}).get("parent_id"))
            current[lang, rid] = entry
    assert len(current) == 2118

    for source, run, matrix in [("archive_v1", v1, matrix1), ("archive_v2", v2, matrix2)]:
        records, reviews, attempts = {}, {}, {}
        for lang in decisions["languages"]:
            records[lang], attempts[lang] = inputs.latest(run + "/records/" + lang)
            reviews[lang], _ = inputs.latest(run + "/reviews/" + lang)
        for lang in decisions["languages"]:
            for rid, meta in sorted(matrix.items()):
                attrs = {**meta["intrinsic_attributes"], **meta["additional_tags"]}
                chosen = records[lang].get(rid)
                if chosen is None:
                    # Missing texts remain visible in the matrix coverage, not invented records.
                    continue
                row, rel, line = chosen
                review = reviews[lang].get(rid, ({}, None, None))[0]
                accepted = archive_acceptance(row, review)
                aligned = lang == "en" or (rid in records["en"] and row.get("english_hash") == digest({f: records["en"][rid][0][f] for f in FIELDS}))
                conflict = attempts[lang][rid]["same_rank_conflict"]
                accepted = accepted and aligned and not conflict
                review_issues = []
                if not review:
                    review_issues.append("no_review_record_in_this_version")
                elif review.get("text_hash") != digest({f: row[f] for f in FIELDS}):
                    review_issues.append("review_refers_to_another_text_version")
                elif review.get("decision") != "accepted":
                    review_issues.append("latest_review_not_accepted")
                if not aligned:
                    review_issues.append("translation_refers_to_superseded_english")
                if conflict:
                    review_issues.append("same_rank_record_conflict")
                review_issues.extend(structural_issues(row))
                review_issues.extend(row.get("contract_issues", []))
                if not accepted and not review_issues:
                    review_issues.append("required_review_criteria_not_all_passed")
                add(source, rid, lang, meta["case_id"], {f: row[f] for f in FIELDS}, attrs,
                    context_scope=meta.get("subset"), source_path=rel, source_line=line,
                    reporter=meta["reporter"], framing=meta["framing"],
                    review_status="historical_model_review_passed" if accepted else "review_or_alignment_unresolved",
                    review_path=reviews[lang].get(rid, ({}, None, None))[1],
                    historical_acceptance=accepted, aligned_to_latest_english=aligned, review_issues=review_issues,
                    record_versions=attempts[lang][rid]["records"], distinct_text_versions=len(attempts[lang][rid]["hashes"]),
                    same_rank_conflict=conflict, reuse_status="archive_requires_current_semantic_review",
                    superseded_by_release_ids=sorted(k[1] for k, r in current.items() if k[0] == lang and r["original_source_item_id"] == rid))
        sources.append({"source": source, "matrix_items": len(matrix), "case_families": len({r["case_id"] for r in matrix.values()}),
                        "complete_text_ids_by_language": {lang: len(set(records[lang]) & set(matrix)) for lang in decisions["languages"]},
                        "note": "Latest archive versions under original precedence; historical acceptance does not authorize automatic reuse after later corrections."})

    for source, run, configuration in [("release_qa", base_qa, "eval"), ("release_intersectional_qa", inter_qa, "eval_intersectional")]:
        for lang in decisions["languages"]:
            rel = run + f"/data/{configuration}_{lang}/unassigned.jsonl"
            for line, row in enumerate(inputs.rows(rel), 1):
                cid = public_to_source[row["source_case_id"]]
                attrs = {**cases[cid]["intrinsic_attributes"]}
                if source == "release_intersectional_qa":
                    attrs.update(gender=row["target_gender"], age=row["target_age_group"])
                add(source, row["id"], lang, cid, {k: row[k] for k in ("context", "question", "choices")}, attrs,
                    kind="qa", source_path=rel, source_line=line,
                    scenario_id=row["scenario_id"], situation_id="qa:" + row["scenario_id"],
                    base_id=row.get("base_id", row["id"]), condition=row["evidence_condition"], polarity=row["question_polarity"],
                    answer_index=row["answer_index"], answer_roles={k: row[k] for k in ("target_index", "comparison_index", "unknown_index", "stereotype_consistent_index")},
                    review_status=row["review_status"], reuse_status="preferred_released_text")

    lens_pairs = inputs.rows(lens + "/prompt_pairs.jsonl")
    lens_map = collections.defaultdict(list)
    for row in lens_pairs:
        lens_map[row["unique_pair_id"]].append(row)
    lens_unique = inputs.rows(lens + "/unique_prompt_pairs.jsonl")
    assert len(lens_unique) == len(lens_map) == 3201
    for line, row in enumerate(lens_unique, 1):
        refs = lens_map[row["unique_pair_id"]]
        cids = {r["case_id"] for r in refs}
        assert len(cids) == 1, cids
        cid = next(iter(cids))
        profiles = {(normalize_gender(r["gender_track"]), normalize_age(r["age_track"])) for r in refs}
        g, a = next(iter(profiles)) if len(profiles) == 1 and row["level"] != "L3" else (NS, NS)
        for lang in decisions["languages"]:
            entry = add("lens", row["unique_pair_id"], lang, cid, {"prompt": row["prompt_" + lang]}, {"gender": g, "age": a},
                        kind="lens", source_path=lens + "/unique_prompt_pairs.jsonl", source_line=line,
                        probe_type=row["probe_type"], level=row["level"], legacy_split_group=row["split_group_id"],
                        source_matrix_ids=sorted({r["matrix_item_id"] for r in refs}),
                        review_status=row["review_status"], reuse_status="archive_requires_current_semantic_review",
                        actor_visibility="masked" if row["level"] == "L3" else "explicit")
            if any(r["target_code"] == "veteran_ato_oos" for r in refs):
                entry["exclusion_reasons"] = sorted(set(entry["exclusion_reasons"] + ["deferred_local_profile"]))
                entry["scope_eligible"] = False
            if any(r["target_code"] not in ("idp_current", "idp_unregistered", "veteran_generic", "veteran_ato_oos") for r in refs):
                entry["exclusion_reasons"] = sorted(set(entry["exclusion_reasons"] + ["legacy_actor_profile_requires_scope_review"]))
                entry["scope_eligible"] = False

    # Reconcile text duplicates without collapsing distinct record provenance.
    duplicates = collections.defaultdict(list)
    parallel = collections.defaultdict(dict)
    for row in inventory:
        duplicates[row["language"], row["text_sha256"]].append(row["record_key"])
        parallel[row["source"], row["item_id"]][row["language"]] = row
    duplicate_rows = [{"language": lang, "text_sha256": h, "record_keys": refs, "copies": len(refs)} for (lang, h), refs in sorted(duplicates.items()) if len(refs) > 1]
    for row in inventory:
        pair = parallel[row["source"], row["item_id"]]
        row["bilingual_complete"] = set(pair) == set(decisions["languages"])
        row["exact_duplicate_count"] = len(duplicates[row["language"], row["text_sha256"]])
        row["final_text_severity"] = None
        row["final_text_severity_status"] = "review_required"
        row["source_severity_candidate"] = cases[row["source_case_id"]]["source_severity_candidate"]

    # Freeze semantic and provenance components before assigning splits.
    union = Union(cases)
    for case in cases.values():
        union.join([case["source_case_id"], *case["source_parent_case_ids"]])
    for group in decisions["claim_families"] + decisions["additional_split_links"]:
        union.join(public_to_source[f"WB-C{n:03d}"] for n in group["cases"])
    for rows in itertools.groupby(sorted(lens_pairs, key=lambda r: r["split_group_id"]), key=lambda r: r["split_group_id"]):
        union.join(r["case_id"] for r in rows[1])
    hash_cases = collections.defaultdict(set)
    for row in inventory:
        hash_cases[row["language"], row["leakage_sha256"]].add(row["source_case_id"])
    for members in hash_cases.values():
        union.join(members)
    components = collections.defaultdict(list)
    for cid in sorted(cases):
        components[union.find(cid)].append(cid)
    split_groups = []
    for members in components.values():
        active = [cid for cid in members if cases[cid]["scope_status"] == "eligible"]
        split_groups.append({"split_group_id": "SG-" + digest(members)[:16], "source_case_ids": members,
                             "eligible_case_count": len(active), "status_counts": dict(collections.Counter(cases[c]["target_group"] for c in active)),
                             "reserved_split": "train" if any(cases[c]["claim_family_id"] in decisions.get("reserve_train_claim_families", []) for c in active) else None})
    allocate_splits(split_groups, decisions)
    for group in split_groups:
        for cid in group["source_case_ids"]:
            cases[cid]["split_group_id"] = group["split_group_id"]
            cases[cid]["split"] = group["split"] if cases[cid]["scope_status"] == "eligible" else "deferred"
            cases[cid]["reserved_component_split"] = group["split"]
    for row in inventory:
        case = cases[row["source_case_id"]]
        row["split_group_id"] = case["split_group_id"]
        row["claim_family_id"] = case["claim_family_id"]
        row["split"] = case["split"] if row["scope_eligible"] else "deferred"
        row["reserved_component_split"] = case["reserved_component_split"]

    finish(root, out, inputs, decisions, decisions_path, cases, inventory, sources, duplicate_rows, split_groups, parallel, blocks, old_matrix, specs, public_to_source, unresolved)


def allocate_splits(groups, decisions):
    """Allocate whole semantic components; deterministic and status-balanced.

    Larger components first; choose the split minimizing squared deviations from
    target case counts by status. This balances coverage, not statistical power.
    """
    active = [g for g in groups if g["eligible_case_count"]]
    totals = {status: sum(g["status_counts"].get(status, 0) for g in active) for status in decisions["statuses"]}
    used = {split: collections.Counter() for split in decisions["split_fractions"]}
    for group in active:
        if group.get("reserved_split"):
            group["split"] = group["reserved_split"]
            used[group["split"]].update(group["status_counts"])
    ordered = sorted((g for g in active if not g.get("reserved_split")), key=lambda g: (-g["eligible_case_count"], digest([decisions["seed"], g["split_group_id"]])))
    for group in ordered:
        def cost(candidate):
            return sum(((used[split][status] + (group["status_counts"].get(status, 0) if split == candidate else 0) - fraction * totals[status]) / max(1, totals[status])) ** 2 for split, fraction in decisions["split_fractions"].items() for status in totals)
        group["split"] = min(decisions["split_fractions"], key=lambda s: (cost(s), digest([decisions["seed"], group["split_group_id"], s])))
        used[group["split"]].update(group["status_counts"])
    for group in groups:
        group.setdefault("split", "deferred")


def clean_profile(row, case):
    """A proposed demographic comparison must hold all other attributes fixed."""
    attrs = {k: v for k, v in row["attributes"].items() if k not in ("gender", "age")}
    intrinsic = {k: v for k, v in case["intrinsic_attributes"].items() if k not in ("gender", "age")}
    return row["scope_eligible"] and row.get("reporter") is None and attrs == intrinsic


def finish(root, out, inputs, decisions, decisions_path, cases, inventory, sources, duplicate_rows, split_groups, parallel, blocks, old_matrix, specs, public_to_source, unresolved):
    by_source = collections.defaultdict(list)
    by_case = collections.defaultdict(list)
    keys = {}
    for row in inventory:
        assert row["record_key"] not in keys
        keys[row["record_key"]] = row
        by_source[row["source"]].append(row)
        by_case[row["source_case_id"]].append(row)
    source_meta = {r["source"]: r for r in sources}
    source_inventory = []
    for source, rows in sorted(by_source.items()):
        source_inventory.append({**source_meta.get(source, {}), "source": source,
            "records_by_language": dict(collections.Counter(r["language"] for r in rows)),
            "distinct_item_ids": len({r["item_id"] for r in rows}),
            "source_case_count": len({r["source_case_id"] for r in rows}),
            "unique_texts_by_language": {lang: len({r["text_sha256"] for r in rows if r["language"] == lang}) for lang in decisions["languages"]},
            "scope_eligible_bilingual_items": len({r["item_id"] for r in rows if r["language"] == "uk" and r["scope_eligible"] and r["bilingual_complete"]}),
            "historically_accepted_by_language": {lang: sum(r.get("historical_acceptance", False) for r in rows if r["language"] == lang) for lang in decisions["languages"]},
            "review_issue_counts": dict(collections.Counter(reason for r in rows for reason in r.get("review_issues", []))),
            "reuse_status_counts": dict(collections.Counter(r["reuse_status"] for r in rows)),
            "excluded_record_reasons": dict(collections.Counter(reason for r in rows for reason in r["exclusion_reasons"])),
            "note": source_meta.get(source, {}).get("note", "Translations, formulations and repeated copies do not count as independent source cases.")})

    coverage_rows, gap_rows, comparison_rows = [], [], []
    public_rows = sorted((c for c in cases.values() if c["public_case_id"]), key=lambda c: c["public_case_id"])
    case_table = []
    for case in public_rows:
        cid = case["source_case_id"]
        rows = by_case[cid]
        desired = expected_profiles(case)
        case["required_profiles"] = [dict(gender=g, age=a) for g, a in desired]
        case["gender_baseline_applicable"] = "gender" not in case["intrinsic_attributes"]
        case["eligibility_note"] = "Intrinsic attributes stay fixed; excluded profiles are not missing-data gaps."
        case["demographic_information_label_uk"] = "не зазначено в тексті"
        source_profile_maps = {}
        for source in ("release_triplets", "archive_v1", "archive_v2", "release_qa", "release_intersectional_qa"):
            all_rows = [r for r in rows if r["source"] == source and r["language"] == "uk"]
            available = [r for r in all_rows if r["bilingual_complete"] and clean_profile(r, case)]
            profiles = collections.defaultdict(list)
            for r in available:
                profiles[r["gender"], r["age"]].append(r)
            source_profile_maps[source] = profiles
            coverage_rows.append({"public_case_id": case["public_case_id"], "source_case_id": cid, "source": source,
                "scope_status": case["scope_status"], "observed_coverage": coverage((r["gender"], r["age"]) for r in all_rows),
                "eligible_clean_coverage": contrast_coverage(profiles), "bilingual_clean_items": len(available),
                "observed_profiles": [dict(gender=g, age=a) for g, a in sorted({(r["gender"], r["age"]) for r in all_rows})],
                "available_profiles": [dict(gender=g, age=a) for g, a in sorted(profiles)],
                "desired_profiles": [dict(gender=g, age=a) for g, a in desired],
                "missing_profiles": [dict(gender=g, age=a) for g, a in desired if (g, a) not in profiles] if case["scope_status"] == "eligible" else [],
                "profile_coverage_complete": set(desired).issubset(profiles) if case["scope_status"] == "eligible" else None,
                "matching_note": "Profile availability alone does not establish matched situations or semantic equivalence."})

        rel_profiles = source_profile_maps["release_triplets"]
        qa_cases = [r for r in rows if r["source"] == "release_qa" and r["language"] == "uk"]
        case["has_released_qa"] = bool(qa_cases)
        case["released_triplet_coverage"] = contrast_coverage(rel_profiles)
        case["released_observed_explicit_axes"] = coverage(rel_profiles)
        case["released_triplet_profile_complete"] = set(desired).issubset(rel_profiles) if case["scope_status"] == "eligible" else None
        case["released_matched_full_panel_ids"] = []
        for block in blocks:
            if block["case_id"] != case["public_case_id"]:
                continue
            member_keys = [f"release_triplets:uk:{m['id']}" for m in block["members"]]
            available = [keys[k] for k in member_keys if k in keys and keys[k]["scope_eligible"]]
            profile_set = {(r["gender"], r["age"]) for r in available}
            complete = set(desired).issubset(profile_set) and all(clean_profile(r, case) for r in available)
            if complete:
                case["released_matched_full_panel_ids"].append(block["block_id"])
            comparison_rows.append({"comparison_id": "triplet:" + block["block_id"], "axis": "demographic",
                "source_case_ids": [cid], "public_case_ids": [case["public_case_id"]],
                "record_keys": [k for r in available for k in (r["record_key"], r["record_key"].replace(":uk:", ":en:"))],
                "matching_basis": "published_comparison_block", "same_situation": "source_metadata_claim_not_revalidated",
                "full_required_profile_panel": complete, "status": "available_model_reviewed" if available else "deferred",
                "scope": "within_case", "split": case["split"], "split_group_id": case["split_group_id"]})

        if case["scope_status"] == "eligible":
            for g, a in desired:
                released = rel_profiles.get((g, a), [])
                archived = [r for source in ("archive_v2", "archive_v1") for r in source_profile_maps[source].get((g, a), [])]
                gap_rows.append({"public_case_id": case["public_case_id"], "source_case_id": cid,
                    "target_group": case["target_group"], "claim_family_id": case["claim_family_id"], "gender": g, "age": a,
                    "split": case["split"], "split_group_id": case["split_group_id"], "format": "triplet",
                    "action": "reuse_release_verify_matching" if released else "review_archive_before_generation" if archived else "generate_missing_profile",
                    "released_item_ids": sorted({r["item_id"] for r in released}),
                    "archive_record_keys": [r["record_key"] for r in archived],
                    "archive_accepted_pair_candidates": sum(r.get("historical_acceptance", False) and parallel[r["source"], r["item_id"]]["en"].get("historical_acceptance", False) for r in archived),
                    "controlled_panel_already_complete": bool(case["released_matched_full_panel_ids"]),
                    "final_text_severity_status": "review_required"})
            if not qa_cases:
                gap_rows.append({"public_case_id": case["public_case_id"], "source_case_id": cid,
                    "target_group": case["target_group"], "claim_family_id": case["claim_family_id"],
                    "format": "qa", "action": "author_new_situation_and_evidence_before_expansion",
                    "split": case["split"], "split_group_id": case["split_group_id"],
                    "note": "Existing source association, not yet a validated BBQ-style QA situation; six conditions per planned base scenario."})

        case_table.append({k: case[k] for k in ("public_case_id", "source_case_id", "target_group", "source_claim_uk", "original_category", "claim_family_id", "family_relation", "family_note", "scope_status", "scope_reason", "intrinsic_attributes", "gender_baseline_applicable", "harm_mechanisms_uk", "source_severity_candidate", "severity_rationale", "severity_review_status", "final_text_severity_status", "released_triplet_coverage", "released_observed_explicit_axes", "released_triplet_profile_complete", "released_matched_full_panel_ids", "required_profiles", "has_released_qa", "split", "split_group_id")})

    # UK/EN identity is necessary, but not proof of semantic translation quality.
    for (source, rid), pair in sorted(parallel.items()):
        if set(pair) != {"uk", "en"}:
            continue
        a, b = pair["uk"], pair["en"]
        assert a["source_case_id"] == b["source_case_id"]
        if (a["gender"], a["age"]) != (b["gender"], b["age"]):
            unresolved.append({"type": "bilingual_metadata_mismatch", "record_keys": [a["record_key"], b["record_key"]]})
        comparison_rows.append({"comparison_id": f"language:{source}:{rid}", "axis": "language",
            "source_case_ids": [a["source_case_id"]], "record_keys": [a["record_key"], b["record_key"]],
            "matching_basis": "shared_source_item_id", "status": "available_model_reviewed" if source.startswith("release") and a["scope_eligible"] and b["scope_eligible"] else "archive_review_required" if a["scope_eligible"] and b["scope_eligible"] else "deferred",
            "semantic_validation": "existing_model_review_only_or_pending; no new bilingual content review performed",
            "split": a["split"], "split_group_id": a["split_group_id"]})

    qa_panels = collections.defaultdict(list)
    for row in by_source["release_qa"] + by_source["release_intersectional_qa"]:
        if row["language"] == "uk":
            qa_panels[row["source_case_id"], row["gender"], row["age"]].append(row)
    expected_conditions = set(itertools.product(("ambiguous", "stereotype_aligned", "stereotype_conflicting"), ("negative", "positive")))
    for (cid, g, a), rows in sorted(qa_panels.items()):
        observed = {(r["condition"], r["polarity"]) for r in rows}
        assert len(rows) == 6 and observed == expected_conditions, (cid, g, a)
        case = cases[cid]
        comparison_rows.append({"comparison_id": f"qa-panel:{case['public_case_id']}:{g}:{a}", "axis": "evidence_and_polarity",
            "source_case_ids": [cid], "gender": g, "age": a, "situation_id": rows[0]["situation_id"],
            "record_keys": [k for r in rows for k in (r["record_key"], r["record_key"].replace(":uk:", ":en:"))],
            "matching_basis": "same_scenario_six_conditions", "status": "available_model_reviewed" if all(r["scope_eligible"] for r in rows) else "deferred",
            "split": case["split"], "split_group_id": case["split_group_id"]})
    base_rows = {r["item_id"]: r for r in by_source["release_qa"] if r["language"] == "uk"}
    for row in by_source["release_intersectional_qa"]:
        if row["language"] != "uk":
            continue
        base = base_rows[row["base_id"]]
        assert base["source_case_id"] == row["source_case_id"] and base["situation_id"] == row["situation_id"]
        assert base["condition"] == row["condition"] and base["polarity"] == row["polarity"]
        assert base["answer_index"] == row["answer_index"] and base["answer_roles"] == row["answer_roles"]
        assert base["payload"]["question"] == row["payload"]["question"] and base["payload"]["choices"] == row["payload"]["choices"]
        comparison_rows.append({"comparison_id": "qa-profile:" + row["item_id"], "axis": "base_to_demographic",
            "source_case_ids": [row["source_case_id"]], "situation_id": row["situation_id"],
            "record_keys": [base["record_key"], row["record_key"], base["record_key"].replace(":uk:", ":en:"), row["record_key"].replace(":uk:", ":en:")],
            "matching_basis": "published_base_id_same_question_choices_answer_roles", "status": "available_model_reviewed" if row["scope_eligible"] and base["scope_eligible"] else "deferred",
            "split": row["split"], "split_group_id": row["split_group_id"]})

    cross_actor = []
    for family in decisions["claim_families"]:
        member_cases = [cases[public_to_source[f"WB-C{n:03d}"]] for n in family["cases"]]
        members = [c for c in member_cases if c["scope_status"] == "eligible"]
        if {c["target_group"] for c in members} != {"idp", "veteran"}:
            continue
        exact = family["relation"] == "exact_source_template"
        if exact:
            assert len({c["source_claim_uk"] for c in members}) == 1
        cross_actor.append({"claim_family_id": family["id"], "source_case_ids": [c["source_case_id"] for c in members],
            "public_case_ids": [c["public_case_id"] for c in members], "relation": family["relation"],
            "status": "source_claim_match_final_situations_require_review" if exact else "related_only_not_an_exact_actor_counterfactual",
            "matching_note": family["note"], "directions": ["idp_to_veteran", "veteran_to_idp"],
            "strict_transfer_eligible_now": False,
            "required_work": "Match final claim strength, situation, non-status attributes and scoring protocol before registering a strict comparison.",
            "split_group_id": members[0]["split_group_id"], "split": members[0]["split"]})

    enriched_matrix = []
    for original in old_matrix:
        case = cases[original["case_id"]]
        attrs = {"gender": original["gender_track"], "age": original["age_track"]}
        enriched_matrix.append({**original, "public_case_id": case["public_case_id"], "claim_family_id": case["claim_family_id"],
            "claim_match_type": case["family_relation"], "explicit_gender": normalize_gender(attrs["gender"]), "explicit_age": normalize_age(attrs["age"]),
            "demographic_baseline_label_uk": "не зазначено в тексті", "scope_status": case["scope_status"],
            "source_severity_candidate": case["source_severity_candidate"], "severity_review_status": case["severity_review_status"],
            "final_text_severity_status": "review_required", "split": case["split"], "split_group_id": case["split_group_id"],
            "coverage_design": contrast_coverage((normalize_gender(r["gender_track"]), normalize_age(r["age_track"])) for r in old_matrix if r["case_id"] == original["case_id"]),
            "released_triplet_coverage": case.get("released_triplet_coverage", "outside_current_release"),
            "legacy_row_use": "source_design_only; actor scope and final text must be reconciled before reuse"})

    protocols = {
        "version": decisions["version"], "status": "frozen_design_not_executed", "languages": decisions["languages"],
        "primary_scope": "Eligible globally reusable veteran and IDP material; preserve intrinsic attributes and recorded profile exclusions.",
        "prospective_split": {"fractions": decisions["split_fractions"], "seed": decisions["seed"], "algorithm": "semantic/provenance/duplicate connected components; apply declared transfer-pilot reservation, then largest first, deterministic status-balanced greedy allocation", "reserved_train_claim_families": decisions.get("reserve_train_claim_families", []), "reservation_reason": decisions.get("reservation_reason"), "note": decisions["split_note"], "assignment_file": "split_groups.jsonl", "future_items": "Inherit the case's split_group_id and reserved_component_split. New claims need a versioned semantic-leakage audit before assignment; never reallocate this frozen version."},
        "baseline_benchmark": "Report all prespecified eligible released cases descriptively; this is separate from claims of unseen-case generalization after training.",
        "unseen_claim_generalization": "Train on train components only. Calibrate prompts, thresholds and selection on development components. Evaluate untouched evaluation components; keep all actors, languages, formats, paraphrases and related claims together. Existing exposure must be audited before calling any checkpoint's evaluation unseen.",
        "within_claim_transfer": "Separate experiment using train components only: intervene on one actor/profile/language and measure the paired non-target before/after. Deliberate shared-claim exposure is the design; never report this as held-out-claim generalization. Development components calibrate protocol; evaluation components provide a separate unseen-claim test.",
        "gender_age": "Within each status compare common case/situation panels, holding non-demographic attributes and evidence/polarity fixed. Intrinsically gendered claims cannot acquire a neutral or opposite-gender baseline by relabeling. Use the five shared explicit age/gender profiles for cross-status summaries, reporting the IDP older-women cell separately.",
        "actor_transfer": "Primary candidates require exact or reviewed equivalent claims, matched situations and matched profiles. Related topics are exploratory and retain separate labels. Run both intervention directions; do not assume transfer is absent.",
        "topic_transfer": "Use registered source/target claim families and preserve topics as separate labels. Related-family links here are not calibrated topic distances. Near/far labels remain unvalidated and must be frozen before inspecting model effects.",
        "language_transfer": "Use linked UK/EN records and comparable per-language scores; preserve actor, situation, answer roles, settings and before/after pairing. Do not compare raw cross-language token likelihood magnitudes as a bias effect.",
        "uncertainty": "Retain all variants and translations within resampled source-case/semantic-family clusters; use common resamples for paired comparisons. Report rows, situations, cases and split groups separately. Recompute extrema/ratios on each draw; zero or unstable denominators remain explicit.",
        "severity": decisions["severity_rubric"],
        "metric_inputs": {"native_and_group_scores": "Existing family-specific outcomes; retain denominators and eligibility.", "NNR_LMS_ICAT": "Candidate likelihood scoring; chat answers alone are insufficient.", "before_after_transfer_RI": "Paired checkpoint/intervention outputs; no unbiased reference required.", "over_refusal_and_retained_helpfulness": "Dedicated benign requests and scoring criteria still need authoring; no such prompts were generated here.", "training_imbalance": "Recorded training exposure or controlled training conditions; evaluation row counts are not training counts."},
        "generation_contract": ["Reuse preferred release text before considering archives.", "Archive historical acceptance is not current semantic clearance.", "Matched variants preserve claim, strength, situation, counter strategy and unrelated event, with natural grammatical changes.", "Required profile cells absent because of policy or intrinsic attributes are not generation gaps.", "Rare constructed combinations are experimental stimuli, not evidence of stereotype prevalence.", "No local-theme generation in this version.", "No model calls or annotator assignments in steps 1–3."],
        "unresolved_content_checks": ["Final-text severity", "Cross-actor situation equivalence", "Archive eligibility after released corrections", "Bilingual semantic alignment beyond existing review", "Near/far topic-distance labels", "Historical training exposure"]
    }
    # Count actual situations separately; triplet variants have no established independent situation IDs.
    summary = {
        "version": decisions["version"], "steps": {"1_inventory": "complete_for_declared_source_streams", "2_metadata": "prepared_with_provisional_semantic_and_severity_labels", "3_design_and_splits": "frozen_prospective_design"},
        "source_streams": len(source_inventory), "inventory_records": len(inventory), "unique_texts_across_streams_and_languages": len({(r["language"], r["text_sha256"]) for r in inventory}),
        "duplicate_text_groups": len(duplicate_rows), "current_release_case_families": len(public_rows),
        "eligible_case_families": sum(c["scope_status"] == "eligible" for c in public_rows),
        "deferred_local_case_families": [c["public_case_id"] for c in public_rows if c["scope_status"] != "eligible"],
        "legacy_or_superseded_cases_preserved": len(cases) - len(public_rows),
        "qa_case_families": sum(c["has_released_qa"] for c in public_rows),
        "published_cases_without_qa": sum(not c["has_released_qa"] for c in public_rows),
        "eligible_cases_without_qa": sum(not c["has_released_qa"] and c["scope_status"] == "eligible" for c in public_rows),
        "triplet_gap_actions": dict(collections.Counter(r["action"] for r in gap_rows if r["format"] == "triplet")),
        "cross_actor_families": len(cross_actor), "exact_source_cross_actor_families": [r["claim_family_id"] for r in cross_actor if r["relation"] == "exact_source_template"],
        "split_case_counts": dict(collections.Counter(c["split"] for c in public_rows)),
        "split_status_counts": {split: dict(collections.Counter(c["target_group"] for c in public_rows if c["split"] == split)) for split in decisions["split_fractions"]},
        "eligible_split_components": sum(g["eligible_case_count"] > 0 for g in split_groups),
        "scope_exclusions_are_preserved": True, "generated_new_texts": 0, "model_calls": 0, "annotator_assignments_added": 0,
        "limitations": protocols["unresolved_content_checks"],
        "inventory_boundary": "Latest item versions and all-version counts in v1/v2 archives; released triplets/base QA/intersectional QA; deduplicated LENS prompts; legacy source matrix and superseded claim lineage. Intermediate HF mirror copies and raw transport retries are not separate datasets. Source files are not rewritten."
    }
    validate(cases, inventory, split_groups, comparison_rows, parallel, decisions)
    changed = [p for p, h in inputs.hashes.items() if sha(root / p) != h]
    assert not changed, changed
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    write_json(out / "protocol.json", protocols)
    write_json(out / "decisions.json", decisions)
    write_lines(out / "inventory.jsonl", inventory)
    write_csv(out / "source_inventory.csv", source_inventory)
    write_lines(out / "source_inventory.jsonl", source_inventory)
    write_lines(out / "duplicate_texts.jsonl", duplicate_rows)
    write_lines(out / "cases.jsonl", list(cases.values()))
    write_csv(out / "case_matrix.csv", case_table)
    write_csv(out / "legacy_matrix_enriched.csv", enriched_matrix)
    write_lines(out / "coverage.jsonl", coverage_rows)
    write_csv(out / "coverage.csv", coverage_rows)
    write_csv(out / "generation_gaps.csv", gap_rows)
    write_lines(out / "generation_gaps.jsonl", gap_rows)
    write_lines(out / "comparisons.jsonl", comparison_rows)
    write_lines(out / "cross_actor_candidates.jsonl", cross_actor)
    write_csv(out / "cross_actor_candidates.csv", cross_actor)
    write_lines(out / "split_groups.jsonl", sorted(split_groups, key=lambda g: g["split_group_id"]))
    write_csv(out / "case_splits.csv", [{k: c[k] for k in ("public_case_id", "source_case_id", "claim_family_id", "target_group", "split_group_id", "split", "reserved_component_split", "scope_status")} for c in cases.values()])
    write_lines(out / "item_splits.jsonl", [{k: r[k] for k in ("record_key", "source_case_id", "claim_family_id", "split_group_id", "split", "reserved_component_split", "scope_eligible", "reuse_status")} for r in inventory])
    write_json(out / "unresolved_checks.json", {"automated_findings": unresolved, "required_content_checks": protocols["unresolved_content_checks"]})
    write_csv(out / "source_hashes.csv", [{"path": p, "sha256": h} for p, h in sorted(inputs.hashes.items())])
    validation = {"passed": True, "checks": ["127 current cases mapped through recorded IDs", "all source claims assigned provisional rubric labels", "UK/EN and all formats inherit one component", "known related claims, split parents and exact text duplicates remain together", "local/occupation/older-women-veteran exclusions cannot enter active splits", "all QA profiles have six conditions with preserved semantic answer roles", "comparison references exist", "source files unchanged"], "source_files_verified": len(inputs.hashes), "human_validation": False}
    write_json(out / "validation.json", validation)
    render_overview(out, summary, source_inventory, case_table, cross_actor)
    manifest = {"version": decisions["version"], "builder_sha256": sha(Path(__file__)), "decisions_sha256": sha(decisions_path),
                "source_index_sha256": sha(out / "source_hashes.csv"), "files": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"},
                "frozen": True, "scope": "offline_preparation_steps_1_2_3_only", "regeneration_command": "python outputs/warbias_preparation_20260925/build.py --output /tmp/warbias-preparation-rebuild"}
    write_json(out / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate(cases, inventory, groups, comparisons, parallel, decisions):
    assert len({r["record_key"] for r in inventory}) == len(inventory)
    available = {r["record_key"] for r in inventory}
    for comp in comparisons:
        assert set(comp["record_keys"]) <= available
    for group in groups:
        assert len({cases[c]["reserved_component_split"] for c in group["source_case_ids"]}) == 1
    hash_splits = collections.defaultdict(set)
    for row in inventory:
        assert row["split_group_id"] == cases[row["source_case_id"]]["split_group_id"]
        assert row["source_severity_candidate"] == cases[row["source_case_id"]]["source_severity_candidate"]
        assert row["final_text_severity"] is None
        if not row["scope_eligible"]:
            assert row["split"] == "deferred"
        if row["scope_eligible"]:
            assert row["split"] in decisions["split_fractions"]
            assert not (row["target_group"] == "veteran" and row["gender"] == "women" and row["age"] == "older")
        hash_splits[row["language"], row["leakage_sha256"]].add(row["reserved_component_split"])
    assert all(len(v) == 1 for v in hash_splits.values())
    for pair in parallel.values():
        assert len({r["reserved_component_split"] for r in pair.values()}) == 1
    for group in decisions["claim_families"]:
        member_cases = [c for c in cases.values() if c["public_case_id"] in {f"WB-C{n:03d}" for n in group["cases"]}]
        assert len({c["reserved_component_split"] for c in member_cases}) == 1


def render_overview(out, summary, sources, cases, actor):
    def table(rows, fields):
        return '<table><thead><tr>' + ''.join('<th>' + html.escape(k.replace('_', ' ')) + '</th>' for k in fields) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(r.get(k, ''))) + '</td>' for k in fields) + '</tr>' for r in rows) + '</tbody></table>'
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>WarBias preparation — 25 September 2026</title><style>
    body{font:16px/1.5 system-ui;margin:40px auto;max-width:1250px;padding:0 24px;color:#203040;background:#fafbfc}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:13px;background:white;margin:18px 0}th,td{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid #dce3e9}th{position:sticky;top:0;background:#e9eff5}p{max-width:950px}a{color:#1558a8}.notice{background:#fff3d7;padding:16px;border-left:4px solid #ce991c}input{padding:10px;width:95%;font:inherit}code{background:#edf0f4;padding:2px 5px}</style>
    <h1>WarBias: inventory, metadata and comparison design</h1><p>Preparation frozen 25 September 2026. No new stimulus generation, model experiments or annotator assignments.</p>'''
    document += '<p class="notice">Source severity and semantic families are assistant-reviewed provisional metadata. Final-text severity and cross-actor situation equivalence still require content review. The split is prospective; historical model exposure has not been audited.</p>'
    document += '<p><b>' + str(summary['eligible_case_families']) + '</b> eligible current case families; <b>' + str(len(summary['deferred_local_case_families'])) + '</b> local families deferred. <b>' + str(summary['eligible_cases_without_qa']) + '</b> eligible families have no released QA counterpart.</p>'
    document += '<p>Files: <a href="case_matrix.csv">case matrix</a> · <a href="generation_gaps.csv">generation/reuse gaps</a> · <a href="cross_actor_candidates.csv">cross-actor candidates</a> · <a href="case_splits.csv">case splits</a> · <a href="protocol.json">comparison protocol</a> · <a href="validation.json">validation</a> · <a href="manifest.json">frozen manifest</a></p>'
    document += '<h2>Source inventory</h2>' + table(sources, ['source', 'records_by_language', 'source_case_count', 'scope_eligible_bilingual_items'])
    document += '<p>Archive versions and translations overlap. Counts in this table must not be added as independent cases. Archive text requires review against the later release.</p>'
    document += '<h2>Prospective split</h2><p>' + html.escape(str(summary['split_status_counts'])) + '</p><p>Related claims, all profiles, translations and formats stay in the same component. Within-claim transfer is a separate protocol, not unseen-claim generalization.</p>'
    document += '<h2>Claims and coverage</h2><input id="filter" aria-label="Filter cases" placeholder="Filter by claim, case ID, scope or split"><div id="cases">' + table(cases, ['public_case_id', 'target_group', 'source_claim_uk', 'claim_family_id', 'released_triplet_coverage', 'has_released_qa', 'source_severity_candidate', 'scope_status', 'split']) + '</div>'
    document += '<script>document.getElementById("filter").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll("#cases tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))});</script></html>'
    (out / "overview.html").write_text(document)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE.parents[1])
    parser.add_argument("--output", type=Path, default=HERE / "artifacts")
    parser.add_argument("--decisions", type=Path, default=HERE / "decisions.json")
    args = parser.parse_args()
    build(args.root.resolve(), args.output.resolve(), args.decisions.resolve())
