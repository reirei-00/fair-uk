"""Offline tests for metadata distinctions, lineage and leakage protection."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("warbias_preparation", Path(__file__).with_name("build.py"))
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class DesignTests(unittest.TestCase):
    def test_separate_axes_do_not_become_joint_coverage(self):
        separate = [(build.NS, build.NS), ("men", build.NS), ("women", build.NS), (build.NS, "older")]
        self.assertEqual(build.coverage(separate), "both_separately")
        self.assertEqual(build.coverage(separate + [("men", "older")]), "joint_gender_age")
        self.assertEqual(build.coverage([]), "none")

    def test_intrinsic_gender_does_not_create_impossible_baselines(self):
        profiles = build.expected_profiles({"target_group": "veteran", "intrinsic_attributes": {"gender": "women"}})
        self.assertEqual(set(profiles), {("women", build.NS), ("women", "young"), ("women", "middle_aged")})
        self.assertNotIn(("women", "older"), profiles)
        self.assertEqual(len(build.expected_profiles({"target_group": "idp", "intrinsic_attributes": {}})), 12)
        self.assertEqual(build.contrast_coverage(profiles), "age_only")
        self.assertEqual(build.contrast_coverage([("women", build.NS)]), "baseline")

    def test_unknown_metadata_cannot_be_silently_normalized(self):
        with self.assertRaises(ValueError):
            build.normalize_gender("missing_metadata")
        with self.assertRaises(ValueError):
            build.normalize_age("unknown_code")
        self.assertEqual(build.normalize_gender("generic_plural"), build.NS)

    def test_review_acceptance_is_bound_to_exact_text(self):
        row = dict(zip(build.FIELDS, ["Example S.", "Example C.", "Example U."]))
        row["language"] = "uk"
        review = {"decision": "accepted", "text_hash": build.digest({k: row[k] for k in build.FIELDS}), "criteria": {k: "pass" for k in (*build.CRITERIA, "TRANSLATION_FIDELITY")}}
        self.assertTrue(build.archive_acceptance(row, review))
        self.assertFalse(build.archive_acceptance({**row, "stereotype": "Changed claim."}, review))
        review["criteria"]["TRANSLATION_FIDELITY"] = "fail"
        self.assertFalse(build.archive_acceptance(row, review))

    def test_archive_precedence_uses_attempt_not_filename_or_mtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "rows").mkdir()
            (root / "rows/z.jsonl").write_text(json.dumps({"id": "x", "attempt": 1, "value": "old"}) + "\n")
            (root / "rows/a.jsonl").write_text(json.dumps({"id": "x", "attempt": 2, "value": "new"}) + "\n")
            rows, stats = build.Inputs(root).latest("rows")
            self.assertEqual(rows["x"][0]["value"], "new")
            self.assertEqual(stats["x"]["records"], 2)

    def test_archive_same_rank_conflict_is_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "rows").mkdir()
            (root / "rows/a.jsonl").write_text('\n'.join(json.dumps({"id": "x", "attempt": 2, "value": v}) for v in ["one", "two"]) + '\n')
            _, stats = build.Inputs(root).latest("rows")
            self.assertTrue(stats["x"]["same_rank_conflict"])

    def test_comparison_rejects_nuisance_attribute_changes(self):
        case = {"intrinsic_attributes": {}}
        row = {"scope_eligible": True, "reporter": None, "attributes": {"gender": "women", "age": "young"}}
        self.assertTrue(build.clean_profile(row, case))
        self.assertFalse(build.clean_profile({**row, "attributes": {**row["attributes"], "registration": "unregistered"}}, case))
        self.assertFalse(build.clean_profile({**row, "reporter": "landlord"}, case))


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parent / "artifacts"
        cls.rows = [json.loads(s) for s in (cls.root / "inventory.jsonl").read_text().splitlines()]
        cls.cases = [json.loads(s) for s in (cls.root / "cases.jsonl").read_text().splitlines()]
        cls.decisions = json.loads((cls.root / "decisions.json").read_text())

    def test_shared_claim_and_all_languages_remain_in_one_split(self):
        selected = [r for r in self.rows if r["public_case_id"] in ("WB-C021", "WB-C069")]
        self.assertEqual({r["target_group"] for r in selected}, {"idp", "veteran"})
        self.assertEqual({r["language"] for r in selected}, {"uk", "en"})
        self.assertEqual(len({r["split_group_id"] for r in selected}), 1)
        self.assertEqual(len({r["reserved_component_split"] for r in selected}), 1)
        self.assertEqual({r["reserved_component_split"] for r in selected}, {"train"})

    def test_superseded_compound_and_children_cannot_leak(self):
        cases = {r["source_case_id"]: r for r in self.cases}
        parent = cases["ssuk_vet_benefits_deservingness_mercenary_case_002"]
        children = [cases[c] for c in ("wb3_vet_benefits_dependency", "wb3_vet_benefits_work_motivation")]
        self.assertEqual(parent["scope_status"], "superseded_compound")
        self.assertEqual({c["split_group_id"] for c in [parent, *children]}, {parent["split_group_id"]})

    def test_explicit_exclusions_stay_out_of_active_partitions(self):
        deferred = set(self.decisions["scope"]["deferred_public_cases"])
        for row in self.rows:
            excluded = row["public_case_id"] in deferred or (row["target_group"] == "veteran" and row["gender"] == "women" and row["age"] == "older") or any(k in row["attributes"] for k in ("occupation", "actor_occupation", "prior_occupation"))
            if excluded:
                self.assertFalse(row["scope_eligible"])
                self.assertEqual(row["split"], "deferred")

    def test_final_text_severity_is_not_falsely_validated(self):
        for row in self.rows:
            self.assertIsNone(row["final_text_severity"])
            self.assertEqual(row["final_text_severity_status"], "review_required")
        current = [c for c in self.cases if c["public_case_id"]]
        self.assertEqual(len(current), 127)
        self.assertTrue(all(c["source_severity_candidate"] in (1, 2, 3) for c in current))

    def test_complete_profile_presence_is_not_claimed_as_semantic_matching(self):
        gaps = [json.loads(s) for s in (self.root / "generation_gaps.jsonl").read_text().splitlines()]
        self.assertTrue(any(g.get("released_item_ids") and not g["controlled_panel_already_complete"] for g in gaps if g["format"] == "triplet"))
        actors = [json.loads(s) for s in (self.root / "cross_actor_candidates.jsonl").read_text().splitlines()]
        self.assertTrue(any(a["relation"] == "exact_source_template" for a in actors))
        self.assertFalse(any(a["strict_transfer_eligible_now"] for a in actors))

    def test_artifact_checksums(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        for name, h in manifest["files"].items():
            self.assertEqual(build.sha(self.root / name), h, name)


if __name__ == "__main__":
    unittest.main()
