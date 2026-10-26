"""Primary PMI intervention effects use matched saved original and neutral runs."""

import json
import shutil
from dataclasses import replace

import pytest
from warbias_support import neutral_records, phase_runs, write_jsonl

from lm_eval.fair_uk.contracts import canonical_sha256
from lm_eval.fair_uk.data import dataset_spec, digest, load
from lm_eval.fair_uk.warbias.intervention import (
    _language_transfer,
    compare_items,
    compare_runs,
    items_sha256,
)
from lm_eval.fair_uk.warbias.rows import adapt_annotated
from lm_eval.fair_uk.warbias.schema import TripletItem


def item(identifier, actor, cluster, margin):
    return TripletItem(
        id=identifier,
        actor=actor,
        cluster=cluster,
        comparison_unit=identifier,
        gender="not_stated",
        age="not_stated",
        preference=float(margin > 0),
        tie=float(margin == 0),
        margin=margin,
        delta_stereotype=margin,
        delta_counter=0.0,
        delta_unrelated=-0.1,
        length_difference=2,
    )


def balanced_panels(components=6):
    before, after = [], []
    for actor in ("idp", "veteran"):
        for component in range(components):
            for profile in range(2):
                before_margin = 2.0 if actor == "idp" or profile == 0 else -2.0
                after_margin = 2.0 if actor == "veteran" or profile == 0 else -2.0
                before.append(
                    item(
                        f"{actor}-{component}-{profile}",
                        actor,
                        f"{actor}-{component}",
                        before_margin,
                    )
                )
                after.append(
                    item(
                        f"{actor}-{component}-{profile}",
                        actor,
                        f"{actor}-{component}",
                        after_margin,
                    )
                )
    return before, after


def compare(before, after, **kwargs):
    return compare_items(
        before, after, target_axis="actor", target_group="idp", **kwargs
    )


def calibration(before, tau=0.5):
    return {
        "value": tau,
        "calibration_id": "frozen-development-rule",
        "baseline_eligible_ids": [i.id for i in before if i.margin > tau],
        "baseline_items_sha256": items_sha256(before),
    }


def test_no_update_has_zero_paired_effect_and_no_invented_nnr_or_ri():
    before, _ = balanced_panels()
    report = compare(before, before, draws=30, minimum_target_reduction=0.05)
    for group in (report["all"], report["target"], *report["non_target"].values()):
        assert group["ss"]["after_minus_before"] == 0.0
        assert group["ss"]["ci95"] == [0.0, 0.0]
        assert group["margin"]["after_minus_before"] == 0.0
        assert group["distance_from_half"]["target_reduction"] == 0.0
    assert report["nnr"]["reason"] == "missing_frozen_neutrality_calibration"
    assert report["redistribution"]["value"] is None
    assert report["harm_change_disparity"]["harm_change_range"] == 0.0


def test_primary_reduction_signed_change_and_redistribution_have_distinct_signs():
    before, after = balanced_panels()
    report = compare(before, after, minimum_target_reduction=0.05, draws=50)
    target = report["target"]
    assert target["ss"]["after_minus_before"] == -0.5
    assert target["margin"]["after_minus_before"] == -2.0
    assert target["distance_from_half"]["target_reduction"] == 0.5
    assert target["distance_from_half"]["reduction_ci95"] == [0.5, 0.5]
    assert (
        report["non_target"]["veteran"]["distance_from_half"]["after_minus_before"]
        == 0.5
    )
    ri = report["redistribution"]
    assert ri["status"] == "available" and ri["value"] == 1.0
    assert ri["numerator"] == 0.5 and ri["non_target_group_count"] == 1
    reverse = compare(after, before, minimum_target_reduction=0.05, draws=50)
    assert reverse["target"]["ss"]["after_minus_before"] == 0.5
    assert reverse["target"]["distance_from_half"]["target_reduction"] == -0.5
    assert reverse["redistribution"]["status"] == "unavailable"


def test_reversing_stereotype_preference_is_not_reduction_toward_half():
    before, _ = balanced_panels()
    after = [
        replace(
            i,
            margin=-i.margin,
            delta_stereotype=-i.margin,
            preference=float(i.margin < 0),
        )
        for i in before
    ]
    report = compare(before, after, draws=20)
    assert report["target"]["ss"]["after_minus_before"] == -1.0
    assert report["target"]["distance_from_half"]["target_reduction"] == 0.0


def test_frozen_nnr_reports_cures_reversals_components_and_raw_transitions():
    before = [item(f"i{i}", "idp", "a" if i < 3 else "b", 2.0) for i in range(4)]
    before.append(item("v", "veteran", "v", -2.0))
    after = [
        replace(
            i, margin=m, delta_stereotype=m, preference=float(m > 0), tie=float(m == 0)
        )
        for i, m in zip(before, [0.0, 0.0, -2.0, -2.0, -2.0], strict=True)
    ]
    report = compare(before, after, tau=calibration(before), draws=40)["nnr"]
    assert report["status"] == "available"
    assert report["eligible_rows"] == 4 and report["eligible_components"] == 2
    assert report["cured_rows"] == report["reversed_rows"] == 2
    assert report["nnr"] == pytest.approx(-1 / 3)
    assert report["raw_item_nnr"] == 0.0
    assert report["all_state_transitions"]["S_to_N"] == 2
    assert report["all_state_transitions"]["S_to_C"] == 2
    assert report["all_state_transitions"]["C_to_C"] == 1


@pytest.mark.parametrize(
    "tau, reason",
    [
        (0.5, "frozen_baseline_cohort_required"),
        ({"value": 0.5}, "missing_baseline_items_fingerprint"),
    ],
)
def test_unqualified_nnr_remains_unavailable(tau, reason):
    before, after = balanced_panels()
    assert compare(before, after, tau=tau, draws=0)["nnr"]["reason"] == reason


@pytest.mark.parametrize("change", ["fingerprint", "cohort"])
def test_stale_frozen_nnr_evidence_is_rejected(change):
    before, after = balanced_panels()
    rule = calibration(before)
    if change == "fingerprint":
        rule["baseline_items_sha256"] = "0" * 64
    else:
        rule["baseline_eligible_ids"] = []
    with pytest.raises(ValueError, match="Frozen NNR"):
        compare(before, after, tau=rule, draws=0)


def test_support_practical_minimum_and_uncertainty_gate_ri():
    before, after = balanced_panels(2)
    report = compare(before, after, minimum_target_reduction=0.05, draws=20)
    assert report["target"]["supported"] is False
    assert report["harm_change_disparity"]["status"] == "unavailable"
    assert report["redistribution"]["reason"] == "insufficient_component_support"
    before, after = balanced_panels()
    assert (
        compare(before, after, draws=20)["redistribution"]["reason"]
        == "missing_practical_target_reduction"
    )
    assert (
        compare(before, after, minimum_target_reduction=0.5, draws=20)[
            "redistribution"
        ]["reason"]
        == "target_reduction_below_practical_minimum"
    )
    assert (
        compare(before, after, minimum_target_reduction=0.05, draws=0)[
            "redistribution"
        ]["reason"]
        == "target_reduction_not_supported_positive"
    )


@pytest.mark.parametrize("change", ["ids", "cluster", "metadata", "raw_margin"])
def test_pairing_rejects_cohort_source_or_nonprimary_scores(change):
    before, after = balanced_panels()
    if change == "ids":
        after = after[:-1]
    elif change == "cluster":
        after[0] = replace(after[0], cluster="other")
    elif change == "metadata":
        after[0] = replace(after[0], comparison_unit="different-case")
    else:
        after[0] = replace(after[0], delta_stereotype=None)
    with pytest.raises(ValueError, match="identical IDs|metadata differs|primary PMI"):
        compare(before, after, draws=0)


@pytest.fixture
def saved_panels(warbias_inputs, monkeypatch):
    files, inv, tmp = warbias_inputs
    full, _phases, original, neutral = phase_runs(
        files, inv, tmp, neutral_records(files, monkeypatch)
    )
    for stream in (original, neutral):
        for entries in stream.values():
            for bundle, directory in entries:
                path = directory / "run.json"
                run = json.loads(path.read_text())
                rows = load(run["task"], dataset_bundle=bundle)
                run["evaluation_asset_identity"] = {
                    "tokenizer_sha256": "f" * 64,
                    "input_sha256": canonical_sha256(
                        [{"id": r["id"], "sentences": r["sentences"]} for r in rows]
                    ),
                    "rows": len(rows),
                    "protocol": run["protocol"],
                    "input_mode": "likelihood",
                }
                run["model_args"] = (
                    "pretrained=org/model,revision=" + "a" * 40 + ",max_model_len=4096"
                )
                run["batch_size"] = "1"
                path.write_text(json.dumps(run))
    after_original, after_neutral = {}, {}
    for source, target in ((original, after_original), (neutral, after_neutral)):
        for language, entries in source.items():
            target[language] = []
            for bundle, directory in entries:
                copied = tmp / "after" / directory.relative_to(tmp / "runs")
                shutil.copytree(directory, copied)
                run_path = copied / "run.json"
                run = json.loads(run_path.read_text())
                run["model_identity"] = {
                    "repository": "org/after",
                    "revision": "b" * 40,
                }
                run["model_args"] = (
                    "pretrained=org/after,revision=" + "b" * 40 + ",max_model_len=4096"
                )
                run_path.write_text(json.dumps(run))
                target[language].append((bundle, copied))
    return full, original, after_original, neutral, after_neutral, tmp


def run_comparison(saved_panels, output="paired", **kwargs):
    full, before, after, before_neutral, after_neutral, tmp = saved_panels
    return compare_runs(
        full,
        before,
        after,
        before_neutral,
        after_neutral,
        tmp / output,
        target_axis="actor",
        target_group="idp",
        draws=20,
        minimum_support=1,
        **kwargs,
    )


def test_saved_four_run_groups_bind_checkpoint_source_and_primary_outputs(saved_panels):
    output = run_comparison(saved_panels)
    report = json.loads((output / "intervention.json").read_text())
    assert report["language_transfer"]["reason"] == "missing_training_language"
    for language in ("uk", "en"):
        assert report["languages"][language]["all"]["ss"]["after_minus_before"] == 0.0
        assert (
            report["languages"][language]["all"]["margin"]["after_minus_before"] == 0.0
        )
        for state in ("before", "after"):
            for stream in ("original", "neutral"):
                assert report["sources"][state][language][stream][0][
                    "predictions_sha256"
                ]
                assert report["sources"][state][language][stream][0][
                    "run_manifest_canonical_sha256"
                ]
    with pytest.raises(ValueError, match="empty output"):
        run_comparison(saved_panels)
    declared = run_comparison(saved_panels, "declared_training", training_language="uk")
    transfer = json.loads((declared / "intervention.json").read_text())[
        "language_transfer"
    ]
    assert transfer["status"] == "available"
    assert transfer["language_pair"] == {"lg1": "uk", "lg2": "en"}
    assert transfer["opposite_slot"] == "lg2"
    assert transfer["signed_ss_change"]["lg2_minus_lg1"] == 0.0


def test_m3_is_paired_language_change_independent_of_target_success():
    strong, balanced = balanced_panels()
    result = _language_transfer(
        {"uk": strong, "en": balanced},
        {"uk": strong, "en": strong},
        target_axis="actor",
        target_group="idp",
        training_language="uk",
        minimum_support=5,
        draws=30,
        seed=42,
    )
    assert result["status"] == "available"
    assert result["signed_ss_change"]["lg1"] == 0.0
    assert result["signed_ss_change"]["lg2"] == 0.5
    assert result["signed_ss_change"]["lg2_minus_lg1"] == 0.5
    assert result["harm_reduction"]["lg2_minus_lg1"] == -0.5
    assert result["harm_reduction"]["ci95"] == [-0.5, -0.5]


@pytest.mark.parametrize(
    "changed",
    ["phase", "seed", "tokenizer", "inputs", "model_args", "tokens", "missing_assets"],
)
def test_saved_run_mismatches_fail_before_writing(saved_panels, changed):
    _full, _before, after, _neutral, _after_neutral, tmp = saved_panels
    _bundle, directory = after["uk"][0]
    if changed == "phase":
        after["uk"] = after["uk"][:-1]
    elif changed == "tokens":
        path = directory / "predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["token_counts"][0] += 1
        write_jsonl(path, rows)
    else:
        path = directory / "run.json"
        run = json.loads(path.read_text())
        if changed == "seed":
            run["seed"] += 1
        elif changed == "model_args":
            run["model_args"] += ",add_bos_token=True"
        elif changed == "missing_assets":
            del run["evaluation_asset_identity"]
        else:
            key = "tokenizer_sha256" if changed == "tokenizer" else "input_sha256"
            run["evaluation_asset_identity"][key] = "0" * 64
        path.write_text(json.dumps(run))
    with pytest.raises(
        ValueError, match="cover|identity|inference settings|fingerprints"
    ):
        run_comparison(saved_panels, "rejected")
    assert not (tmp / "rejected").exists()


def test_changed_phase_text_is_rejected_even_with_rebound_bundle(saved_panels):
    _full, _before, after, _neutral, _after_neutral, tmp = saved_panels
    bundle, directory = after["uk"][0]
    replacement = tmp / "changed_bundle"
    shutil.copytree(bundle.parent, replacement)
    new_bundle = replacement / "manifest.json"
    manifest = json.loads(new_bundle.read_text())
    run = json.loads((directory / "run.json").read_text())
    entry = manifest["datasets"][run["task"]]
    path = replacement / entry["filename"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    source = rows[0]["source"]
    source["released"]["stereotype"] = "A different source claim."
    rows[0] = adapt_annotated(source["released"], source["metadata"], "uk")
    write_jsonl(path, rows)
    entry["sha256"] = digest(path)
    new_bundle.write_text(json.dumps(manifest))
    run["dataset"] = dataset_spec(run["task"], new_bundle)
    (directory / "run.json").write_text(json.dumps(run))
    after["uk"][0] = (new_bundle, directory)
    with pytest.raises(ValueError, match="phase input text differs"):
        run_comparison(saved_panels, "changed_source")
    assert not (tmp / "changed_source").exists()
