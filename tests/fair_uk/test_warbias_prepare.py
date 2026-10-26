"""Panel bundle preparation: row mapping, selection, provenance and neutral tasks."""

import json

import pytest
from warbias_support import PROFILES, inventory_record, neutral_records, released

from lm_eval.fair_uk.data import load
from lm_eval.fair_uk.warbias.prepare import prepare
from lm_eval.fair_uk.warbias.rows import (
    FIELDS,
    SOURCE,
    adapt_annotated,
    validate_adapted_row,
)


UK, EN = "warbias_triplets_annotated_uk", "warbias_triplets_annotated_en"


def test_adapt_maps_profile_component_case_and_texts():
    row = released("WB-0002", "idp", "uk")
    meta = inventory_record(row, "uk", "women", "not_stated", "SG-a", "C1", "train")
    meta.pop("payload")
    adapted = adapt_annotated(row, meta, "uk")
    assert adapted["task"] == UK
    assert adapted["group"] == "idp|women|not_stated"
    assert adapted["cluster"] == "SG-a" and adapted["comparison_unit"] == "C1"
    assert adapted["sentences"] == [
        row["stereotype"],
        row["counter_stereotype"],
        row["unrelated"],
    ]
    assert adapted["final_text_severity"] is None
    validate_adapted_row(adapted)
    with pytest.raises(ValueError, match="differs"):
        validate_adapted_row({**adapted, "group": "idp|men|not_stated"})
    with pytest.raises(ValueError, match="actor"):
        adapt_annotated(row, {**meta, "target_group": "veteran"}, "uk")
    with pytest.raises(ValueError, match="gender"):
        adapt_annotated(row, {**meta, "gender": "female"}, "uk")
    neutral = adapt_annotated(
        row,
        {
            **meta,
            "neutral_variant": dict.fromkeys(FIELDS, "people x."),
        },
        "uk",
    )
    assert neutral["task"] == "warbias_triplets_annotated_neutral_uk"
    assert neutral["sentences"] == ["people x."] * 3


def test_prepare_selects_by_split_scope_and_ids(warbias_inputs):
    files, inv, tmp = warbias_inputs
    manifest = prepare(files["uk"], files["en"], inv, tmp / "all")
    rows = {
        lang: load(f"warbias_triplets_annotated_{lang}", dataset_bundle=manifest)
        for lang in ("uk", "en")
    }
    assert [r["id"] for r in rows["uk"]] == [p[0] for p in PROFILES]
    assert [r["group"] for r in rows["uk"]] == [r["group"] for r in rows["en"]]
    meta = json.loads(manifest.read_text())
    assert meta["pairs"][0]["matched_rows"] == len(PROFILES)
    assert meta["source"]["revision"] == SOURCE["revision"]
    summary = json.loads((tmp / "all" / "panel_summary.json").read_text())["uk"]
    assert summary["rows_by"]["actor"] == {"idp": 3, "veteran": 3}
    assert summary["counter_longer_than_stereotype_rows"] == len(PROFILES)

    def rows_in(path, task=UK):
        return json.loads(path.read_text())["datasets"][task]["rows"]

    assert (
        rows_in(
            prepare(files["uk"], files["en"], inv, tmp / "eligible", scope="eligible")
        )
        == 5
    )
    assert (
        rows_in(
            prepare(files["uk"], files["en"], inv, tmp / "eval", split="evaluation"), EN
        )
        == 2
    )
    restricted = prepare(
        files["uk"],
        files["en"],
        inv,
        tmp / "ids",
        ids=["WB-0001", "WB-0004", "WB-1250"],
    )
    info = json.loads(restricted.read_text())
    assert info["datasets"][UK]["rows"] == 2
    assert info["id_restriction"]["absent_from_release"] == ["WB-1250"]
    with pytest.raises(ValueError, match="empty output"):
        prepare(files["uk"], files["en"], inv, tmp / "all")


def test_prepare_rejects_changed_release_or_inventory_text(warbias_inputs):
    files, inv, tmp = warbias_inputs
    original = files["uk"].read_text(encoding="utf-8")
    files["uk"].write_text(
        original.replace("watch rain", "watch snow", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="pinned revision"):
        prepare(files["uk"], files["en"], inv, tmp / "changed")
    files["uk"].write_text(original, encoding="utf-8")
    records = [
        json.loads(line) for line in inv.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["payload"]["stereotype"] = "edited"
    inv.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    with pytest.raises(ValueError, match="inventory text differs"):
        prepare(files["uk"], files["en"], inv, tmp / "inventory")


def test_aliases_resolve_and_links_mark_exact_cross_actor_claims(warbias_inputs):
    files, inv, tmp = warbias_inputs
    manifest = prepare(
        files["uk"],
        files["en"],
        inv,
        tmp / "aliased",
        ids=["WB-0001", "WB-9001"],
        aliases={"WB-9001": "WB-0004"},
        cross_actor_links={"WB-0001": "claim-x", "WB-0004": "claim-x"},
    )
    info = json.loads(manifest.read_text())
    assert info["id_restriction"]["resolved_aliases"] == {"WB-9001": "WB-0004"}
    assert info["id_restriction"]["absent_from_release"] == []
    rows = load(UK, dataset_bundle=manifest)
    assert {r["id"]: r["cross_actor_claim_id"] for r in rows} == {
        "WB-0001": "claim-x",
        "WB-0004": "claim-x",
    }
    with pytest.raises(ValueError, match="released IDs"):
        prepare(files["uk"], files["en"], inv, tmp / "bad", aliases={"X": "WB-9999"})
    with pytest.raises(ValueError, match="reuse"):
        prepare(
            files["uk"], files["en"], inv, tmp / "bad2", aliases={"WB-0001": "WB-0002"}
        )


def test_deferred_split_is_a_detection_only_evaluation_role(warbias_inputs):
    files, inv, tmp = warbias_inputs
    manifest = prepare(
        files["uk"], files["en"], inv, tmp / "deferred", split="deferred"
    )
    spec = json.loads(manifest.read_text())["datasets"][UK]
    assert spec["rows"] == 1
    assert spec["source_role"] == "evaluation"
    assert spec["measurement_purpose"] == "pre_intervention_detection_only_measurement"
    assert spec["training_gradients_authorized"] is False
    assert spec["source_provenance"][0]["published"] is True


def test_neutral_tasks_cover_only_ids_valid_in_both_languages(
    warbias_inputs, monkeypatch
):
    files, inv, tmp = warbias_inputs
    records = neutral_records(files, monkeypatch)
    assert sum(r["status"] == "valid" for r in records) == 2 * (len(PROFILES) - 1)
    manifest = prepare(files["uk"], files["en"], inv, tmp / "n", neutral=records)
    info = json.loads(manifest.read_text())
    assert info["neutral_variants"]["valid_in_both_languages"] == len(PROFILES) - 1
    neutral_rows = load(
        "warbias_triplets_annotated_neutral_uk", dataset_bundle=manifest
    )
    assert "WB-0001" not in {r["id"] for r in neutral_rows}
    assert all("people" in r["sentences"][0] for r in neutral_rows)
    assert info["datasets"]["warbias_triplets_annotated_neutral_en"]["counterpart"] == (
        "warbias_triplets_annotated_neutral_uk"
    )
    with pytest.raises(ValueError, match="Unsupported neutral"):
        prepare(
            files["uk"],
            files["en"],
            inv,
            tmp / "v0",
            neutral=[{**records[0], "version": "v0"}],
        )
