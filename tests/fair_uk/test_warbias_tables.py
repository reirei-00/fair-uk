"""Rendering uses explicit language slots, including historical saved reports."""

import csv
import json

import pytest

from lm_eval.fair_uk.warbias.detection import (
    detection_analysis,
    prior_normalised_report,
)
from lm_eval.fair_uk.warbias.schema import TripletItem
from lm_eval.fair_uk.warbias.tables import write_tables


def item(actor, preference, *, suffix=""):
    return TripletItem(
        id=actor + suffix,
        cluster="component_" + actor,
        actor=actor,
        gender="not_stated",
        age="not_stated",
        comparison_unit="case_" + actor,
        preference=preference,
    )


def save_panel(root, items_by_language, *, pair=("fr", "de"), model="m"):
    panel = root / f"panel_{model}"
    panel.mkdir()
    for language, items in items_by_language.items():
        directory = panel / language
        directory.mkdir()
        (directory / "prior_normalised.json").write_text(
            json.dumps(prior_normalised_report(items, len(items)))
        )
    (panel / "detection_analysis.json").write_text(
        json.dumps(detection_analysis(items_by_language, language_pair=pair))
    )
    return panel


def tables(tmp_path, lg1, lg2, *, pair=("fr", "de")):
    save_panel(tmp_path, dict(zip(pair, (lg1, lg2), strict=True)), pair=pair)
    paths = write_tables(tmp_path)
    with paths[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return paths[2].read_text(), rows


@pytest.mark.parametrize("actor", ["idp", "veteran"])
def test_tables_render_missing_actor_as_unavailable(tmp_path, actor):
    text, rows = tables(tmp_path, [item(actor, 0.0)], [item(actor, 1.0)])
    assert "## LG1 vs LG2 (same 1 triplets)" in text
    assert "LG1 = fr; LG2 = de. Differences are LG2 − LG1." in text
    contrasts = "100.0 | —" if actor == "idp" else "— | 100.0"
    assert f"| m | 0.0 | 100.0 | 100.0 | {contrasts} |" in text
    assert {r["group"] for r in rows if r["axis"] == "actor"} == {actor}
    assert "| m | fr | — | — | — |" in text  # no cross-actor match
    assert {(r["lg1_code"], r["lg2_code"]) for r in rows} == {("fr", "de")}
    all_rows = [r for r in rows if r["axis"] == "all"]
    assert [(r["language_slot"], r["language"]) for r in all_rows] == [
        ("lg1", "fr"),
        ("lg2", "de"),
    ]
    assert all_rows[0]["gap_ss_lg2_minus_lg1"] == ""
    assert float(all_rows[1]["gap_ss_lg2_minus_lg1"]) == 100.0
    baseline = (tmp_path / "baseline_tables.md").read_text()
    assert "LG2−LG1 SS" in baseline
    assert "| all: all † | LG1 (fr) |" in baseline
    assert "| all: all † | LG2 (de) |" in baseline


def test_tables_keep_complete_actor_contrasts(tmp_path):
    text, _ = tables(
        tmp_path,
        [item("idp", 1.0), item("veteran", 0.0)],
        [item("idp", 0.0), item("veteran", 1.0)],
    )
    assert "## LG1 vs LG2 (same 2 triplets)" in text
    assert "| m | 50.0 | 50.0 | 0.0 | -100.0 | 100.0 |" in text


def test_tables_render_unmatched_languages_as_unavailable(tmp_path):
    text, rows = tables(
        tmp_path,
        [item("idp", 0.0)],
        [item("idp", 1.0, suffix="_different")],
    )
    assert "## LG1 vs LG2 (no matched triplets)" in text
    assert "| m | — | — | — | — | — |" in text
    assert all(not row["gap_ss_lg2_minus_lg1"] for row in rows)


def test_reversed_pair_controls_direction_and_row_order(tmp_path):
    save_panel(
        tmp_path,
        {"fr": [item("idp", 0.0)], "de": [item("idp", 1.0)]},
        pair=("de", "fr"),
    )
    paths = write_tables(tmp_path)
    with paths[0].open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["axis"] == "all"]
    assert [(r["language_slot"], r["language"]) for r in rows] == [
        ("lg1", "de"),
        ("lg2", "fr"),
    ]
    assert float(rows[1]["gap_ss_lg2_minus_lg1"]) == -100.0
    text = paths[2].read_text()
    assert "LG1 = de; LG2 = fr." in text
    assert "| m | 100.0 | 0.0 | -100.0 | -100.0 | — |" in text


def test_models_with_different_ordered_pairs_rejected_before_writes(tmp_path):
    items = {"fr": [item("idp", 0.0)], "de": [item("idp", 1.0)]}
    save_panel(tmp_path, items, pair=("fr", "de"), model="m")
    save_panel(tmp_path, items, pair=("de", "fr"), model="n")
    output = tmp_path / "new_tables"
    with pytest.raises(ValueError, match="same ordered language pair"):
        write_tables(tmp_path, output)
    assert not output.exists()


def test_legacy_v2_reader_preserves_saved_report_bytes(tmp_path):
    panel = save_panel(
        tmp_path,
        {"uk": [item("idp", 0.0)], "en": [item("idp", 1.0)]},
        pair=("uk", "en"),
    )
    path = panel / "detection_analysis.json"
    report = json.loads(path.read_text())
    report["version"] = "warbias_detection_report_v2"
    contrast = report["language_contrast"]
    del contrast["language_pair"]

    def historical(block):
        if block is None:
            return None
        renamed = {"lg1": "uk", "lg2": "en", "lg2_minus_lg1": "en_minus_uk"}
        return {renamed.get(k, k): v for k, v in block.items()}

    contrast["all"] = historical(contrast["all"])
    contrast["by_actor"] = {k: historical(v) for k, v in contrast["by_actor"].items()}
    for cell in contrast["cells"]:
        for field in ("preference", "lms_per_byte"):
            cell[field] = historical(cell[field])
    path.write_text(json.dumps(report))
    before = {p: p.read_bytes() for p in panel.rglob("*.json")}
    paths = write_tables(tmp_path)
    assert {p: p.read_bytes() for p in before} == before
    text = paths[2].read_text()
    assert "LG1 = uk; LG2 = en." in text
    assert "| m | 0.0 | 100.0 | 100.0 | 100.0 | — |" in text


@pytest.mark.parametrize("version", [None, "warbias_detection_report_v1", "future"])
def test_unknown_report_versions_are_rejected(tmp_path, version):
    panel = save_panel(tmp_path, {"fr": [item("idp", 0)], "de": [item("idp", 1)]})
    path = panel / "detection_analysis.json"
    report = json.loads(path.read_text())
    report["version"] = version
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="unsupported detection report version"):
        write_tables(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "mapping",
    [None, {}, {"lg1": "fr"}, {"lg1": "fr", "lg2": "fr"}, {"lg1": "fr", "lg2": "it"}],
)
def test_current_reports_require_a_valid_language_mapping(tmp_path, mapping):
    panel = save_panel(tmp_path, {"fr": [item("idp", 0)], "de": [item("idp", 1)]})
    path = panel / "detection_analysis.json"
    report = json.loads(path.read_text())
    report["language_contrast"]["language_pair"] = mapping
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="language|Language"):
        write_tables(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
