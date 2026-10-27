"""Review file extension preserves edited bytes and rejects incompatible schemas."""

import csv

import pytest

from lm_eval.fair_uk.warbias.review import write_sheet


FIELDS = ("case_id", "comment")


def test_append_after_missing_final_newline_preserves_quoted_review(tmp_path):
    sheet = tmp_path / "human.csv"
    original = b'case_id,comment\r\nA,"Human comment, with comma"'
    sheet.write_bytes(original)
    write_sheet(sheet, FIELDS, [{"case_id": "A"}, {"case_id": "B"}])
    assert sheet.read_bytes().startswith(original)
    with sheet.open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [
            {"case_id": "A", "comment": "Human comment, with comma"},
            {"case_id": "B", "comment": ""},
        ]


@pytest.mark.parametrize(
    "content, expected",
    [
        ("case_id,other\nA,review\n", "different schema"),
        ("case_id,comment\nA,review\nA,second\n", "duplicate or removed"),
        ("case_id,comment\nZ,review\n", "duplicate or removed"),
    ],
)
def test_incompatible_sheet_remains_untouched(tmp_path, content, expected):
    sheet = tmp_path / "human.csv"
    sheet.write_text(content)
    with pytest.raises(ValueError, match=expected):
        write_sheet(sheet, FIELDS, [{"case_id": "A"}])
    assert sheet.read_text() == content
