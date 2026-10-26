"""CSV review templates that preserve existing human annotations on resume."""

import csv
from pathlib import Path


def write_sheet(path, fields, rows):
    """Create or extend a review sheet without rewriting existing human edits."""
    path = Path(path)
    try:
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    except FileExistsError:
        _append_review_rows(path, fields, rows)


def _append_review_rows(path, fields, rows):
    """Append only new cases; incompatible sheets need an explicit new version."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(fields):
            raise ValueError("Existing review sheet has a different schema")
        existing = [row["case_id"] for row in reader]
    current = {row["case_id"] for row in rows}
    existing_ids = set(existing)
    if len(existing_ids) != len(existing) or existing_ids - current:
        raise ValueError("Existing review sheet has duplicate or removed case IDs")
    added = [row for row in rows if row["case_id"] not in existing_ids]
    if not added:
        return
    with path.open("rb") as stream:
        stream.seek(-1, 2)
        needs_newline = stream.read(1) not in (b"\n", b"\r")
    with path.open("a", encoding="utf-8", newline="") as stream:
        if needs_newline:
            stream.write("\n")
        csv.DictWriter(stream, fieldnames=fields).writerows(added)
