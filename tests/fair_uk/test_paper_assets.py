"""Large data assets are portable and independently checked outside Git."""

import io
import json
import tarfile

import pytest

from lm_eval.fair_uk.assets import materialize
from lm_eval.fair_uk.contracts import file_sha256


def asset(tmp_path, name="manifest.json", content=b"{}", *, wrong_hash=False):
    archive = tmp_path / "input.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        stream.addfile(info, io.BytesIO(content))
    catalog = tmp_path / "catalog.json"
    import hashlib

    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "evaluation_input_archive",
                "manifest": "manifest.json",
                "archive": {"sha256": file_sha256(archive)},
                "files": {
                    name: {
                        "sha256": "0" * 64
                        if wrong_hash
                        else hashlib.sha256(content).hexdigest()
                    }
                },
            }
        )
    )
    return catalog, archive


def test_materialize_checks_bytes_and_preserves_destination(tmp_path):
    catalog, archive = asset(tmp_path)
    output = tmp_path / "restored"
    assert materialize(catalog, archive, output).read_bytes() == b"{}"
    with pytest.raises(ValueError, match="new output"):
        materialize(catalog, archive, output)
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from its catalog"):
        materialize(catalog, archive, tmp_path / "other")


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/absolute"])
def test_archive_cannot_escape_destination(tmp_path, name):
    catalog, archive = asset(tmp_path, name)
    with pytest.raises(ValueError, match="nonportable"):
        materialize(catalog, archive, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_file_hash_failure_removes_partial_output(tmp_path):
    catalog, archive = asset(tmp_path, wrong_hash=True)
    output = tmp_path / "restored"
    with pytest.raises(ValueError, match="digest differs"):
        materialize(catalog, archive, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "manifest",
    ["../outside.json", "/absolute.json", "C:/absolute.json", "missing.json"],
)
def test_catalog_manifest_must_be_inside_the_inventoried_archive(tmp_path, manifest):
    catalog, archive = asset(tmp_path)
    record = json.loads(catalog.read_text())
    record["manifest"] = manifest
    catalog.write_text(json.dumps(record))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="portable inventoried"):
        materialize(catalog, archive, output)
    assert not output.exists()
