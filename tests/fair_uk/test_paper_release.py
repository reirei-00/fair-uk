"""The paper release remains byte-identical to the frozen evaluation inputs."""

import json
import shutil
from pathlib import Path

import pytest

from lm_eval.fair_uk.contracts import file_sha256, validate_data_release
from lm_eval.fair_uk.paper import main
from lm_eval.fair_uk.releases import freeze_evaluation_bundle


RELEASE = Path(__file__).parents[2] / "data/paper/warbias-evaluation-v3"


def test_supplied_release_checksums_and_exact_coverage(capsys):
    record = json.loads((RELEASE / "release.json").read_text())
    validate_data_release(record, root=RELEASE)
    assert len(record["item_ids"]) == 186
    assert record["qualification"]["human_validated"] is False
    assert record["coverage"]["warbias_triplets_annotated_neutral_en"] == 175
    assert record["results"]["after"] is None
    main(["validate-release", "--release", str(RELEASE / "release.json")])
    assert json.loads(capsys.readouterr().out)["files_verified"] == 10


def test_refreeze_preserves_adapter_bytes_and_cannot_clobber(tmp_path):
    output = tmp_path / "copy"
    release = freeze_evaluation_bundle(
        RELEASE / "manifest.json", output, release_id="copy"
    )
    assert file_sha256(output / "manifest.json") == file_sha256(
        RELEASE / "manifest.json"
    )
    validate_data_release(json.loads(release.read_text()), root=output)
    with pytest.raises(ValueError, match="new output"):
        freeze_evaluation_bundle(RELEASE / "manifest.json", output, release_id="again")


@pytest.mark.parametrize(
    "task", ["warbias_triplets_annotated_en", "warbias_triplets_annotated_neutral_en"]
)
def test_refreeze_rejects_wrong_source_coverage_before_writing(tmp_path, task):
    source = tmp_path / "source"
    shutil.copytree(RELEASE, source)
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["datasets"][task]["rows"] += 1
    path.write_text(json.dumps(manifest))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="unexpected row count"):
        freeze_evaluation_bundle(path, output, release_id="wrong-coverage")
    assert not output.exists()
