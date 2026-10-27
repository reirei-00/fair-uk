"""Portable source export and integrity checks, without network or model calls."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "export_warbias", ROOT / "scripts/export_warbias_submission.py"
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


@pytest.fixture
def exported(tmp_path):
    return exporter.export(ROOT, tmp_path / "submission")


def test_export_is_standalone_and_reproducible(exported, tmp_path):
    other = exporter.export(ROOT, tmp_path / "second")
    assert (exported / exporter.MANIFEST).read_bytes() == (
        other / exporter.MANIFEST
    ).read_bytes()
    assert exporter.verify(exported) > 100
    assert (exported / "src/lm_eval/fair_uk/warbias/cli.py").is_file()
    assert (exported / "src/lm_eval/tasks/manager.py").is_file()
    assert not (exported / "src/lm_eval/tasks/mmlu").exists()
    assert not any(p.is_symlink() for p in exported.rglob("*"))
    metadata = exporter._read_toml((exported / "pyproject.toml").read_text())
    assert metadata["project"]["scripts"]["warbias-detect"].endswith("cli:main")
    assert "authors" not in metadata["project"] and "urls" not in metadata["project"]
    for excluded in (".git", ".venv", "outputs", "CONVENTIONS.md"):
        assert not (exported / excluded).exists()
    with pytest.raises(ValueError, match="new directory"):
        exporter.export(ROOT, exported)


@pytest.mark.parametrize("damage", ["alter", "delete", "extra", "hidden"])
def test_manifest_detects_changed_missing_and_added_source(exported, damage):
    target = exported / "src/lm_eval/fair_uk/warbias/scoring.py"
    if damage == "alter":
        target.write_text("# changed\n")
    elif damage == "delete":
        target.unlink()
    elif damage == "extra":
        (exported / "unexpected.py").write_text("# extra\n")
    else:
        (exported / ".env").write_text("UNEXPECTED=value\n")
    with pytest.raises(ValueError, match="Manifest mismatch"):
        exporter.verify(exported)


def test_verifier_rejects_import_from_parent_checkout(exported):
    # The test interpreter resolves the maintained checkout. A copied verifier
    # must detect that contamination rather than count those imports as proof.
    with pytest.raises(ValueError, match="this export's src tree"):
        exporter.verify(exported, check_imports=True)


def test_copied_verifier_runs_without_parent_checkout(exported, tmp_path):
    result = subprocess.run(  # noqa: S603 -- execute the just-exported verifier with fixed arguments
        [
            sys.executable,
            "-I",
            str(exported / "tools/verify_manifest.py"),
            "verify",
            "--directory",
            str(exported),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Verified" in result.stdout


def test_source_symlink_is_rejected(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    (source / "external.py").symlink_to(ROOT / "lm_eval/__init__.py")
    with pytest.raises(ValueError, match="Symlinks"):
        exporter._copy_tree(source, target)
