"""Build and verify a portable WarBias code submission."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path


MANIFEST = "MANIFEST.sha256.json"
EXCLUDED = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
}
SOURCE_SUFFIXES = {".py", ".json", ".yaml", ".yml"}
README = """# WarBias paper artifact

Standalone source for the WarBias detection pipeline and its evaluator support.
Start with the [paper workflow](docs/paper-reproduction.md) and
[WarBias guide](docs/warbias-detection.md) for installation,
input requirements, commands, metric definitions and anonymous-release notes.

- `src/lm_eval/`: pipeline and reusable evaluator interfaces.
- `tests/fair_uk/`: offline software verification and optional saved regressions.
- `MANIFEST.sha256.json` and `tools/verify_manifest.py`: source integrity and
  import-origin verification; the guide shows their use.

Use a dedicated environment: this package retains the `lm_eval` namespace.
Preserve the [upstream MIT license](LICENSE.md). Scientific source citations
remain in the guide. The compact frozen WarBias evaluation inputs are included
under `data/paper/`; model weights and raw run outputs remain separate artifacts.
"""


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_path(parts):
    return any(p in EXCLUDED or p.endswith(".egg-info") for p in parts) or parts[
        -1
    ].endswith((".pyc", ".pyo"))


def _copy_tree(source, target, *, omit_task_collections=False, suffixes=None):
    if source.is_symlink():
        raise ValueError(f"Symlinks are not permitted: {source}")
    if not source.is_dir():
        raise ValueError(f"Missing source directory: {source}")
    for parent, directories, files in os.walk(source, followlinks=False):
        relative = Path(parent).relative_to(source)
        retained = []
        for name in sorted(directories):
            path = Path(parent) / name
            parts = (*relative.parts, name)
            if _runtime_path(parts) or name.startswith("."):
                continue
            if omit_task_collections and parts[0] == "tasks" and len(parts) > 1:
                continue
            if path.is_symlink():
                raise ValueError(f"Symlinks are not permitted: {path}")
            retained.append(name)
        directories[:] = retained
        for name in sorted(files):
            parts = (*relative.parts, name)
            if (
                not _runtime_path(parts)
                and not name.startswith(".")
                and (suffixes is None or Path(name).suffix in suffixes)
            ):
                _copy_file(Path(parent) / name, target / relative / name)


def _copy_file(source, target):
    if source.is_symlink():
        raise ValueError(f"Symlinks are not permitted: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _project_toml(source):
    raw = source.read_text(encoding="utf-8")
    metadata = _read_toml(raw)
    project = metadata["project"]
    dump = lambda value: json.dumps(value, ensure_ascii=False)
    lines = [
        "[build-system]",
        'requires = ["setuptools>=64.0"]',
        'build-backend = "setuptools.build_meta"',
        "",
        "[project]",
        'name = "lm_eval"',
        f"version = {dump(project['version'])}",
        'description = "WarBias detection pipeline and evaluator support"',
        'readme = "README.md"',
        f"requires-python = {dump(project['requires-python'])}",
        'license = {text = "MIT"}',
        f"dependencies = {dump(project['dependencies'])}",
        "",
        "[project.optional-dependencies]",
        *(
            f"{dump(key)} = {dump(value)}"
            for key, value in project.get("optional-dependencies", {}).items()
        ),
        "",
        "[project.scripts]",
        'fair-uk-eval = "lm_eval.fair_uk.__main__:main"',
        'warbias-detect = "lm_eval.fair_uk.warbias.cli:main"',
        "",
        "[tool.setuptools.packages.find]",
        'where = ["src"]',
        'include = ["lm_eval*"]',
        "",
        "[tool.setuptools.package-data]",
        'lm_eval = ["**/*.yaml", "**/*.json"]',
        "",
    ]
    # Preserve maintained lint/test settings without author or repository metadata.
    current = None
    for line in raw.splitlines():
        if line.startswith("["):
            current = line.split("]", 1)[0] + "]"
        if current and (current.startswith("[tool.ruff") or current == "[tool.pytest]"):
            lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def _read_toml(raw):
    """Use the standard parser, with pytest's compatibility dependency on 3.10."""
    try:
        from tomllib import loads
    except ModuleNotFoundError:
        from tomli import loads
    return loads(raw)


def _source_files(directory):
    for path in sorted(directory.rglob("*")):
        parts = path.relative_to(directory).parts
        if _runtime_path(parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks are not permitted: {path}")
        if path.is_file() and str(path.relative_to(directory)) != MANIFEST:
            yield path


def verify(directory, *, check_imports=False):
    """Validate the complete file inventory and optional editable import origin."""
    directory = Path(directory).resolve()
    record = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or not isinstance(record.get("files"), dict):
        raise ValueError("Unsupported submission manifest")
    actual = {str(p.relative_to(directory)): _hash(p) for p in _source_files(directory)}
    expected = record["files"]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            k for k in set(actual) & set(expected) if actual[k] != expected[k]
        )
        raise ValueError(
            f"Manifest mismatch: missing={missing}, extra={extra}, changed={changed}"
        )
    if check_imports:
        spec = importlib.util.find_spec("lm_eval")
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        if origin != directory / "src/lm_eval/__init__.py":
            raise ValueError(
                "lm_eval must resolve to this export's src tree; install it editable in a dedicated environment"
            )
    return len(actual)


def export(source, output):
    """Export allowlisted current source without modifying or linking the checkout."""
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists():
        raise ValueError("Export destination must be a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".warbias-export-", dir=output.parent))
    try:
        _copy_tree(
            source / "lm_eval",
            staging / "src/lm_eval",
            omit_task_collections=True,
            suffixes=SOURCE_SUFFIXES,
        )
        _copy_tree(source / "tests/fair_uk", staging / "tests/fair_uk")
        _copy_tree(source / "data/paper", staging / "data/paper")
        _copy_tree(source / "notebooks/paper", staging / "notebooks/paper")
        _copy_tree(source / "configs/paper", staging / "configs/paper")
        for relative in (
            "tests/conftest.py",
            "docs/warbias-detection.md",
            "docs/paper-reproduction.md",
            "LICENSE.md",
        ):
            _copy_file(source / relative, staging / relative)
        _copy_file(Path(__file__), staging / "tools/verify_manifest.py")
        (staging / "README.md").write_text(README, encoding="utf-8")
        (staging / "pyproject.toml").write_text(
            _project_toml(source / "pyproject.toml"), encoding="utf-8"
        )
        config = json.loads(
            (source / "configs/fair_uk/jev_benign.json").read_text(encoding="utf-8")
        )
        config.pop("keychain", None)
        (staging / "configs/fair_uk").mkdir(parents=True)
        (staging / "configs/fair_uk/jev_benign.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        files = {str(p.relative_to(staging)): _hash(p) for p in _source_files(staging)}
        (staging / MANIFEST).write_text(
            json.dumps({"schema_version": 1, "files": files}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        verify(staging)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Create a standalone source directory")
    build.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    build.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("verify", help="Check exported source integrity")
    check.add_argument("--directory", type=Path, required=True)
    check.add_argument("--check-imports", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            print(export(args.source, args.output))
        else:
            print(
                f"Verified {verify(args.directory, check_imports=args.check_imports)} files"
            )
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
