"""Install the registered source-matched English evaluation release."""

import argparse
import errno
import os
from pathlib import Path
from tempfile import TemporaryDirectory


NAME = "matched-en-v1"
REGISTRATION = Path(__file__).with_name(f"{NAME}.json")


def installed_manifest():
    home = Path(os.environ.get("FAIR_UK_DATA_HOME", Path.home() / ".cache/fair-uk"))
    return home / NAME / "manifest.json"


def verify_manifest(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Install {NAME} with python -m lm_eval.fair_uk.matched_release --source-root PATH"
        )
    if path.read_bytes() != REGISTRATION.read_bytes():
        raise ValueError(f"{path} differs from registered release {NAME}")
    return path


def verify_release(path):
    from lm_eval.fair_uk.data import bundle_registry, load

    for task in bundle_registry(verify_manifest(path)):
        load(task, dataset_bundle=path)


def install(source_root):
    from lm_eval.fair_uk.bilingual import prepare

    path = installed_manifest()
    if path.parent.exists():
        verify_release(path)
        return path
    path.parent.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent.parent) as tmp:
        staged = prepare(source_root, Path(tmp) / NAME)
        verify_release(staged)
        try:
            staged.parent.rename(path.parent)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            verify_release(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    print(install(parser.parse_args().source_root))
