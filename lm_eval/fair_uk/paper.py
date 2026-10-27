"""Portable, saved-evidence entry points for paper evaluation."""

import argparse
import json
from pathlib import Path

from lm_eval.fair_uk.contracts import validate_data_release
from lm_eval.fair_uk.releases import freeze_evaluation_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "validate-release", help="Check exact frozen input bytes"
    )
    check.add_argument("--release", type=Path, required=True)
    freeze = commands.add_parser(
        "freeze-evaluation", help="Package a verified adapted evaluation bundle"
    )
    freeze.add_argument("--bundle", type=Path, required=True)
    freeze.add_argument("--release-id", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    assets = commands.add_parser(
        "materialize-inputs",
        help="Verify and unpack the separate native evaluation asset",
    )
    assets.add_argument("--catalog", type=Path, required=True)
    assets.add_argument("--archive", type=Path, required=True)
    assets.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate-release":
        record = json.loads(args.release.read_text(encoding="utf-8"))
        validate_data_release(record, root=args.release.parent)
        print(
            json.dumps(
                {
                    "release_id": record["release_id"],
                    "files_verified": len(record["files"]),
                    "qualification": record["qualification"],
                }
            )
        )
    elif args.command == "freeze-evaluation":
        print(
            freeze_evaluation_bundle(
                args.bundle, args.output, release_id=args.release_id
            )
        )
    elif args.command == "materialize-inputs":
        from lm_eval.fair_uk.assets import materialize

        print(materialize(args.catalog, args.archive, args.output))


if __name__ == "__main__":
    main()
