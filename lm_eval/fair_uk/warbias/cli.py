"""``fair-uk-eval warbias``: the WarBias detection pipeline from the command line.

prepare           build the pinned panel bundle (optionally with neutral variants)
report            whole-panel report for one checkpoint from saved phase runs
calibrate         tau on prior-normalised development margins from a report
tables            paper tables from a directory of panel reports
compare           paired checkpoint effects using original + neutral runs

WarBias creation, translation, neutral rewriting and case annotation live in
the FairForget producer repository. Fair-UK consumes their frozen releases.
"""

import argparse
from pathlib import Path

from lm_eval.fair_uk.warbias.calibration import development_tau
from lm_eval.fair_uk.warbias.files import read_pairs, read_rows, write_json
from lm_eval.fair_uk.warbias.panel import panel_report
from lm_eval.fair_uk.warbias.prepare import ROLES, prepare
from lm_eval.fair_uk.warbias.schema import LANGUAGES
from lm_eval.fair_uk.warbias.tables import write_tables


def _ids(path):
    return [
        line.strip() for line in Path(path).read_text().splitlines() if line.strip()
    ]


def _phase_runs(entries):
    runs = {language: [] for language in LANGUAGES}
    for language, bundle, directory in entries or []:
        if language not in runs:
            raise ValueError(f"Unknown language {language}")
        runs[language].append((Path(bundle), Path(directory)))
    return runs


def _parser():
    parser = argparse.ArgumentParser(
        prog="fair-uk-eval warbias",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("prepare", help="Build an adapted panel bundle")
    build.add_argument("--uk", type=Path, required=True, help="Released UK triplets")
    build.add_argument("--en", type=Path, required=True, help="Released EN triplets")
    build.add_argument("--inventory", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--split", choices=("all", *ROLES), default="all")
    build.add_argument("--scope", choices=("all", "eligible"), default="all")
    build.add_argument("--ids", type=Path, help="Restrict to these IDs (one per line)")
    build.add_argument("--aliases", type=Path, help="CSV workbook_id,release_id")
    build.add_argument(
        "--cross-actor-links", type=Path, help="CSV id,cross_actor_claim_id"
    )
    build.add_argument("--case-splits", type=Path, help="Published case_splits.csv")
    build.add_argument("--neutral", type=Path, help="Neutral-variant records (JSONL)")

    report = commands.add_parser("report", help="Whole-panel report from phase runs")
    report.add_argument("--full-bundle", type=Path, required=True)
    run_help = "Repeat once per language and phase"
    report.add_argument(
        "--run",
        nargs=3,
        action="append",
        required=True,
        metavar=("LANG", "BUNDLE", "RUN_DIR"),
        help=run_help,
    )
    report.add_argument(
        "--neutral-run",
        nargs=3,
        action="append",
        metavar=("LANG", "BUNDLE", "RUN_DIR"),
        help=run_help,
    )
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--min-independent-support", type=int, default=5)
    report.add_argument("--bootstrap", type=int, default=1000)
    report.add_argument("--seed", type=int, default=42)

    calibrate = commands.add_parser(
        "calibrate", help="Tau from a panel report (development)"
    )
    calibrate.add_argument("--dev-bundle", type=Path, required=True)
    calibrate.add_argument(
        "--panel", type=Path, required=True, help="Panel report directory"
    )
    calibrate.add_argument("--model-id", required=True)
    calibrate.add_argument("--floor", type=float, default=0.0)
    calibrate.add_argument("--output", type=Path, required=True)

    tables = commands.add_parser("tables", help="Paper tables from panel reports")
    tables.add_argument(
        "--results", type=Path, required=True, help="Directory of panel_<model>/"
    )
    tables.add_argument("--output", type=Path, help="Defaults to --results")
    tables.add_argument(
        "--model", action="append", help="Model order (default: all, sorted)"
    )

    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            args.uk,
            args.en,
            args.inventory,
            args.output,
            split=args.split,
            scope=args.scope,
            ids=None if args.ids is None else _ids(args.ids),
            aliases=None
            if args.aliases is None
            else read_pairs(args.aliases, "workbook_id", "release_id"),
            cross_actor_links=None
            if args.cross_actor_links is None
            else read_pairs(args.cross_actor_links, "id", "cross_actor_claim_id"),
            case_splits=args.case_splits,
            neutral=None if args.neutral is None else read_rows(args.neutral),
        )
    elif args.command == "report":
        result = panel_report(
            args.full_bundle,
            _phase_runs(args.run),
            args.output,
            neutral_runs=_phase_runs(args.neutral_run) if args.neutral_run else None,
            min_independent_support=args.min_independent_support,
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
    elif args.command == "calibrate":
        write_json(
            args.output,
            development_tau(
                args.dev_bundle, args.panel, model_id=args.model_id, floor=args.floor
            ),
        )
        result = args.output
    elif args.command == "tables":
        result = "\n".join(
            map(str, write_tables(args.results, args.output, args.model))
        )
    print(result)


if __name__ == "__main__":
    main()
