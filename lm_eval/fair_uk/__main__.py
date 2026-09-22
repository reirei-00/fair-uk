"""Run model evaluations or rescore saved predictions without model inference."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from lm_eval.fair_uk import VERSION
from lm_eval.fair_uk.answers import NORMALIZED, POLICIES
from lm_eval.fair_uk.data import REGISTRY, load, select_clusters
from lm_eval.fair_uk.metrics import make_report, score_all
from lm_eval.fair_uk.provenance import code_identity
from lm_eval.fair_uk.reporting import (
    group_rows,
    markdown,
    paired_markdown,
    summarize,
    table,
    write_csv,
)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "suite":
        from lm_eval.fair_uk.suite import main as suite_main

        return suite_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("suite", help="Plan or explicitly run a set of bias benchmarks")
    listing = commands.add_parser("list", help="List pinned benchmark tasks")
    listing.add_argument("--format", choices=("json", "table"), default="json")
    listing.add_argument("--language", choices=("uk", "en"))
    for command in ("validate", "run", "score", "run-openai", "run-gemini"):
        sub = commands.add_parser(command)
        sub.add_argument("--task", choices=sorted(REGISTRY), required=True)
        sub.add_argument(
            "--input",
            type=Path,
            help="Local copy of the registered HF file; checksum is still enforced",
        )
        sub.add_argument("--limit-clusters-per-stratum", type=int)
        if command != "validate":
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--bootstrap", type=int, default=1000)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--answer-policy", choices=POLICIES, default=NORMALIZED)
        if command == "run":
            sub.add_argument("--model", default="hf", help="Harness model backend")
            sub.add_argument("--model-args", required=True)
            sub.add_argument("--batch-size", default="auto")
            sub.add_argument("--chunk-size", type=int, default=32)
        if command == "run-openai":
            from lm_eval.fair_uk.openai_runner import SNAPSHOTS

            sub.add_argument("--snapshot", choices=sorted(SNAPSHOTS), required=True)
            sub.add_argument("--concurrency", type=int, default=4)
            sub.add_argument("--max-estimated-usd", type=float, default=2.0)
            sub.add_argument(
                "--dry-run",
                action="store_true",
                help="Show a request/cost estimate without calling OpenAI",
            )
        if command == "run-gemini":
            from lm_eval.fair_uk.gemini_runner import MODELS

            sub.add_argument("--snapshot", choices=sorted(MODELS), required=True)
            sub.add_argument("--concurrency", type=int, default=4)
            sub.add_argument("--max-estimated-usd", type=float, default=2.0)
            sub.add_argument("--dry-run", action="store_true")
        if command == "score":
            sub.add_argument("--predictions", type=Path, required=True)
    compare = commands.add_parser("compare", help="Compare paired UK/EN WarBias runs")
    compare.add_argument("--uk-run", type=Path, required=True)
    compare.add_argument("--en-run", type=Path, required=True)
    compare.add_argument("--uk-input", type=Path)
    compare.add_argument("--en-input", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--bootstrap", type=int, default=1000)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--answer-policy", choices=POLICIES, default=NORMALIZED)
    summary = commands.add_parser(
        "summarize", help="Build model-by-task experiment tables"
    )
    summary.add_argument("--reports", type=Path, nargs="+", required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "summarize":
        summarize(args.reports, args.output)
        print(f"Summary written to {args.output.resolve() / 'summary.md'}")
        return
    if args.command == "compare":
        from lm_eval.fair_uk.comparison import compare_runs

        report = compare_runs(
            args.uk_run,
            args.en_run,
            args.uk_input,
            args.en_input,
            args.bootstrap,
            args.seed,
            answer_policy=args.answer_policy,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "comparison.json", report)
        (args.output / "comparison.md").write_text(paired_markdown(report))
        print(f"Comparison written to {args.output.resolve() / 'comparison.md'}")
        return
    if args.command == "list":
        selected = {
            task: spec
            for task, spec in REGISTRY.items()
            if args.language is None or spec["language"] == args.language
        }
        if args.format == "table":
            print(
                table(
                    ["Task", "Language", "Rows", "Scoring input", "HF dataset"],
                    [
                        (
                            task,
                            spec["language"],
                            spec["rows"],
                            "generated answer"
                            if task.startswith("warbias_")
                            else "token log probabilities",
                            spec["repo"],
                        )
                        for task, spec in sorted(selected.items())
                    ],
                )
            )
            print(
                "\nPinned pilot releases; human validation pending. Use --format json for revisions and checksums."
            )
        else:
            print(json.dumps(selected, indent=2))
        return
    registry_rows = load(args.task, args.input)
    rows = select_clusters(registry_rows, args.limit_clusters_per_stratum)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "task": args.task,
                    "rows": len(rows),
                    "source_cases": len({r["cluster"] for r in rows}),
                    "groups": sorted({r["group"] for r in rows}),
                    "passed": True,
                }
            )
        )
        return
    if args.bootstrap < 0:
        parser.error("--bootstrap cannot be negative")
    if args.command in ("run-openai", "run-gemini") and args.dry_run:
        from lm_eval.fair_uk.openai_runner import estimate, identity, request_payload

        if args.command == "run-gemini":
            from lm_eval.fair_uk.gemini_runner import (
                estimate,
                identity,
                request_payload,
            )
        model = identity(args.snapshot, args.seed)
        for row in rows:
            request_payload(row, model)
        print(json.dumps(estimate(rows, args.snapshot), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    run = {
        "version": VERSION,
        "code_identity": code_identity(),
        "dataset": REGISTRY[args.task],
        "task": args.task,
        "protocol": rows[0]["protocol"],
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(r["id"] for r in rows).encode()
        ).hexdigest(),
        "selected_rows": len(rows),
        "limit_clusters_per_stratum": args.limit_clusters_per_stratum,
    }
    if args.task.startswith("warbias_"):
        run["scoring_policy"] = args.answer_policy
    if args.command == "run":
        from lm_eval.api.registry import get_model
        from lm_eval.fair_uk.provenance import model_identity
        from lm_eval.fair_uk.runner import run_predictions

        run.update(
            model_backend=args.model,
            batch_size=str(args.batch_size),
            model_identity=model_identity(args.model_args),
            seed=args.seed,
            harness_version=importlib.metadata.version("lm_eval"),
            runtime_versions={
                name: importlib.metadata.version(name)
                for name in ("numpy", "transformers", "torch")
            },
        )
        manifest = args.output / "run.json"
        if manifest.exists() and json.loads(manifest.read_text()) != run:
            raise ValueError(
                "Output belongs to a different run; use a new output directory"
            )
        if not manifest.exists() and (args.output / "predictions.jsonl").exists():
            raise ValueError("Checkpoint exists without a run manifest")
        write_json(manifest, run)
        import random

        import numpy as np
        import torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        backend = get_model(args.model).create_from_arg_string(
            args.model_args, {"batch_size": args.batch_size}
        )
        predictions = run_predictions(
            backend, rows, args.output / "predictions.jsonl", args.chunk_size
        )
    elif args.command == "run-openai":
        from lm_eval.fair_uk.openai_runner import run as run_openai

        predictions, run = run_openai(
            rows,
            run,
            args.snapshot,
            args.seed,
            args.output,
            args.concurrency,
            args.max_estimated_usd,
        )
    elif args.command == "run-gemini":
        from lm_eval.fair_uk.gemini_runner import run as run_gemini

        predictions, run = run_gemini(
            rows,
            run,
            args.snapshot,
            args.seed,
            args.output,
            args.concurrency,
            args.max_estimated_usd,
        )
    else:
        predictions = [
            json.loads(line) for line in args.predictions.read_text().splitlines()
        ]
        run.update(
            predictions_sha256=hashlib.sha256(
                args.predictions.read_bytes()
            ).hexdigest(),
            model_identity="external_predictions_not_verified",
        )
    scored = score_all(rows, predictions, args.answer_policy)
    report = make_report(scored, registry_rows, args.bootstrap, args.seed)
    report["provenance"] = run
    if args.command == "run-openai":
        from lm_eval.fair_uk.openai_runner import usage_summary

        report["api_usage"] = usage_summary(predictions, args.snapshot)
    report.update(
        tool="Fair-UK",
        version=VERSION,
        report_schema_version=2,
        scorer_code_identity=code_identity(),
    )
    if args.command == "run-gemini":
        from lm_eval.fair_uk.gemini_runner import usage_summary

        report["api_usage"] = usage_summary(predictions, args.snapshot)
    write_json(args.output / "report.json", report)
    (args.output / "records.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in scored
        )
    )
    write_csv(args.output / "groups.csv", list(group_rows(report)))
    (args.output / "report.md").write_text(markdown(report))
    print(f"Report written to {args.output.resolve() / 'report.md'}")


if __name__ == "__main__":
    main()
