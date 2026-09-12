"""Run model evaluations or rescore saved predictions without model inference."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from lm_eval.fairforget import VERSION
from lm_eval.fairforget.data import REGISTRY, load, select_clusters
from lm_eval.fairforget.metrics import make_report, score_all


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def markdown(report):
    lines = [
        f"# {report['task']}",
        "",
        f"Protocol: `{report['protocol']}`. Dataset status: **pilot, human-unvalidated**.",
        "",
        f"{report['rows']} rows; {report['source_cases']} source cases. Partial run: {report['partial_run']}.",
        "",
        "Rates below use equal source-case weights within each group. Native metrics retain their documented aggregation.",
        "",
        "| Condition / stratum / control | Metric | Worst available group rate | Group(s) | Gap | Min/max ratio | Missing groups |",
        "| --- | --- | ---: | --- | ---: | ---: | --- |",
    ]

    def fmt(value):
        return "undefined" if value is None else f"{value:.4f}"

    for comparison in report["comparisons"]:
        scope = " / ".join(
            str(comparison[k])
            for k in ("condition", "stratum", "score_group")
            if comparison[k] is not None
        )
        groups = ", ".join(comparison["worst_groups"]).replace("|", " / ")
        lines.append(
            f"| {scope} | {comparison['metric']} | {fmt(comparison['worst'])} | {groups} | {fmt(comparison['gap'])} | {fmt(comparison['minmax_ratio'])} | {', '.join(comparison['missing_groups']).replace('|', ' / ')} |"
        )
    lines.extend(
        [
            "",
            "Full group numerators, denominators, source-case counts, native scores and bootstrap intervals are in `report.json`. Per-item scores are in `records.jsonl`.",
            "",
            "A parity ratio of one does not imply low harm. Zero/zero ratios are undefined. Intervals are exploratory source-case bootstrap intervals, not simultaneous coverage guarantees. Repeated profiles and translations are not independent source cases.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List pinned benchmark tasks")
    for command in ("validate", "run", "score"):
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
        if command == "run":
            sub.add_argument("--model", default="hf", help="Harness model backend")
            sub.add_argument("--model-args", required=True)
            sub.add_argument("--batch-size", default="auto")
            sub.add_argument("--chunk-size", type=int, default=32)
        if command == "score":
            sub.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(REGISTRY, indent=2))
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
    args.output.mkdir(parents=True, exist_ok=True)
    run = {
        "version": VERSION,
        "dataset": REGISTRY[args.task],
        "task": args.task,
        "protocol": rows[0]["protocol"],
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(r["id"] for r in rows).encode()
        ).hexdigest(),
        "selected_rows": len(rows),
        "limit_clusters_per_stratum": args.limit_clusters_per_stratum,
    }
    if args.command == "run":
        from lm_eval.api.registry import get_model
        from lm_eval.fairforget.provenance import model_identity
        from lm_eval.fairforget.runner import run_predictions

        run.update(
            model_backend=args.model,
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
    scored = score_all(rows, predictions)
    report = make_report(scored, registry_rows, args.bootstrap, args.seed)
    report["provenance"] = run
    write_json(args.output / "report.json", report)
    (args.output / "records.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in scored
        )
    )
    (args.output / "report.md").write_text(markdown(report))
    print(f"Report written to {args.output.resolve() / 'report.md'}")


if __name__ == "__main__":
    main()
