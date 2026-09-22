"""Prepare or run a bias benchmark suite against one pinned model checkpoint."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from lm_eval.fair_uk.answers import NORMALIZED, POLICIES
from lm_eval.fair_uk.data import PROTOCOLS, REGISTRY, family


DEFAULT_TASKS = (
    "warbias_uk",
    "warbias_intersectional_uk",
    "bbq_uk",
    "stereoset_uk",
    "winobias_uk_natural",
    "winobias_uk_controlled",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", choices=sorted(REGISTRY), default=DEFAULT_TASKS
    )
    parser.add_argument("--model", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--model-args")
    parser.add_argument("--output", type=Path, default=Path("results/fair-uk"))
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answer-policy", choices=POLICIES, default=NORMALIZED)
    parser.add_argument("--limit-clusters-per-stratum", type=int)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run inference; without this flag only print the local registry plan",
    )
    args = parser.parse_args(argv)
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("Select each task only once")
    if args.bootstrap < 0 or args.chunk_size < 1:
        parser.error("Bootstrap must be nonnegative and chunk size positive")
    if (
        args.limit_clusters_per_stratum is not None
        and args.limit_clusters_per_stratum < 1
    ):
        parser.error("Cluster limit must be positive")
    if args.execute and not args.model_args:
        parser.error("--execute requires --model-args with a pinned checkpoint")
    plan = {
        "status": "execution_requested" if args.execute else "prepared_not_running",
        "backend": args.model,
        "model_arguments_configured": bool(args.model_args),
        "selection": "full datasets"
        if args.limit_clusters_per_stratum is None
        else "complete source-case subset",
        "limit_clusters_per_stratum": args.limit_clusters_per_stratum,
        "tasks": [
            {
                "task": task,
                "dataset": REGISTRY[task],
                "protocol": PROTOCOLS[family(task)],
                "scoring_policy": args.answer_policy
                if family(task) == "warbias"
                else None,
                "output": str(args.output / task),
            }
            for task in args.tasks
        ],
        "note": "Registry planning does not download data, load a model, verify model access, or call an API. Each execution validates data/model fingerprints. vLLM GPU execution is not validated in this release.",
    }
    # Model arguments can contain credentials; never echo them in a plan/error.
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    from lm_eval.fair_uk.provenance import model_identity

    model_identity(args.model_args)  # Reject unpinned models before starting any task.
    reports = []
    for task in args.tasks:
        command = [
            sys.executable,
            "-m",
            "lm_eval.fair_uk",
            "run",
            "--task",
            task,
            "--model",
            args.model,
            "--model-args",
            args.model_args,
            "--output",
            str(args.output / task),
            "--batch-size",
            args.batch_size,
            "--chunk-size",
            str(args.chunk_size),
            "--bootstrap",
            str(args.bootstrap),
            "--seed",
            str(args.seed),
            "--answer-policy",
            args.answer_policy,
        ]
        if args.limit_clusters_per_stratum is not None:
            command += [
                "--limit-clusters-per-stratum",
                str(args.limit_clusters_per_stratum),
            ]
        # Separate processes release model memory between benchmark tasks.
        result = subprocess.run(command, check=False)  # noqa: S603 -- fixed interpreter; argument list, no shell
        if result.returncode:
            raise SystemExit(
                f"{task} failed (exit {result.returncode}); saved checkpoints remain. Later tasks were not started."
            )
        reports.append(str(args.output / task / "report.json"))
    result = subprocess.run(  # noqa: S603 -- fixed interpreter; argument list, no shell
        [
            sys.executable,
            "-m",
            "lm_eval.fair_uk",
            "summarize",
            "--reports",
            *reports,
            "--output",
            str(args.output / "summary"),
        ],
        check=False,
    )
    if result.returncode:
        raise SystemExit("Summary generation failed; per-task reports remain available")


if __name__ == "__main__":
    main()
