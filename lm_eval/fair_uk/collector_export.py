"""Export a saved pair using its immutable collector's validation code.

Execute this file with the collector Python and only the frozen evaluator on
PYTHONPATH. The analysis package is deliberately not imported by this process.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text())

    from lm_eval.fair_uk import runs
    from lm_eval.fair_uk.answers import NORMALIZED
    from lm_eval.fair_uk.data import family
    from lm_eval.fair_uk.paired import _same_sources
    from lm_eval.fair_uk.provenance import code_identity

    root = Path(request["evaluator_root"]).resolve()
    if root not in Path(runs.__file__).resolve().parents:
        raise ValueError("Validation did not import the selected frozen collector")
    collector_identity = code_identity()
    phase = request.get("phase", "evaluation")
    if phase not in ("evaluation", "development", "train_signal"):
        raise ValueError("Unsupported explicitly selected study phase")
    expected_split = "train" if phase == "train_signal" else phase
    loaded, provenance = [], {}
    for phase in ("before", "after"):
        directory = Path(request[phase + "_run"])
        saved = runs.load_verified_run(
            directory,
            dataset_bundle=Path(request["dataset_bundle"]),
            judgments_path=request.get(phase + "_judgments"),
        )
        manifest = saved.provenance["run"]
        if manifest.get("code_identity") != collector_identity:
            raise ValueError(
                "Saved collector code differs from selected validation source"
            )
        if manifest["task"] != request["task"]:
            raise ValueError("Saved task differs from the requested pair")
        if manifest["dataset"].get("measurement_eligibility", "active") != "active":
            raise ValueError("Source policy blocks this preparation-only task")
        if manifest.get("limit_clusters_per_stratum") is not None:
            raise ValueError("Smoke subsets cannot stand in for complete measurements")
        if len(saved.selected_rows) != len(saved.source_rows):
            raise ValueError("Incomplete native source cohort")
        if {row.get("split") for row in saved.source_rows} != {expected_split}:
            raise ValueError(
                "Source rows do not match the explicitly requested study phase"
            )
        provenance[phase] = {
            **saved.provenance,
            "files_sha256": {
                name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in ("run.json", "report.json", "predictions.jsonl")
            },
        }
        loaded.append(saved)
    _same_sources(*loaded)
    before, after = loaded
    result = {
        "schema_version": 1,
        "kind": "collector_validated_saved_pair",
        "task": request["task"],
        "collector_code_identity": collector_identity,
        "answer_policy": before.provenance["run"].get("scoring_policy", NORMALIZED)
        if family(request["task"]) == "warbias"
        else NORMALIZED,
        "rows": before.selected_rows,
        "before_predictions": before.predictions,
        "after_predictions": after.predictions,
        "before_scored": before.scored_rows,
        "after_scored": after.scored_rows,
        "provenance": provenance,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
