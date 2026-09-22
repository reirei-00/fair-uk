import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest

from lm_eval.fair_uk.__main__ import main
from lm_eval.fair_uk.data import REGISTRY, load


def source_fixture(path):
    records = []
    # Deliberately use a source order that differs from lexical ID order.
    for case in ("z", "a"):
        for condition in ("ambiguous", "stereotype_aligned", "stereotype_conflicting"):
            for polarity in ("negative", "positive"):
                stereotype = 0 if polarity == "negative" else 1
                records.append(
                    {
                        "id": f"{case}/{condition}/{polarity}",
                        "scenario_id": case,
                        "source_case_id": case,
                        "target_group": "idp",
                        "language": "uk",
                        "evidence_condition": condition,
                        "question_polarity": polarity,
                        "answer_index": 2
                        if condition == "ambiguous"
                        else stereotype
                        if condition == "stereotype_aligned"
                        else 1 - stereotype,
                        "stereotype_consistent_index": stereotype,
                        "unknown_index": 2,
                        "choices": ["one", "two", "unknown"],
                        "context": "Дві людини.",
                        "question": "Хто?",
                    }
                )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return {
        **REGISTRY["warbias_uk"],
        "rows": len(records),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class MockClient:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **payload):
        self.calls.append(payload)
        raw = {
            "model": self.snapshot,
            "system_fingerprint": "fp-offline",
            "choices": [{"finish_reason": "stop", "message": {"content": "C"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }
        return SimpleNamespace(model_dump=lambda **_: raw, _request_id="offline")

    def post(self, url, *, json):
        self.calls.append((url, json))
        raw = {
            "modelVersion": self.snapshot,
            "candidates": [
                {"content": {"parts": [{"text": "C"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1},
        }
        return SimpleNamespace(status_code=200, is_success=True, json=lambda: raw)


@pytest.mark.parametrize(
    ("provider", "snapshot"),
    [("openai", "gpt-4.1-mini-2025-04-14"), ("gemini", "gemini-3.6-flash")],
)
@pytest.mark.parametrize("checkpoint", ["fresh", "partial", "complete"])
def test_legacy_cli_hashes_saved_source_order(
    tmp_path, monkeypatch, provider, snapshot, checkpoint
):
    runner = importlib.import_module(f"lm_eval.fair_uk.{provider}_runner")
    source = tmp_path / "source.jsonl"
    monkeypatch.setitem(REGISTRY, "warbias_uk", source_fixture(source))
    rows = load("warbias_uk", source)
    order = {row["id"]: index for index, row in enumerate(rows)}
    client = MockClient(snapshot)
    original_run, original_write = runner.run, runner.atomic_write
    returned, saved_orders = [], []

    def mocked_run(*args, **kwargs):
        predictions, manifest = original_run(*args, **kwargs, client=client)
        returned.append(predictions)
        return predictions, manifest

    def track_write(path, predictions):
        saved_orders.append([order[prediction["id"]] for prediction in predictions])
        original_write(path, predictions)

    monkeypatch.setattr(runner, "run", mocked_run)
    monkeypatch.setattr(runner, "atomic_write", track_write)
    # Force the consumer to observe the last submitted future first without
    # relying on sleeps or scheduler timing, while exercising the real executor.
    monkeypatch.setattr(runner, "as_completed", lambda futures: reversed(futures))
    output = tmp_path / "run"
    arguments = [
        f"run-{provider}",
        "--task",
        "warbias_uk",
        "--snapshot",
        snapshot,
        "--input",
        str(source),
        "--output",
        str(output),
        "--concurrency",
        "4",
        "--bootstrap",
        "0",
    ]
    preserved = []
    if checkpoint != "fresh":
        main(arguments)
        full = returned[-1]
        # Simulate exports saved by the old runners, which used completion order.
        preserved = (
            [full[-1], full[0], full[-2]]
            if checkpoint == "partial"
            else list(reversed(full))
        )
        original_write(output / "predictions.jsonl", preserved)
        client.calls.clear()
        saved_orders.clear()

    main(arguments)

    saved = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text().splitlines()
    ]
    assert saved == returned[-1]
    assert [prediction["id"] for prediction in saved] == [row["id"] for row in rows]
    assert saved_orders and all(order == sorted(order) for order in saved_orders)
    assert len(client.calls) == len(rows) - len(preserved)
    by_id = {prediction["id"]: prediction for prediction in saved}
    assert all(by_id[prediction["id"]] == prediction for prediction in preserved)
    canonical_saved = "".join(
        json.dumps(prediction, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
        for prediction in saved
    )
    report = json.loads((output / "report.json").read_text())
    assert (
        report["predictions_sha256"]
        == hashlib.sha256(canonical_saved.encode()).hexdigest()
    )
