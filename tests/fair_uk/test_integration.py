# ruff: noqa: S106
# Tokenizer vocabulary symbols below are not authentication credentials.
import hashlib
import json

import pytest

from lm_eval.fair_uk.__main__ import main
from lm_eval.fair_uk.data import REGISTRY, adapt, load, select_clusters
from lm_eval.fair_uk.provenance import model_identity
from lm_eval.fair_uk.runner import predict


def qa_fixture(path):
    rows = []
    for case in range(2):
        for condition in ("ambiguous", "stereotype_aligned", "stereotype_conflicting"):
            for polarity in ("negative", "positive"):
                stereotype = 0 if polarity == "negative" else 1
                rows.append(
                    {
                        "id": f"{case}/{condition}/{polarity}",
                        "scenario_id": str(case),
                        "source_case_id": str(case),
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
                        "context": "two people",
                        "question": "who?",
                    }
                )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {
        **REGISTRY["warbias_uk"],
        "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_registered_checksum_and_complete_cluster_limits(tmp_path, monkeypatch):
    path = tmp_path / "fixture.jsonl"
    monkeypatch.setitem(REGISTRY, "warbias_uk", qa_fixture(path))
    rows = load("warbias_uk", path)
    assert len(select_clusters(rows, 1)) == 6
    with pytest.raises(ValueError):
        select_clusters(rows, 0)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load("warbias_uk", path)


def test_model_fingerprints_and_secret_redaction(tmp_path):
    with pytest.raises(ValueError, match="Pin a remote model"):
        model_identity("pretrained=organization/model")
    identity = model_identity(
        "pretrained=organization/model,revision=" + "a" * 40 + ",token=SECRET"
    )
    assert "SECRET" not in json.dumps(identity)
    file = tmp_path / "model.bin"
    file.write_bytes(b"first")
    first = model_identity(f"pretrained={tmp_path}")
    file.write_bytes(b"second")
    assert first != model_identity(f"pretrained={tmp_path}")


def tiny_model(path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    torch.manual_seed(23)
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"[EOS]": 0, "[UNK]": 1, "A": 2, "B": 3, "C": 4}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[EOS]",
        eos_token="[EOS]",
        pad_token="[EOS]",
        unk_token="[UNK]",
    )
    fast.save_pretrained(path)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=5,
            n_positions=512,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0,
        )
    )
    model.save_pretrained(path)


def test_real_harness_generation_resume_and_offline_rescore(tmp_path, monkeypatch):
    source = tmp_path / "fixture.jsonl"
    monkeypatch.setitem(REGISTRY, "warbias_uk", qa_fixture(source))
    model = tmp_path / "model"
    tiny_model(model)
    output = tmp_path / "run"
    args = [
        "run",
        "--task",
        "warbias_uk",
        "--input",
        str(source),
        "--model",
        "hf",
        "--model-args",
        f"pretrained={model},device=cpu",
        "--batch-size",
        "4",
        "--output",
        str(output),
        "--bootstrap",
        "10",
    ]
    main(args)
    before = (output / "predictions.jsonl").read_bytes()
    main(args)
    assert (output / "predictions.jsonl").read_bytes() == before
    rescored = tmp_path / "rescored"
    main(
        [
            "score",
            "--task",
            "warbias_uk",
            "--input",
            str(source),
            "--predictions",
            str(output / "predictions.jsonl"),
            "--output",
            str(rescored),
            "--bootstrap",
            "10",
        ]
    )
    run_report = json.loads((output / "report.json").read_text())
    score_report = json.loads((rescored / "report.json").read_text())
    assert run_report["comparisons"] == score_report["comparisons"]
    assert run_report["source_cases"] == 2
    source_en = tmp_path / "fixture_en.jsonl"
    source_en.write_text(
        source.read_text().replace('"language": "uk"', '"language": "en"')
    )
    monkeypatch.setitem(
        REGISTRY,
        "warbias_en",
        {
            **REGISTRY["warbias_uk"],
            "language": "en",
            "sha256": hashlib.sha256(source_en.read_bytes()).hexdigest(),
        },
    )
    output_en = tmp_path / "run_en"
    args_en = [
        str(output_en)
        if a == str(output)
        else str(source_en)
        if a == str(source)
        else "warbias_en"
        if a == "warbias_uk"
        else a
        for a in args
    ]
    main(args_en)
    paired = tmp_path / "paired"
    compare_args = [
        "compare",
        "--uk-run",
        str(output),
        "--en-run",
        str(output_en),
        "--uk-input",
        str(source),
        "--en-input",
        str(source_en),
        "--output",
        str(paired),
        "--bootstrap",
        "100",
    ]
    main(compare_args)
    result = json.loads((paired / "comparison.json").read_text())
    assert result["paired_rows"] == 12 and result["source_cases"] == 2
    assert result["choice_disagreement_rate"] == 0
    assert all(c["worst_delta_ci95"] == [0, 0] for c in result["comparisons"])
    summary = tmp_path / "summary"
    main(
        [
            "summarize",
            "--reports",
            str(output / "report.json"),
            str(output_en / "report.json"),
            "--output",
            str(summary),
        ]
    )
    assert "warbias_uk" in (summary / "summary.md").read_text()
    assert "warbias_en" in (summary / "coverage.csv").read_text()
    assert (output / "groups.csv").exists()
    with pytest.raises(ValueError, match="Duplicate experiment"):
        main(
            [
                "summarize",
                "--reports",
                str(output / "report.json"),
                str(output / "report.json"),
                "--output",
                str(summary),
            ]
        )
    # Re-score raw predictions instead of trusting possibly stale computed records.
    (output_en / "records.jsonl").write_text("invalid cached derived results")
    main(compare_args)
    run_en = json.loads((output_en / "run.json").read_text())
    run_en["model_identity"]["local_files_sha256"]["model.safetensors"] = "different"
    (output_en / "run.json").write_text(json.dumps(run_en))
    with pytest.raises(ValueError, match="identical model_identity"):
        main(compare_args)
    (model / "modified.txt").write_text("changed")
    with pytest.raises(ValueError, match="different run"):
        main(args)


def test_real_harness_sentence_likelihood(tmp_path):
    tiny_model(tmp_path)
    from lm_eval.models.huggingface import HFLM

    model = HFLM(pretrained=str(tmp_path), device="cpu", batch_size=2)
    row = adapt(
        "stereoset_uk",
        {
            "item_id": "s",
            "target_uk": "target",
            "bias_type_uk": "category",
            "template_uk": "A [BLANK]",
            "stereotype_fill_uk": "A",
            "anti_stereotype_fill_uk": "B",
            "unrelated_fill_uk": "C",
        },
    )
    result = predict(model, [row])[0]
    assert len(result["scores"]) == 3 and result["token_counts"] == [2, 2, 2]
    import torch

    ids = torch.tensor([[0, 2, 2]])
    with torch.no_grad():
        logits = model.model(ids).logits[0, :-1].float().log_softmax(-1)
    expected = (logits[0, 2] + logits[1, 2]).item() / 2
    assert result["scores"][0] == pytest.approx(expected, abs=1e-6)
