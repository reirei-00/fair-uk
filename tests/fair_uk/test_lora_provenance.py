"""Separate adapter weights remain part of checkpoint identity."""

import copy
import json

import pytest

from lm_eval.fair_uk.provenance import model_identity, paired_inference_settings


def adapter(tmp_path):
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    (path / "adapter_model.safetensors").write_bytes(b"synthetic weights")
    return path


def test_adapter_content_and_configuration_are_fingerprinted(tmp_path):
    path = adapter(tmp_path)
    args = f"pretrained=fixture/base,revision={'a' * 40},lora_local_path={path}"
    first = model_identity(args)
    assert first["repository"] == "fixture/base"
    assert set(first["adapter"]["local_files_sha256"]) == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    (path / "adapter_model.safetensors").write_bytes(b"changed weights")
    assert model_identity(args) != first
    (path / "adapter_config.json").write_text('{"peft_type": "OTHER"}')
    with pytest.raises(ValueError, match="LoRA adapter"):
        model_identity(args)
    (path / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="adapter_model"):
        model_identity(args)


def test_paired_settings_allow_adapter_weights_but_reject_missing_identity(tmp_path):
    path = adapter(tmp_path)
    args = f"pretrained=fixture/base,revision={'a' * 40},dtype=bfloat16"
    before = {
        "model_backend": "vllm",
        "model_identity": model_identity(args),
        "evaluation_asset_identity": {
            "tokenizer_sha256": "b" * 64,
            "input_sha256": "c" * 64,
        },
    }
    after = copy.deepcopy(before)
    after["model_identity"] = model_identity(args + f",lora_local_path={path}")
    assert paired_inference_settings(before) == paired_inference_settings(after)
    after["model_identity"]["parameters"]["dtype"] = "float16"
    assert paired_inference_settings(before) != paired_inference_settings(after)
    del after["model_identity"]["adapter"]
    with pytest.raises(ValueError, match="adapter hashes"):
        paired_inference_settings(after)


def test_warbias_rejects_adapter_mismatch(tmp_path):
    from test_warbias_baseline import _saved_runs, release as release_fixture

    from lm_eval.fair_uk.warbias.baseline_inputs import verify_baseline_run

    release = release_fixture.__wrapped__(tmp_path / "release")
    results = tmp_path / "results"
    _saved_runs(release, results)
    task = "warbias_uk"
    directory = results / "fixture" / "full" / task
    run_path = directory / "run.json"
    run = json.loads(run_path.read_text())
    expected = copy.deepcopy(run["model_identity"])
    run["model_identity"]["adapter"] = {
        "type": "lora",
        "local_files_sha256": {"weights": "d" * 64},
    }
    run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="adapter files"):
        verify_baseline_run(
            directory,
            task,
            release / "inputs/manifest.json",
            expected["parameters"],
            expected,
        )
