"""Portable checkpoint views preserve every actual training file identity."""

import copy
import json

import pytest

from lm_eval.fair_uk import all_metrics, checkpoint_views as views


@pytest.fixture
def view_case(tmp_path):
    repository = "google/gemma-3-12b-it"
    base = "/pinned/base"
    original = {
        "config.json": "a" * 64,
        "tokenizer.json": "b" * 64,
        "model.safetensors": "c" * 64,
    }
    assets = {
        name: {"source_path": base + "/" + name, "sha256": digit * 64}
        for name, digit in (
            ("preprocessor_config.json", "d"),
            ("processor_config.json", "e"),
        )
    }
    observed = {**original, **{name: item["sha256"] for name, item in assets.items()}}
    receipt = {
        "method": "wga_kl",
        "status": "trained_merged",
        "model_id": repository,
        "model_revision": "1" * 40,
        "tokenizer_revision": "2" * 40,
        "merged_checkpoint": "/original/wga",
        "merged_checkpoint_files": [
            {"path": name, "sha256": sha} for name, sha in original.items()
        ],
        "source_model_files": [
            {"path": name, "sha256": item["sha256"]} for name, item in assets.items()
        ],
    }
    path = tmp_path / "copied-receipt.json"
    path.write_text(json.dumps(receipt))
    descriptor = {
        "kind": "add_pinned_processor_assets",
        "original_receipt": {
            "path": "/original/intervention.json",
            "sha256": views.digest(path),
        },
        "original_checkpoint": "/original/wga",
        "original_files_sha256": original,
        "evaluation_checkpoint": "/derived/wga",
        "evaluation_files_sha256": observed,
        "added_assets": assets,
    }
    model = {
        "base_path": base,
        "model_revision": "1" * 40,
        "tokenizer_revision": "2" * 40,
        "base_config_sha256": "f" * 64,
        "methods": {method: copy.deepcopy(descriptor) for method in views.METHODS},
    }
    document = {
        "schema_version": 1,
        "kind": "fair_uk_checkpoint_inference_views",
        "models": {repository: model},
    }
    sidecar = tmp_path / "views.json"
    sidecar.write_text(json.dumps(document))
    kwargs = {
        "sidecar_sha256": views.digest(sidecar),
        "model_id": repository,
        "method": "wga_kl",
        "receipt_path": path,
        "receipt_sha256": views.digest(path),
        "original_checkpoint": "/original/wga",
        "original_files_sha256": original,
        "evaluation_checkpoint": "/derived/wga",
        "evaluation_files_sha256": observed,
        "model_revision": "1" * 40,
        "tokenizer_revision": "2" * 40,
        "base_path": base,
        "base_config_sha256": "f" * 64,
    }
    return sidecar, document, kwargs


def test_portable_copy_needs_no_original_checkpoint_or_source_files(view_case):
    path, _, kwargs = view_case
    result = views.validate_inference_view(path, **kwargs)
    assert result["original_receipt"]["path"] == "/original/intervention.json"
    assert result["original_weight_config_tokenizer_files_unchanged"] is True
    assert result["evaluation_files_sha256"] == kwargs["evaluation_files_sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        "weights",
        "config",
        "tokenizer",
        "extra",
        "removed",
        "source",
        "asset_sha",
        "receipt_sha",
        "wrong_path",
        "pin",
        "third_method",
    ],
)
def test_changed_or_unpinned_views_fail(view_case, mutation):
    path, data, kwargs = view_case
    model = data["models"][kwargs["model_id"]]
    view = model["methods"]["wga_kl"]
    if mutation in ("weights", "config", "tokenizer"):
        name = {
            "weights": "model.safetensors",
            "config": "config.json",
            "tokenizer": "tokenizer.json",
        }[mutation]
        view["evaluation_files_sha256"][name] = "0" * 64
        kwargs["evaluation_files_sha256"] = dict(view["evaluation_files_sha256"])
    elif mutation == "extra":
        view["evaluation_files_sha256"]["unexpected.json"] = "0" * 64
        kwargs["evaluation_files_sha256"] = dict(view["evaluation_files_sha256"])
    elif mutation == "removed":
        view["evaluation_files_sha256"].pop("model.safetensors")
        kwargs["evaluation_files_sha256"] = dict(view["evaluation_files_sha256"])
    elif mutation == "source":
        view["added_assets"]["processor_config.json"]["source_path"] = (
            "/other/processor_config.json"
        )
    elif mutation == "asset_sha":
        view["added_assets"]["processor_config.json"]["sha256"] = "0" * 64
        view["evaluation_files_sha256"]["processor_config.json"] = "0" * 64
        kwargs["evaluation_files_sha256"] = dict(view["evaluation_files_sha256"])
    elif mutation == "receipt_sha":
        view["original_receipt"]["sha256"] = "0" * 64
    elif mutation == "wrong_path":
        kwargs["evaluation_checkpoint"] = "/other/model"
    elif mutation == "pin":
        model["tokenizer_revision"] = "0" * 40
    else:
        model["methods"].pop("task_vector")
    path.write_text(json.dumps(data))
    kwargs["sidecar_sha256"] = views.digest(path)
    with pytest.raises(ValueError):
        views.validate_inference_view(path, **kwargs)


def test_original_kind_cannot_add_assets_or_change_path(view_case):
    path, data, kwargs = view_case
    descriptor = data["models"][kwargs["model_id"]]["methods"]["wga_kl"]
    descriptor.update(
        kind="original",
        added_assets={},
        evaluation_checkpoint="/original/wga",
        evaluation_files_sha256=kwargs["original_files_sha256"],
    )
    kwargs.update(
        evaluation_checkpoint="/original/wga",
        evaluation_files_sha256=kwargs["original_files_sha256"],
    )
    path.write_text(json.dumps(data))
    kwargs["sidecar_sha256"] = views.digest(path)
    assert views.validate_inference_view(path, **kwargs)["kind"] == "original"
    kwargs["evaluation_checkpoint"] = "/derived/wga"
    with pytest.raises(ValueError, match="Native checkpoint path"):
        views.validate_inference_view(path, **kwargs)


def test_certificate_hash_and_duplicate_keys_are_checked(view_case):
    path, _, _kwargs = view_case
    with pytest.raises(ValueError, match="certificate hash"):
        views.load_inference_views(path, "0" * 64)
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="Duplicate"):
        views.load_inference_views(path, views.digest(path))


def test_rmu_processor_assets_must_match_observed_pinned_base_when_available(view_case):
    path, data, kwargs = view_case
    receipt = json.loads(kwargs["receipt_path"].read_text())
    receipt.update(
        method="rmu",
        status="trained_not_evaluated",
        checkpoint=receipt.pop("merged_checkpoint"),
        checkpoint_identity=receipt.pop("merged_checkpoint_files"),
    )
    receipt["config"] = {
        name: receipt[name]
        for name in ("model_id", "model_revision", "tokenizer_revision")
    }
    receipt.pop("source_model_files")
    kwargs["receipt_path"].write_text(json.dumps(receipt))
    descriptor = data["models"][kwargs["model_id"]]["methods"]["rmu"]
    descriptor["original_receipt"]["sha256"] = views.digest(kwargs["receipt_path"])
    path.write_text(json.dumps(data))
    kwargs.update(
        method="rmu",
        receipt_sha256=views.digest(kwargs["receipt_path"]),
        sidecar_sha256=views.digest(path),
        base_files_sha256={
            "preprocessor_config.json": "d" * 64,
            "processor_config.json": "e" * 64,
        },
    )
    assert (
        views.validate_inference_view(path, **kwargs)["processor_asset_base_binding"]
        == "measured_native_base_files"
    )
    kwargs["base_files_sha256"]["processor_config.json"] = "0" * 64
    with pytest.raises(ValueError, match="measured pinned base files"):
        views.validate_inference_view(path, **kwargs)


def test_all_metrics_binds_explicit_view_and_default_remains_strict(
    view_case, tmp_path
):
    path, _, kwargs = view_case
    recipe = {
        "method": "wga_kl",
        "model_id": kwargs["model_id"],
        "model_revision": kwargs["model_revision"],
        "tokenizer_revision": kwargs["tokenizer_revision"],
        "model_path": kwargs["base_path"],
    }
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    receipt = json.loads(kwargs["receipt_path"].read_text())
    receipt["source_model_files"].extend(
        [
            {"path": "config.json", "sha256": "f" * 64},
            {"path": "model.safetensors", "sha256": "9" * 64},
        ]
    )
    kwargs["receipt_path"].write_text(json.dumps(receipt))
    data = json.loads(path.read_text())
    data["models"][kwargs["model_id"]]["methods"]["wga_kl"]["original_receipt"][
        "sha256"
    ] = views.digest(kwargs["receipt_path"])
    path.write_text(json.dumps(data))
    execution = tmp_path / "execution.json"
    execution.write_text(
        json.dumps({"methods": {"wga_kl": {"checkpoint": "/original/wga"}}})
    )
    directory = tmp_path / "wga_kl"
    directory.mkdir()
    (directory / "intervention.json").write_bytes(kwargs["receipt_path"].read_bytes())
    exposure = {
        "execution": str(execution),
        "recipe": str(recipe_path),
        "method_directory": str(directory),
        "inference_views": {"path": str(path), "sha256": views.digest(path)},
    }
    pair = {
        "provenance": {
            "before": {
                "run": {
                    "model_identity": {
                        "parameters": {"pretrained": kwargs["base_path"]},
                        "local_files_sha256": {
                            "config.json": "f" * 64,
                            "model.safetensors": "9" * 64,
                            "preprocessor_config.json": "d" * 64,
                            "processor_config.json": "e" * 64,
                        },
                    }
                }
            },
            "after": {
                "run": {
                    "model_identity": {
                        "parameters": {"pretrained": kwargs["evaluation_checkpoint"]},
                        "local_files_sha256": kwargs["evaluation_files_sha256"],
                    }
                }
            },
        }
    }
    result = all_metrics.bind_training_pair(
        pair, exposure, {"method": "wga_kl"}, tmp_path
    )
    assert result["inference_view"]["kind"] == "add_pinned_processor_assets"
    exposure.pop("inference_views")
    with pytest.raises(ValueError, match="After checkpoint path"):
        all_metrics.bind_training_pair(pair, exposure, {"method": "wga_kl"}, tmp_path)
