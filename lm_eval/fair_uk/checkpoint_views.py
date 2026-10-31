"""Validate strictly additive inference checkpoint views without loading models.

The original trainer receipt remains authoritative. A separately hash-bound
certificate may add only the two pinned Gemma processor JSON assets. Native
measurements supply the complete observed file map, so portable verification
does not require original checkpoint weights or their original filesystem.
"""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath


METHODS = frozenset(("rmu", "task_vector", "wga_kl"))
PROCESSOR_ASSETS = frozenset(("preprocessor_config.json", "processor_config.json"))
GEMMA_MODELS = frozenset(
    (
        "google/gemma-3-12b-it",
        "lapa-llm/lapa-v0.1.2-instruct",
        "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0",
        "INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0",
    )
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "Duplicate inference-view JSON key")
        value[key] = item
    return value


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _files(value):
    require(isinstance(value, dict) and bool(value), "Inference-view file map is empty")
    for name, checksum in value.items():
        require(
            isinstance(name, str)
            and bool(name)
            and "\\" not in name
            and not PurePosixPath(name).is_absolute()
            and ".." not in PurePosixPath(name).parts
            and str(PurePosixPath(name)) == name
            and _hash(checksum),
            "Unsafe inference-view file identity",
        )
    return dict(value)


def load_inference_views(path, expected_sha256):
    payload = Path(path).read_bytes()
    require(
        _hash(expected_sha256)
        and hashlib.sha256(payload).hexdigest() == expected_sha256,
        "Inference-view certificate hash differs",
    )
    value = json.loads(payload, object_pairs_hook=_object)
    require(
        value.get("schema_version") == 1
        and value.get("kind") == "fair_uk_checkpoint_inference_views",
        "Unsupported inference-view certificate",
    )
    require(
        isinstance(value.get("models"), dict) and bool(value["models"]),
        "Inference-view model registry is empty",
    )
    for model in value["models"].values():
        require(
            isinstance(model, dict) and set(model.get("methods", {})) == METHODS,
            "Inference-view registry requires exactly three methods",
        )
    return value


def validate_inference_view(
    sidecar_path,
    *,
    sidecar_sha256,
    model_id,
    method,
    receipt_path,
    receipt_sha256,
    original_checkpoint,
    original_files_sha256,
    evaluation_checkpoint,
    evaluation_files_sha256,
    model_revision,
    tokenizer_revision,
    base_path,
    base_config_sha256,
    base_files_sha256=None,
):
    """Bind one measured view to unchanged original bytes and its real receipt.

    ``receipt_path`` may be a byte-identical portable copy. The certificate's
    original receipt/checkpoint/source paths remain metadata and are not rebased.
    All model and file identities are supplied by the caller's independently
    verified recipe, receipt and native measurement, not inferred from a view.
    """
    sidecar = load_inference_views(sidecar_path, sidecar_sha256)
    require(
        method in METHODS and model_id in sidecar["models"],
        "Inference-view model or method is absent",
    )
    model = sidecar["models"][model_id]
    require(
        model.get("base_path") == base_path
        and model.get("model_revision") == model_revision
        and model.get("tokenizer_revision") == tokenizer_revision
        and model.get("base_config_sha256") == base_config_sha256
        and isinstance(model_revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", model_revision)
        and isinstance(tokenizer_revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision)
        and _hash(base_config_sha256),
        "Inference-view base model or tokenizer pin differs",
    )
    view = model["methods"][method]
    require(isinstance(view, dict), "Inference-view method descriptor is invalid")
    original = _files(original_files_sha256)
    observed = _files(evaluation_files_sha256)
    require(
        _hash(receipt_sha256) and digest(receipt_path) == receipt_sha256,
        "Portable original training receipt hash differs",
    )
    reference = view.get("original_receipt", {})
    require(
        isinstance(reference.get("path"), str)
        and Path(reference["path"]).is_absolute()
        and reference.get("sha256") == receipt_sha256,
        "Inference view references another original receipt",
    )
    receipt = json.loads(Path(receipt_path).read_text(), object_pairs_hook=_object)
    require(
        receipt.get("method") == method
        and receipt.get("status")
        == ("trained_not_evaluated" if method == "rmu" else "trained_merged"),
        "Inference-view receipt does not describe completed method training",
    )
    pins = receipt.get("config", {}) if method == "rmu" else receipt
    require(
        pins.get("model_id") == model_id
        and pins.get("model_revision") == model_revision
        and pins.get("tokenizer_revision") == tokenizer_revision,
        "Original receipt model or tokenizer pin differs",
    )
    actual_files = receipt.get(
        "checkpoint_identity" if method == "rmu" else "merged_checkpoint_files", []
    )
    require(
        isinstance(actual_files, list)
        and len(actual_files) == len(original)
        and {item["path"]: item["sha256"] for item in actual_files} == original,
        "Original file identity differs from actual trainer receipt",
    )
    require(
        view.get("original_checkpoint")
        == original_checkpoint
        == receipt.get("checkpoint" if method == "rmu" else "merged_checkpoint")
        and view.get("original_files_sha256") == original,
        "Inference view changes the original training checkpoint identity",
    )
    require(
        view.get("evaluation_checkpoint") == evaluation_checkpoint
        and Path(evaluation_checkpoint).is_absolute(),
        "Native checkpoint path differs from the certified inference view",
    )
    require(
        view.get("evaluation_files_sha256") == observed,
        "Measured checkpoint files differ from the certified inference view",
    )
    assets = view.get("added_assets")
    require(isinstance(assets, dict), "Inference-view added assets are missing")
    kind = view.get("kind")
    asset_binding = "not_applicable"
    if kind == "original":
        require(
            not assets
            and evaluation_checkpoint == original_checkpoint
            and observed == original,
            "Original inference view must preserve its path and every original file",
        )
    elif kind == "add_pinned_processor_assets":
        require(
            model_id in GEMMA_MODELS
            and set(assets) == PROCESSOR_ASSETS
            and not (set(original) & PROCESSOR_ASSETS)
            and evaluation_checkpoint != original_checkpoint,
            "Only the two missing pinned Gemma processor assets may be added",
        )
        expected = dict(original)
        source_files = {
            item["path"]: item["sha256"]
            for item in receipt.get("source_model_files", [])
        }
        measured_base = (
            _files(base_files_sha256) if base_files_sha256 is not None else None
        )
        for filename, asset in assets.items():
            require(
                isinstance(asset, dict)
                and set(asset) == {"source_path", "sha256"}
                and asset["source_path"] == str(Path(base_path) / filename)
                and _hash(asset["sha256"]),
                "Processor asset does not come from its pinned base checkpoint",
            )
            if filename in source_files:
                require(
                    source_files[filename] == asset["sha256"],
                    "Processor asset differs from actual trainer base files",
                )
            if measured_base is not None:
                require(
                    measured_base.get(filename) == asset["sha256"],
                    "Processor asset differs from measured pinned base files",
                )
            expected[filename] = asset["sha256"]
        asset_binding = (
            "measured_native_base_files"
            if measured_base is not None
            else "trainer_source_files"
            if source_files.keys() >= PROCESSOR_ASSETS
            else "certificate_source_paths_only_pending_native_base_proof"
        )
        require(
            observed == expected,
            "Derived checkpoint changes, removes or adds files beyond the two processor assets",
        )
    else:
        raise ValueError("Unsupported inference-view kind")
    return {
        "status": "validated_inference_view_of_actual_training_checkpoint",
        "kind": kind,
        "model_id": model_id,
        "method": method,
        "sidecar": {"path": str(sidecar_path), "sha256": sidecar_sha256},
        "original_receipt": dict(reference),
        "original_checkpoint": original_checkpoint,
        "original_files_sha256": original,
        "evaluation_checkpoint": evaluation_checkpoint,
        "evaluation_files_sha256": observed,
        "added_assets": assets,
        "processor_asset_base_binding": asset_binding,
        "original_weight_config_tokenizer_files_unchanged": True,
    }
