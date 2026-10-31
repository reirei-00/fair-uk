"""Runtime asset identities use local serialization/encoding, never model calls."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from lm_eval.fair_uk.data import PROTOCOLS
from lm_eval.fair_uk.evaluation_assets import (
    _tokenizer_metadata,
    evaluation_asset_identity,
)


class FakeTokenizer:
    def __init__(self):
        self.state = {"model": {"vocab": {"a": 1}}, "normalizer": {"type": "fixture"}}
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps(self.state))
        self.bos_token_id, self.pad_token_id, self.eos_token_id = 10, 11, 12
        self.special_tokens_map = {
            "bos_token": "<bos>",
            "pad_token": "<pad>",
            "eos_token": "<eos>",
        }
        self.all_special_ids = [10, 11, 12]
        self.chat_template = "user:{{content}}:assistant:"
        self.add_bos_token = False
        self.offset = 0
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        return ([self.bos_token_id] if add_special_tokens else []) + [
            ord(c) + self.offset for c in text
        ]


class FakeBackend:
    backend = "causal"
    max_length = 10000

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.chat_template_args = {"enable_thinking": False}
        self.calls = []

    def _loglikelihood_tokens(self, *_args):
        raise AssertionError("Asset identity must never request likelihoods")

    def generate_until(self, *_args):
        raise AssertionError("Asset identity must never request generation")

    def tok_encode(self, text):
        self.calls.append(text)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def apply_chat_template(self, history, add_generation_prompt=True):
        assert add_generation_prompt
        return self.tokenizer.chat_template.replace(
            "{{content}}", history[0]["content"]
        )


def triplet(identifier="a"):
    return {
        "id": identifier,
        "task": "stereoset_uk",
        "protocol": PROTOCOLS["stereoset_uk"],
        "sentences": ["S", "C", "U"],
        "source": {},
    }


def benign(identifier="a"):
    return {
        "id": identifier,
        "task": "warbias_benign_uk",
        "protocol": PROTOCOLS["warbias_benign"],
        "source": {"prompt": "Calculate 2 + 3."},
    }


def test_likelihood_matches_reference_sentence_prefix_and_is_id_order_stable():
    backend = FakeBackend()
    rows = [triplet("b"), triplet("a")]
    original = deepcopy(rows)
    identity = evaluation_asset_identity(backend, rows, "native-chat")
    assert identity["rows"] == 2 and identity["input_mode"] == "likelihood"
    assert identity["generation_prompt_format"] is None
    assert identity["native_template_sha256"] is None
    assert len(identity["tokenizer_sha256"]) == len(identity["input_sha256"]) == 64
    assert identity == evaluation_asset_identity(backend, list(reversed(rows)), "raw")
    assert rows == original
    assert all(
        add_special_tokens is False for _, add_special_tokens in backend.tokenizer.calls
    )
    assert backend.calls == []


def test_tokenizer_normalizer_and_special_token_changes_are_captured():
    backend = FakeBackend()
    first = evaluation_asset_identity(backend, [triplet()], "raw")
    backend.tokenizer.state["normalizer"]["type"] = "different"
    changed = evaluation_asset_identity(backend, [triplet()], "raw")
    assert changed["tokenizer_sha256"] != first["tokenizer_sha256"]
    assert changed["input_sha256"] == first["input_sha256"]
    backend.tokenizer.bos_token_id = 99
    different_bos = evaluation_asset_identity(backend, [triplet()], "raw")
    assert different_bos["input_sha256"] != changed["input_sha256"]
    assert different_bos["tokenizer_sha256"] != changed["tokenizer_sha256"]


def special_token_backend():
    backend = FakeBackend()
    tokenizer = backend.tokenizer
    descriptor = SimpleNamespace(
        content="<pad>",
        single_word=False,
        lstrip=False,
        rstrip=False,
        normalized=False,
        special=True,
    )
    tokenizer.added_tokens_decoder = {11: descriptor}
    tokenizer.convert_tokens_to_ids = lambda value: 11 if value == "<pad>" else None
    tokenizer.special_tokens_map_extended = {"pad_token": "<pad>"}
    return backend, descriptor


def test_runtime_and_saved_special_token_representations_have_the_same_identity():
    backend, descriptor = special_token_backend()
    tokenizer = backend.tokenizer
    original = deepcopy(tokenizer.special_tokens_map_extended)
    string_identity = evaluation_asset_identity(backend, [triplet()], "raw")
    assert tokenizer.special_tokens_map_extended == original
    tokenizer.special_tokens_map_extended["pad_token"] = descriptor
    assert evaluation_asset_identity(backend, [triplet()], "raw") == string_identity
    assert tokenizer.special_tokens_map_extended["pad_token"] is descriptor


@pytest.mark.parametrize(
    "flag", ["single_word", "lstrip", "rstrip", "normalized", "special"]
)
def test_special_token_canonicalization_preserves_actual_descriptor_flags(flag):
    backend, descriptor = special_token_backend()
    original = evaluation_asset_identity(backend, [triplet()], "raw")
    setattr(descriptor, flag, not getattr(descriptor, flag))
    changed = evaluation_asset_identity(backend, [triplet()], "raw")
    assert changed["tokenizer_sha256"] != original["tokenizer_sha256"]
    assert changed["input_sha256"] == original["input_sha256"]
    metadata = _tokenizer_metadata(backend.tokenizer, backend)
    assert metadata["special_tokens_map_extended"]["pad_token"][flag] == getattr(
        descriptor, flag
    )


@pytest.mark.parametrize("identifier", [None, True, -1, 999])
def test_special_token_strings_without_verified_decoder_ids_stay_distinct(identifier):
    backend, descriptor = special_token_backend()
    backend.tokenizer.convert_tokens_to_ids = lambda value: identifier
    string_identity = evaluation_asset_identity(backend, [triplet()], "raw")
    backend.tokenizer.special_tokens_map_extended["pad_token"] = descriptor
    assert (
        evaluation_asset_identity(backend, [triplet()], "raw")["tokenizer_sha256"]
        != string_identity["tokenizer_sha256"]
    )


def test_special_token_decoder_content_must_match_the_original_string():
    backend, descriptor = special_token_backend()
    original = backend.tokenizer.special_tokens_map_extended["pad_token"]
    descriptor.content = "<another-token>"
    assert (
        _tokenizer_metadata(backend.tokenizer, backend)["special_tokens_map_extended"][
            "pad_token"
        ]
        == original
    )
    string_identity = evaluation_asset_identity(backend, [triplet()], "raw")
    backend.tokenizer.special_tokens_map_extended["pad_token"] = descriptor
    assert (
        evaluation_asset_identity(backend, [triplet()], "raw")["tokenizer_sha256"]
        != string_identity["tokenizer_sha256"]
    )


def test_additional_special_token_lists_resolve_only_existing_descriptors():
    backend, descriptor = special_token_backend()
    backend.tokenizer.special_tokens_map_extended = {
        "additional_special_tokens": ["<pad>", "unmapped"]
    }
    first = evaluation_asset_identity(backend, [triplet()], "raw")
    backend.tokenizer.special_tokens_map_extended = {
        "additional_special_tokens": [descriptor, "unmapped"]
    }
    assert evaluation_asset_identity(backend, [triplet()], "raw") == first


def test_effective_input_tokens_detect_changed_encoding_even_same_serialization():
    backend = FakeBackend()
    first = evaluation_asset_identity(backend, [triplet()], "raw")
    backend.tokenizer.offset = 1
    changed = evaluation_asset_identity(backend, [triplet()], "raw")
    assert first["tokenizer_sha256"] == changed["tokenizer_sha256"]
    assert first["input_sha256"] != changed["input_sha256"]


def test_generation_uses_actual_native_context_tok_encode_and_template_identity():
    backend = FakeBackend()
    row = benign()
    identity = evaluation_asset_identity(backend, [row], "native-chat")
    assert backend.calls == ["user:Calculate 2 + 3.:assistant:"]
    assert identity["generation_prompt_format"] == "native-chat"
    assert identity["native_template_sha256"] is not None
    backend.tokenizer.chat_template += "another prefix"
    changed = evaluation_asset_identity(backend, [row], "native-chat")
    assert changed["tokenizer_sha256"] != identity["tokenizer_sha256"]
    assert changed["input_sha256"] != identity["input_sha256"]
    backend.chat_template_args["enable_thinking"] = True
    changed_options = evaluation_asset_identity(backend, [row], "native-chat")
    assert changed_options["tokenizer_sha256"] != changed["tokenizer_sha256"]


def test_raw_generation_ignores_unused_template_but_preserves_input_identity():
    backend = FakeBackend()
    first = evaluation_asset_identity(backend, [benign()], "raw")
    assert backend.calls == ["Calculate 2 + 3."]
    backend.tokenizer.chat_template += "unused"
    assert evaluation_asset_identity(backend, [benign()], "raw") == first


def test_sentencepiece_complete_model_bytes_are_supported_and_changes_detected():
    backend = FakeBackend()
    del backend.tokenizer.backend_tokenizer
    backend.tokenizer.sp_model = SimpleNamespace(
        serialized_model_proto=lambda: b"complete-model-1"
    )
    first = evaluation_asset_identity(backend, [triplet()], "raw")
    assert first["tokenizer_serialization"] == "sentencepiece_model_proto"
    backend.tokenizer.sp_model.serialized_model_proto = lambda: b"complete-model-2"
    assert (
        evaluation_asset_identity(backend, [triplet()], "raw")["tokenizer_sha256"]
        != first["tokenizer_sha256"]
    )


@pytest.mark.parametrize("value", ["", "[]", "not-json"])
def test_incomplete_serialization_is_rejected(value):
    backend = FakeBackend()
    backend.tokenizer.backend_tokenizer.to_str = lambda: value
    with pytest.raises(ValueError, match="serializ"):
        evaluation_asset_identity(backend, [triplet()], "raw")


def test_missing_complete_serialization_fails_instead_of_hashing_vocab_only():
    backend = FakeBackend()
    del backend.tokenizer.backend_tokenizer
    backend.tokenizer.get_vocab = lambda: {"a": 1}
    with pytest.raises(ValueError, match="complete"):
        evaluation_asset_identity(backend, [triplet()], "raw")


def test_non_sentence_likelihood_uses_reference_boundary_encoding():
    backend = FakeBackend()
    row = {
        "id": "a",
        "task": "winobias_uk_natural",
        "protocol": PROTOCOLS["winobias_uk_natural"],
        "source": {"sentence_uk": "Речення"},
        "language": "uk",
        "pronoun": "він",
        "choices": ["first", "second"],
    }
    identity = evaluation_asset_identity(backend, [row], "raw")
    assert identity["input_mode"] == "likelihood"
    assert len(backend.tokenizer.calls) == 4
    assert all(special for _, special in backend.tokenizer.calls)


def test_bad_or_duplicate_rows_and_prompt_format_fail():
    backend = FakeBackend()
    for rows in ([], [triplet(), triplet()], [triplet(), benign("b")]):
        with pytest.raises(ValueError):
            evaluation_asset_identity(backend, rows, "raw")
    with pytest.raises(ValueError, match="prompt format"):
        evaluation_asset_identity(backend, [triplet()], "guess")
