import pytest

from lm_eval.fairforget.runner import encode_request


class Tokenizer:
    bos_token_id = 9
    pad_token_id = 8
    eos_token_id = 7

    def encode(self, text, add_special_tokens=True):
        return {"prompt ": [9, 1, 2], "prompt A": [9, 1, 3, 4], "sentence": [5, 6]}[
            text
        ]


class Backend:
    tokenizer = Tokenizer()
    max_length = 100
    backend = "causal"

    def _loglikelihood_tokens(self, requests):
        return []


def test_boundary_retokenization():
    _, context, continuation = encode_request(
        Backend(), {"context": "prompt ", "continuation": "A"}
    )
    assert context == [9, 1] and continuation == [3, 4]


def test_sentence_prefix_and_truncation():
    _, context, continuation = encode_request(
        Backend(), {"context": "", "continuation": "sentence"}, sentence=True
    )
    assert context == [9] and continuation == [5, 6]
    short = Backend()
    short.max_length = 2
    with pytest.raises(ValueError, match="truncation"):
        encode_request(
            short, {"context": "", "continuation": "sentence"}, sentence=True
        )
