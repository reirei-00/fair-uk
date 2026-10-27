import pytest

from lm_eval.fair_uk.data import prompts
from lm_eval.fair_uk.runner import encode_request, predict


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


def generation_row(task):
    return {
        "id": "item",
        "task": task,
        "language": "en",
        "choices": ["one", "two", "unknown"],
        "source": {
            "prompt": "benign request",
            "context": "evidence",
            "question": "who?",
        },
    }


class GenerationBackend:
    max_length = 10000

    def __init__(self):
        self.chat_template_args = {"enable_thinking": False}
        self.templates = []
        self.requests = []

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        self.templates.append((chat_history, add_generation_prompt))
        # A configured backend owns its template arguments; the runner must call
        # this harness method rather than bypassing it via the tokenizer.
        assert self.chat_template_args["enable_thinking"] is False
        return "<user>" + chat_history[0]["content"] + "</user><assistant>"

    def tok_encode(self, text):
        return list(text)

    def generate_until(self, requests):
        self.requests.extend(requests)
        return ["A"] * len(requests)


@pytest.mark.parametrize("task", ["warbias_en", "warbias_benign_en"])
def test_native_generation_uses_configured_backend_chat_template(task):
    row = generation_row(task)
    backend = GenerationBackend()
    assert predict(backend, [row], generation_prompt_format="native-chat") == [
        {"id": "item", "response": "A"}
    ]
    context = prompts(row)[0]["context"]
    assert backend.templates == [([{"role": "user", "content": context}], True)]
    assert backend.requests[0].args[0] == f"<user>{context}</user><assistant>"
    assert backend.requests[0].args[1]["do_sample"] is False


def test_raw_generation_preserves_legacy_prompt():
    row = generation_row("warbias_en")
    backend = GenerationBackend()
    predict(backend, [row])
    assert backend.templates == []
    assert backend.requests[0].args[0] == prompts(row)[0]["context"]


@pytest.mark.parametrize("template", [None, "unsupported", "missing", "empty"])
def test_native_generation_rejects_unavailable_template(template):
    backend = GenerationBackend()
    if template is None:
        backend.apply_chat_template = None
    elif template == "empty":
        backend.apply_chat_template = lambda *a, **k: ""
    else:

        def unavailable(*args, **kwargs):
            if template == "unsupported":
                raise NotImplementedError
            raise ValueError("Tokenizer has no chat template")

        backend.apply_chat_template = unavailable
    with pytest.raises(ValueError, match="chat template"):
        predict(
            backend,
            [generation_row("warbias_en")],
            generation_prompt_format="native-chat",
        )
    assert backend.requests == []


def test_generation_length_check_includes_native_template():
    row = generation_row("warbias_benign_en")
    backend = GenerationBackend()
    backend.max_length = len(prompts(row)[0]["context"]) + 2
    with pytest.raises(ValueError, match="context length"):
        predict(
            backend, [row], max_output_tokens=1, generation_prompt_format="native-chat"
        )
    assert backend.requests == []


def test_native_generation_option_preserves_raw_sentence_likelihood():
    backend = Backend()

    def forbidden(*args, **kwargs):
        pytest.fail("Sentence likelihood must not apply a chat template")

    backend.apply_chat_template = forbidden
    captured = []

    def likelihood(requests):
        captured.extend(requests)
        return [(-2.0, False)] * len(requests)

    backend._loglikelihood_tokens = likelihood
    row = {
        "id": "item",
        "task": "warbias_triplets_en",
        "source": {},
        "sentences": ["sentence"] * 3,
    }
    output = predict(backend, [row], generation_prompt_format="native-chat")
    assert output == [{"id": "item", "scores": [-1.0] * 3, "token_counts": [2] * 3}]
    assert all(
        context == [9] and continuation == [5, 6]
        for _, context, continuation in captured
    )
