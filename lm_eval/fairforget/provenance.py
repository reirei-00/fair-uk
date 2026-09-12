"""Run identity checks that prevent silently mixing model checkpoints."""

import hashlib
import re
from pathlib import Path

from lm_eval.utils import simple_parse_args_string


def model_identity(model_args):
    arguments = simple_parse_args_string(model_args)
    for key in ("tokenizer", "peft", "delta", "gguf_file"):
        if arguments.get(key):
            raise ValueError(
                f"Separate {key} artifacts are not yet fingerprinted; use a self-contained model checkpoint"
            )
    pretrained = arguments.get("pretrained")
    if not pretrained:
        raise ValueError("Provide pretrained=... in --model-args")
    path = Path(str(pretrained))
    if path.is_dir():
        hashes = {}
        for file in sorted(path.rglob("*")):
            if not file.is_file() or ".cache" in file.relative_to(path).parts:
                continue
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[str(file.relative_to(path))] = digest.hexdigest()
        if not hashes:
            raise ValueError("Local model directory is empty")
        identity = {"local_files_sha256": hashes}
    else:
        revision = str(arguments.get("revision", ""))
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(
                "Pin a remote model with revision=<40-character commit SHA> in --model-args"
            )
        identity = {"repository": str(pretrained), "revision": revision}
    # Credentials must not appear in report artifacts. Hash the full invocation
    # for safe resume matching while displaying only non-secret parameters.
    safe = {
        key: value
        for key, value in arguments.items()
        if not any(
            word in key.lower() for word in ("token", "secret", "password", "api_key")
        )
    }
    identity.update(
        parameters=safe,
        arguments_sha256=hashlib.sha256(model_args.encode()).hexdigest(),
    )
    return identity
