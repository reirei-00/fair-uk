"""Hosted CLI integration with local sources and entirely mocked HTTP."""

import hashlib
import json

import httpx
import pytest

from lm_eval.fair_uk import __main__ as cli, api_runner
from lm_eval.fair_uk.data import REGISTRY, load, prompts


def source_fixture(path, language):
    rows = []
    for case, target in enumerate(("idp", "veteran")):
        for condition in ("ambiguous", "stereotype_aligned", "stereotype_conflicting"):
            for polarity in ("negative", "positive"):
                stereotype = int(polarity == "positive")
                rows.append(
                    {
                        "id": f"{case}/{condition}/{polarity}",
                        "scenario_id": str(case),
                        "source_case_id": str(case),
                        "target_group": target,
                        "language": language,
                        "evidence_condition": condition,
                        "question_polarity": polarity,
                        "answer_index": 2
                        if condition == "ambiguous"
                        else stereotype
                        if condition == "stereotype_aligned"
                        else 1 - stereotype,
                        "stereotype_consistent_index": stereotype,
                        "unknown_index": 2,
                        "choices": ["перша", "друга", "невідомо"]
                        if language == "uk"
                        else ["first", "second", "unknown"],
                        "context": f"Fixture {case} / {condition} / {language}",
                        "question": f"Fixture question {polarity}",
                    }
                )
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return {
        **REGISTRY[f"warbias_{language}"],
        "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.fixture
def local_sources(tmp_path, monkeypatch):
    paths = {}
    for language in ("uk", "en"):
        path = tmp_path / f"source_{language}.jsonl"
        monkeypatch.setitem(
            REGISTRY, f"warbias_{language}", source_fixture(path, language)
        )
        paths[language] = path
    return paths


def provider_config(tmp_path, provider):
    value = {
        "provider": provider,
        "model": "arbitrary-provider/model-name"
        if provider == "openai-compatible"
        else "arbitrary-gemini-model",
        "api_key_env": "KEY_MUST_NOT_BE_READ",
        "parameters": {},
        "max_output_tokens": 128,
    }
    if provider == "openai-compatible":
        value["base_url"] = "https://api.lapathoniia.top"
    path = tmp_path / "provider.json"
    path.write_text(json.dumps(value))
    return value, path


def run_arguments(language, source, configuration, output):
    return [
        "run-api",
        "--task",
        f"warbias_{language}",
        "--input",
        str(source),
        "--config",
        str(configuration),
        "--output",
        str(output),
        "--bootstrap",
        "0",
    ]


def forbid_named_key(monkeypatch):
    original = api_runner.os.environ.get

    def guarded(name, *args):
        if name == "KEY_MUST_NOT_BE_READ":
            pytest.fail("Planning or injected HTTP client read a provider credential")
        return original(name, *args)

    monkeypatch.setattr(api_runner.os.environ, "get", guarded)


@pytest.mark.parametrize("provider", ["openai-compatible", "gemini"])
def test_api_preview_execute_report_resume_and_paired_comparison(
    provider, tmp_path, local_sources, monkeypatch, capsys
):
    chosen, configuration = provider_config(tmp_path, provider)
    forbid_named_key(monkeypatch)
    outputs = {language: tmp_path / f"run_{language}" for language in local_sources}
    calls = []
    expected_answers = {
        prompts(row)[0]["context"]: "ABC"[row["gold"]]
        for language, path in local_sources.items()
        for row in load(f"warbias_{language}", path)
    }
    original_run = api_runner.run
    original_client = httpx.Client

    def forbidden_run(*_args, **_kwargs):
        pytest.fail("Preview executed a model request")

    monkeypatch.setattr(api_runner, "run", forbidden_run)
    monkeypatch.setattr(httpx, "Client", forbidden_run)
    cli.main(run_arguments("uk", local_sources["uk"], configuration, outputs["uk"]))
    preview = json.loads(capsys.readouterr().out)
    assert preview["requests"] == 12
    assert preview["configured_output_token_allowance"] == 12 * 128
    assert preview["estimated_cost_usd"] is None
    assert not outputs["uk"].exists()

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if provider == "gemini":
            prompt = payload["contents"][0]["parts"][0]["text"]
            assert payload["generationConfig"] == {"maxOutputTokens": 128}
            response = {
                "modelVersion": "resolved-provider-version",
                "candidates": [
                    {
                        "content": {"parts": [{"text": expected_answers[prompt]}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 1},
            }
        else:
            assert str(request.url) == "https://api.lapathoniia.top/chat/completions"
            assert payload["model"] == chosen["model"]
            assert payload["max_tokens"] == 128 and "seed" not in payload
            prompt = payload["messages"][0]["content"]
            response = {
                "model": "resolved-provider-version",
                "choices": [
                    {
                        "message": {"content": expected_answers[prompt]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1},
            }
        return httpx.Response(200, json=response)

    def mocked_run(rows, manifest, config, output):
        assert config == chosen
        with original_client(
            transport=httpx.MockTransport(handler), trust_env=False
        ) as client:
            return original_run(rows, manifest, config, output, client=client)

    monkeypatch.setattr(api_runner, "run", mocked_run)
    monkeypatch.setattr(api_runner.time, "sleep", lambda _: None)
    for language, source in local_sources.items():
        arguments = run_arguments(
            language, source, configuration, outputs[language]
        ) + ["--execute"]
        cli.main(arguments)
        output = outputs[language]
        report = json.loads((output / "report.json").read_text())
        manifest = json.loads((output / "run.json").read_text())
        assert report["provenance"] == manifest
        assert manifest["model_backend"] == "hosted-api"
        assert manifest["model_identity"] == api_runner.identity(chosen)
        assert manifest["runtime_versions"]["httpx"]
        assert manifest["seed"] == 42
        assert report["rows"] == 12 and report["source_cases"] == 2
        assert all(
            record["accuracy"] == 1
            for record in report["native"]
            if "condition" in record
        )
        assert report["api_usage"]["estimated_recorded_cost_usd"] is None
        before = {
            path: path.read_bytes() for path in (output / "responses").glob("*.json")
        }
        exported = (output / "predictions.jsonl").read_bytes()
        count = len(calls)
        cli.main(arguments)
        assert len(calls) == count
        assert exported == (output / "predictions.jsonl").read_bytes()
        assert all(path.read_bytes() == original for path, original in before.items())
        assert (output / "groups.csv").is_file() and (output / "report.md").is_file()
    assert len(calls) == 24
    comparison = tmp_path / "comparison"
    compare_arguments = [
        "compare",
        "--uk-run",
        str(outputs["uk"]),
        "--en-run",
        str(outputs["en"]),
        "--uk-input",
        str(local_sources["uk"]),
        "--en-input",
        str(local_sources["en"]),
        "--output",
        str(comparison),
        "--bootstrap",
        "0",
    ]
    cli.main(compare_arguments)
    result = json.loads((comparison / "comparison.json").read_text())
    assert result["paired_rows"] == 12 and result["source_cases"] == 2
    assert result["choice_disagreement_rate"] == 0
    assert (
        result["model_version_comparison"]["status"] == "matching_returned_identifiers"
    )
    assert result["model_version_comparison"]["uk"] == ["resolved-provider-version"]
    assert all(
        record["delta_en_minus_uk"] == 0 for record in result["native_comparisons"]
    )
    assert (comparison / "comparison.md").is_file()
    # Raw evidence, rather than mutable derived records, drives the comparison.
    (outputs["en"] / "records.jsonl").write_text("outdated derived output")
    cli.main(compare_arguments)
    assert len(calls) == 24
    assert "KEY_MUST_NOT_BE_READ" not in capsys.readouterr().out
    manifest_path = outputs["en"] / "run.json"
    changed = json.loads(manifest_path.read_text())
    changed["hosted_config"]["max_output_tokens"] = 64
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identity"):
        cli.main(compare_arguments)
    changed["hosted_config"]["max_output_tokens"] = 128
    changed["source_rows_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="fingerprint"):
        cli.main(compare_arguments)


def test_api_preview_rejects_likelihood_tasks_before_source_access(
    tmp_path, monkeypatch
):
    _, configuration = provider_config(tmp_path, "openai-compatible")

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unsupported task attempted dataset or HTTP access")

    monkeypatch.setattr(cli, "load", forbidden)
    monkeypatch.setattr(httpx, "Client", forbidden)
    output = tmp_path / "absent"
    with pytest.raises(ValueError, match="cannot score supplied candidate"):
        cli.main(
            [
                "run-api",
                "--task",
                "bbq_uk",
                "--config",
                str(configuration),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


@pytest.mark.parametrize("provider", ["openai-compatible", "gemini"])
def test_malformed_api_success_cannot_produce_a_benchmark_report(
    provider, tmp_path, local_sources, monkeypatch
):
    _, configuration = provider_config(tmp_path, provider)
    forbid_named_key(monkeypatch)
    output = tmp_path / "malformed-run"
    calls = []
    body = {"candidates": [{}]} if provider == "gemini" else {"choices": [{}]}

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=body)

    original_run = api_runner.run

    def mocked_run(rows, manifest, config, destination):
        with httpx.Client(
            transport=httpx.MockTransport(handler), trust_env=False
        ) as client:
            return original_run(rows, manifest, config, destination, client=client)

    monkeypatch.setattr(api_runner, "run", mocked_run)
    arguments = run_arguments("uk", local_sources["uk"], configuration, output) + [
        "--execute"
    ]
    with pytest.raises(RuntimeError, match="raw checkpoint was saved"):
        cli.main(arguments)
    with pytest.raises(ValueError):
        cli.main(arguments)
    assert len(calls) == 1
    assert json.loads((output / "status.json").read_text())["completed"] == 0
    assert len(list((output / "responses").glob("*.json"))) == 1
    for name in ("report.json", "report.md", "records.jsonl", "predictions.jsonl"):
        assert not (output / name).exists()


def test_bad_config_is_not_echoed_and_cannot_load_data(tmp_path, monkeypatch, capsys):
    _, configuration = provider_config(tmp_path, "openai-compatible")
    configuration.write_text(
        json.dumps(
            {
                "provider": "openai-compatible",
                "model": "any-model",
                "api_key": "SECRET_SENTINEL",
            }
        )
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("Invalid configuration attempted data or HTTP access")

    monkeypatch.setattr(cli, "load", forbidden)
    monkeypatch.setattr(httpx, "Client", forbidden)
    with pytest.raises(ValueError) as error:
        cli.main(
            [
                "run-api",
                "--task",
                "warbias_uk",
                "--config",
                str(configuration),
                "--output",
                str(tmp_path / "absent"),
            ]
        )
    assert "SECRET_SENTINEL" not in str(error.value)
    assert "SECRET_SENTINEL" not in capsys.readouterr().out
