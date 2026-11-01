"""Credential scoping fixtures never use or print a real API key."""

import os

import pytest

from lm_eval.fair_uk import credentials


def test_existing_environment_does_not_read_keychain(monkeypatch):
    monkeypatch.setenv("FAIRFORGET_FIXTURE_CREDENTIAL", "fixture-only")

    def forbidden(*args):
        pytest.fail("Existing environment must not read Keychain")

    monkeypatch.setattr(credentials, "mac_keychain_secret", forbidden)
    with credentials.credential_environment("FAIRFORGET_FIXTURE_CREDENTIAL"):
        assert os.environ["FAIRFORGET_FIXTURE_CREDENTIAL"] == "fixture-only"
    assert os.environ["FAIRFORGET_FIXTURE_CREDENTIAL"] == "fixture-only"


def test_selected_secret_is_temporary_even_on_failure(monkeypatch):
    monkeypatch.delenv("FAIRFORGET_FIXTURE_CREDENTIAL", raising=False)
    monkeypatch.setattr(
        credentials, "mac_keychain_secret", lambda service, account: "fixture-only"
    )
    with (
        pytest.raises(RuntimeError),
        credentials.credential_environment(
            "FAIRFORGET_FIXTURE_CREDENTIAL",
            {"service": "fixture-service", "account": "fixture-account"},
        ),
    ):
        assert os.environ["FAIRFORGET_FIXTURE_CREDENTIAL"] == "fixture-only"
        raise RuntimeError("Fixture failure")
    assert "FAIRFORGET_FIXTURE_CREDENTIAL" not in os.environ


def test_missing_or_malformed_selection_never_reads_keychain(monkeypatch):
    monkeypatch.delenv("FAIRFORGET_FIXTURE_CREDENTIAL", raising=False)
    for selection in (None, {}, {"service": "fixture"}):
        with (
            pytest.raises(ValueError),
            credentials.credential_environment(
                "FAIRFORGET_FIXTURE_CREDENTIAL", selection
            ),
        ):
            pytest.fail("Missing selection cannot enter credential scope")


def test_keychain_requires_supported_host(monkeypatch):
    monkeypatch.setattr(credentials.sys, "platform", "linux")
    with pytest.raises(ValueError, match="set the credential environment"):
        credentials.mac_keychain_secret("fixture", "fixture")
