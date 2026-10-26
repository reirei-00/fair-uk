import pytest
from warbias_support import case_inputs, panel_inputs


@pytest.fixture
def warbias_inputs(tmp_path, monkeypatch):
    return panel_inputs(tmp_path, monkeypatch)


@pytest.fixture
def inputs(tmp_path):
    """Case matrix and released triplets for the JEV case-annotation tests."""
    return case_inputs(tmp_path)
