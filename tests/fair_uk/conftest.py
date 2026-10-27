import pytest
from warbias_support import panel_inputs


@pytest.fixture
def warbias_inputs(tmp_path, monkeypatch):
    return panel_inputs(tmp_path, monkeypatch)
