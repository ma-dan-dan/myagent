from pathlib import Path

import pytest
import litellm


@pytest.fixture
def catalog_path() -> Path:
    return Path(__file__).parents[1] / "data" / "schema_catalog.json"


@pytest.fixture(autouse=True)
def mock_litellm_token_counter(monkeypatch):
    monkeypatch.setattr(litellm, "token_counter", lambda **kwargs: 1)
