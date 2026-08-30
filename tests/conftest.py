from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _stub_live_ginza(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI must not download or load ja_ginza. Unit tests inject a fake nlp."""
    monkeypatch.setattr("sublocal.extract.load_ginza", lambda: None)


@pytest.fixture
def sample_srt() -> Path:
    return FIXTURES / "sample.es.srt"


@pytest.fixture
def sample_vtt() -> Path:
    return FIXTURES / "sample.vtt"


@pytest.fixture
def sample_ass() -> Path:
    return FIXTURES / "sample.ass"
