from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_srt() -> Path:
    return FIXTURES / "sample.es.srt"


@pytest.fixture
def sample_vtt() -> Path:
    return FIXTURES / "sample.vtt"


@pytest.fixture
def sample_ass() -> Path:
    return FIXTURES / "sample.ass"
