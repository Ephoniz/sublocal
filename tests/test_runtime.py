from pathlib import Path

import pytest

from sublocal.backend import NllbBackend
from sublocal.cli import main
from sublocal.runtime import (
    ANACONDA_WINDOWS_MESSAGE,
    UnsupportedPythonError,
    is_anaconda_python,
    reject_anaconda_on_windows,
)


ANACONDA_VERSION = (
    "3.12.4 | packaged by Anaconda, Inc. | (main, Jun 18 2024, 10:07:17) "
    "[MSC v.1916 64 bit (AMD64)]"
)
ANACONDA_SHORT = "3.12.4 | Anaconda, Inc. | (main, Jun 18 2024)"
CPYTHON_VERSION = "3.11.4 (tags/v3.11.4:d2340ef, Jun  7 2023, 05:45:37) [MSC v.1934 64 bit (AMD64)]"
CONDA_FORGE_VERSION = "3.12.4 | packaged by conda-forge | (main, Jun 18 2024)"


def test_detects_anaconda_markers() -> None:
    assert is_anaconda_python(ANACONDA_VERSION)
    assert is_anaconda_python(ANACONDA_SHORT)
    assert not is_anaconda_python(CPYTHON_VERSION)
    assert not is_anaconda_python(CONDA_FORGE_VERSION)


def test_rejects_anaconda_on_windows() -> None:
    with pytest.raises(UnsupportedPythonError, match="0xC0000005"):
        reject_anaconda_on_windows(platform="win32", version=ANACONDA_VERSION)
    with pytest.raises(UnsupportedPythonError):
        reject_anaconda_on_windows(platform="win32", version=ANACONDA_SHORT)


def test_allows_official_cpython_on_windows() -> None:
    reject_anaconda_on_windows(platform="win32", version=CPYTHON_VERSION)


def test_allows_anaconda_off_windows() -> None:
    reject_anaconda_on_windows(platform="linux", version=ANACONDA_VERSION)
    reject_anaconda_on_windows(platform="darwin", version=ANACONDA_VERSION)


def test_nllb_backend_refuses_anaconda_windows(monkeypatch) -> None:
    monkeypatch.setattr("sublocal.runtime.sys.platform", "win32")
    monkeypatch.setattr("sublocal.runtime.sys.version", ANACONDA_VERSION)
    with pytest.raises(UnsupportedPythonError):
        NllbBackend(device="cpu")


def test_cli_anaconda_windows_nonzero_stderr(
    sample_srt: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("sublocal.runtime.sys.platform", "win32")
    monkeypatch.setattr("sublocal.runtime.sys.version", ANACONDA_VERSION)
    rc = main(
        [
            "translate",
            str(sample_srt),
            "--to",
            "en",
            "--from",
            "es",
            "--backend",
            "nllb",
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    assert "Anaconda" in captured.err
    assert "CTranslate2" in captured.err
    assert ANACONDA_WINDOWS_MESSAGE in captured.err.replace("\n", " ")
