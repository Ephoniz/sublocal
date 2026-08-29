"""Refuse Windows + Anaconda Python before CTranslate2 can AV-crash."""

from __future__ import annotations

import sys


class UnsupportedPythonError(RuntimeError):
    """This interpreter will crash CTranslate2; do not load the NLLB backend."""


ANACONDA_WINDOWS_MESSAGE = (
    "CTranslate2 crashes on Windows Anaconda/Miniconda Python "
    "(access violation 0xC0000005, no traceback). "
    "Use official CPython 3.11+ from https://www.python.org/downloads/ "
    "or uv (https://docs.astral.sh/uv/), recreate the venv, and run sublocal again."
)


def is_anaconda_python(version: str | None = None) -> bool:
    text = sys.version if version is None else version
    return "Anaconda" in text or "packaged by Anaconda" in text


def reject_anaconda_on_windows(
    *,
    platform: str | None = None,
    version: str | None = None,
) -> None:
    """Raise if this is Windows plus Anaconda/Miniconda CPython.

    Official CPython from python.org or uv is fine. Linux/macOS Anaconda is
    not blocked. Detection uses ``sys.version`` only, as specified.
    """
    plat = sys.platform if platform is None else platform
    if plat != "win32":
        return
    if is_anaconda_python(version):
        raise UnsupportedPythonError(ANACONDA_WINDOWS_MESSAGE)
