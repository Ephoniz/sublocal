"""User-level cache directory for downloaded models."""

from __future__ import annotations

import os
from pathlib import Path


def cache_dir() -> Path:
    """Return the sublocal cache root, creating it if needed.

    Override with ``SUBLOCAL_CACHE``. Defaults:

    - Windows: ``%LOCALAPPDATA%\\sublocal``
    - Linux / macOS: ``$XDG_CACHE_HOME/sublocal`` or ``~/.cache/sublocal``
    """
    override = os.environ.get("SUBLOCAL_CACHE")
    if override:
        path = Path(override).expanduser()
    elif os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        path = Path(root) / "sublocal"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        path = Path(xdg) / "sublocal" if xdg else Path.home() / ".cache" / "sublocal"
    path.mkdir(parents=True, exist_ok=True)
    return path


def hf_cache_dir() -> Path:
    path = cache_dir() / "huggingface"
    path.mkdir(parents=True, exist_ok=True)
    return path
