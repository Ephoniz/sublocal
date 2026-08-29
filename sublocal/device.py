"""Pick CUDA vs CPU without silently hiding a working GPU."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from sublocal.progress import status

# Values that make CUDA report zero devices. Dropped in-process only.
_HIDING = frozenset({"", "-1"})
_CT2_RUNTIME_DLLS = ("cublas64_12.dll", "cudart64_12.dll")
_V13_PATH = re.compile(r"(?:^|[\\/])v13(?:\.\d+)?(?:[\\/]|$)", re.IGNORECASE)


class CudaUnavailableError(RuntimeError):
    """`--device cuda` was requested but CTranslate2 still sees no GPU."""


def unhide_cuda_env() -> str | None:
    """If CUDA_VISIBLE_DEVICES hides every GPU, drop it in this process.

    Does not change the parent shell. Call this *before* importing
    ctranslate2 so the CUDA runtime never sees a leftover ``-1``.
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        return None
    raw = os.environ["CUDA_VISIBLE_DEVICES"]
    if raw.strip() not in _HIDING:
        return None
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    return raw


def cuda_device_count() -> int:
    """CTranslate2's visible CUDA device count. Import happens here."""
    import logging

    import ctranslate2

    if not os.environ.get("CTRANSLATE2_LOG_LEVEL"):
        ctranslate2.set_log_level(logging.INFO)
    return int(ctranslate2.get_cuda_device_count())


def cuda_device_name(index: int = 0) -> str | None:
    """Best-effort GPU name (nvidia-smi). None if unavailable."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    name = (proc.stdout or "").strip().splitlines()
    return name[0].strip() if name and name[0].strip() else None


def _cuda_label(index: int = 0) -> str:
    name = cuda_device_name(index)
    if name:
        return f"{name} (cuda:{index})"
    return f"cuda:{index}"


def _dll_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        dirs.append(Path(cuda_path) / "bin")
    for part in os.environ.get("PATH", "").split(os.pathsep):
        if part:
            dirs.append(Path(part))
    return dirs


def _find_runtime_dll(name: str) -> Path | None:
    seen: set[Path] = set()
    for directory in _dll_search_dirs():
        try:
            resolved = directory.resolve()
        except OSError:
            resolved = directory
        if resolved in seen:
            continue
        seen.add(resolved)
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _dll_status(name: str) -> str:
    found = _find_runtime_dll(name)
    if found is not None:
        return f"{name} found at {found}"
    return f"{name} not found on PATH or CUDA_PATH\\bin"


def _cuda_path_looks_v13(cuda_path: str) -> bool:
    return bool(_V13_PATH.search(cuda_path.replace("/", "\\")))


def _cuda_missing_message(*, requested: str, hidden: str | None) -> str:
    bits = [f"CTranslate2 get_cuda_device_count()=0 (--device {requested})"]
    leftover = hidden
    if leftover is None:
        current = os.environ.get("CUDA_VISIBLE_DEVICES")
        if current is not None and current.strip() in _HIDING:
            leftover = current
    if leftover is not None:
        if leftover.strip() == "":
            bits.append(
                "leftover empty CUDA_VISIBLE_DEVICES was cleared in-process"
            )
        else:
            bits.append(
                f"after clearing hiding CUDA_VISIBLE_DEVICES={leftover!r} "
                "in-process"
            )
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        bits.append(f"CUDA_PATH={cuda_path}")
        if _cuda_path_looks_v13(cuda_path):
            bits.append(
                "CUDA_PATH looks like v13.x; the CTranslate2 4.8 wheel still "
                "needs the CUDA 12.x bin on PATH"
            )
    else:
        bits.append("CUDA_PATH is unset")
    bits.extend(_dll_status(name) for name in _CT2_RUNTIME_DLLS)
    bits.append(
        "Official CPython only — Anaconda on Windows still AV-crashes. "
        "CUDA_VISIBLE_DEVICES=-1 hides every GPU."
    )
    return ". ".join(bits)


def resolve_device(requested: str) -> str:
    """Return ``cuda`` or ``cpu``. Never fake CPU when ``requested=='cuda'``."""
    hidden: str | None = None
    if requested != "cpu":
        hidden = unhide_cuda_env()

    if requested == "cpu":
        status("Using CPU")
        return "cpu"

    count = cuda_device_count()
    if count > 0:
        status(f"Using {_cuda_label(0)}")
        return "cuda"

    why = _cuda_missing_message(requested=requested, hidden=hidden)
    if requested == "cuda":
        raise CudaUnavailableError(why)
    status(why)
    return "cpu"
