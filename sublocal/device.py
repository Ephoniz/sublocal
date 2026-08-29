"""Pick CUDA vs CPU without silently hiding a working GPU."""

from __future__ import annotations

import os
import subprocess

from sublocal.progress import status

# Values that make CUDA report zero devices. Dropped in-process only.
_HIDING = frozenset({"", "-1"})


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


def _cuda_missing_message(*, requested: str, hidden: str | None) -> str:
    bits = [f"CTranslate2 get_cuda_device_count()=0 (--device {requested})"]
    if hidden is not None:
        bits.append(
            f"after clearing hiding CUDA_VISIBLE_DEVICES={hidden!r} in-process"
        )
    bits.append(
        "Need a visible GPU and the CUDA 12 runtime the CTranslate2 wheel "
        "loads (12.2 / 12.6 / 13.1). Official CPython only — Anaconda on "
        "Windows still AV-crashes. CUDA_VISIBLE_DEVICES=-1 hides every GPU."
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
    status(f"Using CPU ({why})")
    return "cpu"
