"""Status lines and real tqdm bars on stderr.

Download bars are huggingface_hub's (bytes / file counts). Cue bars use the
same tqdm package huggingface_hub already depends on. Nothing is a fake spinner.
"""

from __future__ import annotations

import os
import sys
from typing import Any


def status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def enable_download_progress() -> None:
    """Turn on huggingface_hub progress bars, including when stderr is not a TTY.

    huggingface_hub disables tqdm when it cannot detect a TTY, which is why a
    600 MB first-run download can look hung. ``TQDM_POSITION=-1`` is the
    library's own switch to force bars on.

    The hub logger is kept at ERROR so the unauthenticated-token warning does
    not bury the bars. tqdm writes to stderr itself, not through that logger.
    """
    import logging

    os.environ.setdefault("TQDM_POSITION", "-1")
    from huggingface_hub.utils import enable_progress_bars

    enable_progress_bars()
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def stderr_tqdm_class() -> type:
    """Vanilla tqdm that always writes to stderr.

    Passed to ``snapshot_download(tqdm_class=...)``. huggingface_hub does not
    TTY-gate or log-level-gate a non-subclass of its own tqdm wrapper, so the
    per-file byte bars actually move.
    """
    from tqdm.auto import tqdm as vanilla_tqdm

    class StderrTqdm(vanilla_tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("file", sys.stderr)
            kwargs.setdefault("dynamic_ncols", True)
            kwargs.setdefault("mininterval", 0.2)
            kwargs.setdefault("disable", False)
            super().__init__(*args, **kwargs)

    return StderrTqdm


def cue_bar(total: int, desc: str = "Translating"):
    """Count cues (not text) on stderr."""
    enable_download_progress()
    cls = stderr_tqdm_class()
    return cls(
        total=total,
        desc=desc,
        unit="cue",
        leave=True,
    )
