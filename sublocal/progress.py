"""Status lines on stderr. Download bars are huggingface_hub tqdm.

Cue progress is one new line per batch (PowerShell-safe). No ``\\r`` bars
as the only signal — Windows PowerShell eats carriage returns.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any


def status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def format_eta(seconds: float) -> str:
    """Short remaining-time label, e.g. ``45s``, ``2m``, ``1h05m``."""
    secs = max(0, int(seconds))
    if secs < 60:
        return f"{secs}s"
    minutes, secs = divmod(secs, 60)
    if minutes < 60:
        return f"{minutes}m" if secs == 0 else f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class BatchCounter:
    """Print ``128/599 cues (21%) ~3m left`` on a new line after each batch."""

    def __init__(self, total: int) -> None:
        self.total = max(0, total)
        self.done = 0
        self._start = time.monotonic()

    def update(self, n: int) -> None:
        self.done = min(self.total, self.done + max(0, n))
        if self.total == 0:
            status("0/0 cues (100%)")
            return
        pct = int(self.done * 100 / self.total)
        line = f"{self.done}/{self.total} cues ({pct}%)"
        eta = self._eta()
        if eta:
            line = f"{line} ~{eta} left"
        status(line)

    def _eta(self) -> str:
        if self.done <= 0 or self.done >= self.total:
            return ""
        elapsed = time.monotonic() - self._start
        if elapsed <= 0:
            return ""
        remaining = (self.total - self.done) * (elapsed / self.done)
        return format_eta(remaining)


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
