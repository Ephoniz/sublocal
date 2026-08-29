from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Cue:
    """One timed subtitle cue. ``timing`` is the original timestamp line/fields."""

    index: str | None
    timing: str
    text: str
    extra: dict = field(default_factory=dict)


@dataclass
class Block:
    """A document chunk: either a translatable cue or raw passthrough text."""

    kind: Literal["cue", "raw"]
    raw: str = ""
    cue: Cue | None = None


@dataclass
class Document:
    format: str
    blocks: list[Block]
    newline: str = "\n"

    @property
    def cues(self) -> list[Cue]:
        return [b.cue for b in self.blocks if b.kind == "cue" and b.cue is not None]


def read_text(path: str) -> str:
    with open(path, encoding="utf-8-sig") as fh:
        return fh.read()


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def normalize_newlines(content: str) -> tuple[str, str]:
    if "\r\n" in content:
        return content.replace("\r\n", "\n"), "\r\n"
    if "\r" in content and "\n" not in content:
        return content.replace("\r", "\n"), "\r"
    return content, "\n"
