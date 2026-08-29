from __future__ import annotations

from pathlib import Path

from sublocal.formats.ass import dumps_ass, parse_ass
from sublocal.formats.base import Document, read_text, write_text
from sublocal.formats.srt import dumps_srt, parse_srt
from sublocal.formats.vtt import dumps_vtt, parse_vtt

SUPPORTED = {
    ".srt": "srt",
    ".vtt": "vtt",
    ".ass": "ass",
    ".ssa": "ass",
}


class UnsupportedFormatError(ValueError):
    pass


def format_of(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED:
        raise UnsupportedFormatError(
            f"Unsupported subtitle format {suffix or '(none)'}. "
            "v0.1 supports .srt (solid) and .vtt / .ass (best-effort)."
        )
    return SUPPORTED[suffix]


def load(path: str | Path) -> Document:
    kind = format_of(path)
    text = read_text(str(path))
    if kind == "srt":
        return parse_srt(text)
    if kind == "vtt":
        return parse_vtt(text)
    return parse_ass(text)


def dumps(doc: Document) -> str:
    if doc.format == "srt":
        return dumps_srt(doc)
    if doc.format == "vtt":
        return dumps_vtt(doc)
    if doc.format == "ass":
        return dumps_ass(doc)
    raise UnsupportedFormatError(f"Unknown document format {doc.format!r}.")


def save(doc: Document, path: str | Path) -> None:
    write_text(str(path), dumps(doc))
