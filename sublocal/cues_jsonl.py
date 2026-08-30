"""Sidecar ``*.cues.jsonl`` — one JSON object per cue (start, end, text, lang)."""

from __future__ import annotations

import json
from pathlib import Path

from sublocal.formats.base import Cue, Document
from sublocal.formats.srt import split_timing


def cues_jsonl_path(path: str | Path) -> Path:
    """``movie.mp4`` / ``movie.srt`` → ``movie.cues.jsonl``."""
    p = Path(path)
    return p.with_name(f"{p.stem}.cues.jsonl")


def write_cues_jsonl(doc: Document, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for cue in doc.cues:
            start, end = _cue_span(cue)
            row = {
                "start": start,
                "end": end,
                "text": cue.text,
                "lang": cue.extra.get("lang"),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def read_cues_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def langs_from_sidecar(path: str | Path, doc: Document) -> list[str | None]:
    """Align sidecar langs to ``doc.cues`` (index, then start time)."""
    rows = read_cues_jsonl(path)
    if len(rows) == len(doc.cues):
        return [_as_lang(r.get("lang")) for r in rows]
    by_start: dict[float, str | None] = {}
    for row in rows:
        try:
            key = round(float(row["start"]), 3)
        except (KeyError, TypeError, ValueError):
            continue
        by_start[key] = _as_lang(row.get("lang"))
    langs: list[str | None] = []
    for cue in doc.cues:
        try:
            start, _ = split_timing(cue.timing)
        except ValueError:
            langs.append(None)
            continue
        langs.append(by_start.get(round(start, 3)))
    return langs


def _as_lang(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _cue_span(cue: Cue) -> tuple[float, float]:
    try:
        return split_timing(cue.timing)
    except ValueError:
        return 0.0, 0.0
