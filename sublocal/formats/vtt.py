from __future__ import annotations

import re

from sublocal.formats.base import Block, Cue, Document, normalize_newlines

TIMESTAMP_RE = re.compile(
    r"^\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3}\s*-->\s*\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3}"
)


def _is_header_block(chunk: str, seen_cue: bool) -> bool:
    first = chunk.lstrip().split("\n", 1)[0]
    if first.startswith("WEBVTT"):
        return True
    if first.startswith(("NOTE", "STYLE", "REGION")):
        return True
    if not seen_cue and "-->" not in chunk:
        return True
    return False


def parse_vtt(content: str) -> Document:
    content, newline = normalize_newlines(content)
    content = content.lstrip("\ufeff")
    stripped = content.strip("\n")
    blocks: list[Block] = []
    if not stripped:
        return Document(format="vtt", blocks=blocks, newline=newline)

    seen_cue = False
    for chunk in re.split(r"\n{2,}", stripped):
        if not chunk.strip():
            continue
        if _is_header_block(chunk, seen_cue):
            blocks.append(Block(kind="raw", raw=chunk))
            continue
        lines = chunk.split("\n")
        idx = 0
        index: str | None = None
        if "-->" not in lines[0]:
            index = lines[0]
            idx = 1
        if idx >= len(lines) or not TIMESTAMP_RE.match(lines[idx].strip()):
            blocks.append(Block(kind="raw", raw=chunk))
            continue
        timing = lines[idx]
        text = "\n".join(lines[idx + 1 :])
        seen_cue = True
        blocks.append(
            Block(kind="cue", cue=Cue(index=index, timing=timing, text=text))
        )
    return Document(format="vtt", blocks=blocks, newline=newline)


def dumps_vtt(doc: Document) -> str:
    parts: list[str] = []
    for block in doc.blocks:
        if block.kind == "raw":
            parts.append(block.raw)
            continue
        assert block.cue is not None
        lines: list[str] = []
        if block.cue.index:
            lines.append(block.cue.index)
        lines.append(block.cue.timing)
        if block.cue.text:
            lines.append(block.cue.text)
        parts.append("\n".join(lines))
    text = "\n\n".join(parts)
    if text and not text.endswith("\n"):
        text += "\n"
    if doc.newline != "\n":
        text = text.replace("\n", doc.newline)
    return text
