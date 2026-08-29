from __future__ import annotations

import re

from sublocal.formats.base import Block, Cue, Document, normalize_newlines

TIMESTAMP_RE = re.compile(
    r"^\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}"
)


def parse_srt(content: str) -> Document:
    content, newline = normalize_newlines(content)
    content = content.lstrip("\ufeff")
    stripped = content.strip("\n")
    if not stripped:
        return Document(format="srt", blocks=[], newline=newline)

    blocks: list[Block] = []
    for chunk in re.split(r"\n{2,}", stripped):
        lines = chunk.split("\n")
        if not any(line.strip() for line in lines):
            continue
        idx = 0
        index: str | None = None
        if lines[0].strip().isdigit():
            index = lines[0].strip()
            idx = 1
        if idx >= len(lines) or not TIMESTAMP_RE.match(lines[idx].strip()):
            blocks.append(Block(kind="raw", raw=chunk))
            continue
        timing = lines[idx]
        text = "\n".join(lines[idx + 1 :])
        blocks.append(
            Block(
                kind="cue",
                cue=Cue(index=index, timing=timing, text=text),
            )
        )
    return Document(format="srt", blocks=blocks, newline=newline)


def dumps_srt(doc: Document) -> str:
    parts: list[str] = []
    cue_no = 0
    for block in doc.blocks:
        if block.kind == "raw":
            parts.append(block.raw)
            continue
        assert block.cue is not None
        cue_no += 1
        index = block.cue.index if block.cue.index is not None else str(cue_no)
        body = f"{index}\n{block.cue.timing}"
        if block.cue.text:
            body += f"\n{block.cue.text}"
        parts.append(body)
    text = "\n\n".join(parts)
    if text and not text.endswith("\n"):
        text += "\n"
    if doc.newline != "\n":
        text = text.replace("\n", doc.newline)
    return text
