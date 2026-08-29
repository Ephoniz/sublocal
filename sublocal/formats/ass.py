from __future__ import annotations

import re

from sublocal.formats.base import Block, Cue, Document, normalize_newlines

_OVERRIDE_RE = re.compile(r"^(\{[^}]*\})+")


def _events_format_fields(lines: list[str]) -> list[str]:
    in_events = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_events = stripped.lower() == "[events]"
            continue
        if in_events and stripped.lower().startswith("format:"):
            fields = stripped.split(":", 1)[1]
            return [f.strip() for f in fields.split(",") if f.strip()]
    return [
        "Layer",
        "Start",
        "End",
        "Style",
        "Name",
        "MarginL",
        "MarginR",
        "MarginV",
        "Effect",
        "Text",
    ]


def _split_fields(payload: str, field_count: int) -> list[str]:
    """Split a Dialogue/Comment payload; last field (Text) may contain commas."""
    return payload.split(",", field_count - 1)


def parse_ass(content: str) -> Document:
    content, newline = normalize_newlines(content)
    content = content.lstrip("\ufeff")
    # Keep a trailing empty line if the file ended with a newline.
    ended_nl = content.endswith("\n")
    lines = content.split("\n")
    if ended_nl and lines and lines[-1] == "":
        lines = lines[:-1]

    fields = _events_format_fields(lines)
    try:
        text_idx = next(i for i, name in enumerate(fields) if name.lower() == "text")
    except StopIteration:
        text_idx = len(fields) - 1
    try:
        start_idx = next(i for i, name in enumerate(fields) if name.lower() == "start")
        end_idx = next(i for i, name in enumerate(fields) if name.lower() == "end")
    except StopIteration:
        start_idx, end_idx = 1, 2

    blocks: list[Block] = []
    raw_buf: list[str] = []

    def flush_raw() -> None:
        if raw_buf:
            blocks.append(Block(kind="raw", raw="\n".join(raw_buf)))
            raw_buf.clear()

    for line in lines:
        if line.startswith("Dialogue:"):
            payload = line.split(":", 1)[1]
            if payload.startswith(" "):
                payload = payload[1:]
            parts = _split_fields(payload, text_idx + 1)
            if len(parts) <= text_idx:
                raw_buf.append(line)
                continue
            flush_raw()
            text = parts[text_idx]
            start = parts[start_idx] if start_idx < len(parts) else ""
            end = parts[end_idx] if end_idx < len(parts) else ""
            prefix_parts = parts[:text_idx]
            leading = ""
            match = _OVERRIDE_RE.match(text)
            if match:
                leading = match.group(0)
                text = text[len(leading) :]
            blocks.append(
                Block(
                    kind="cue",
                    cue=Cue(
                        index=None,
                        timing=f"{start} --> {end}",
                        text=text.replace("\\N", "\n").replace("\\n", "\n"),
                        extra={
                            "prefix": ",".join(prefix_parts),
                            "leading": leading,
                            "use_n": "\\N" in parts[text_idx] or "\\n" in parts[text_idx],
                        },
                    ),
                )
            )
        else:
            raw_buf.append(line)
    flush_raw()
    return Document(format="ass", blocks=blocks, newline=newline)


def dumps_ass(doc: Document) -> str:
    lines: list[str] = []
    for block in doc.blocks:
        if block.kind == "raw":
            lines.append(block.raw)
            continue
        assert block.cue is not None
        text = block.cue.text
        if block.cue.extra.get("use_n"):
            text = text.replace("\n", "\\N")
        leading = block.cue.extra.get("leading", "")
        prefix = block.cue.extra.get("prefix", "")
        lines.append(f"Dialogue: {prefix},{leading}{text}")
    text = "\n".join(lines)
    if text and not text.endswith("\n"):
        text += "\n"
    if doc.newline != "\n":
        text = text.replace("\n", doc.newline)
    return text
