"""Glossary placeholders for NLLB (no prompt) and Whisper canonicalization.

NLLB has no prompt, so proper names must be pulled out of the source string
before translation. Injecting Latin (Drum, Nozaki) into Japanese is how
ドラム became tambor — the model then "translates" the English word.

Algorithm: longest-first replace of JP keys with opaque ASCII sentinels
``xx{n}xx``, translate the protected strings, restore sentinels by exact
match. A missing sentinel fails the cue; names are never silently dropped.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

# Mutations NLLB sometimes emits around the opaque sentinel.
_SENTINEL_MUTATION_RE = re.compile(r"xx\s*(\d+)\s*xx", re.IGNORECASE)

# Whitespace, speaker-tag brackets, quotes, ellipsis, and JP/ASCII punctuation.
_SPLIT_RE = re.compile(
    r"[\s"
    r"（）()＜＞《》〈〉【】\[\]『』「」"
    r"\"'“”‘’«»"
    r"…⋯・。．.!?！？ー\-~,~～,，]"
    r"+"
)

# Groans: うっ+ / あっ+ / えっ+ (also ううっ = う+っ+), ああ+ / ええ+ / ん+ / だっ+ / ヒー+.
# ー is also a split char, so ヒー collapses to ヒ after tokenize.
_GROAN_RE = re.compile(r"^(?:う+っ+|あ+っ+|え+っ+|ああ+|ええ+|ん+|だっ+|ヒー+|ヒ)$")


class GlossaryError(RuntimeError):
    """A glossary sentinel was lost or the mapping file is invalid."""


def _sentinel(n: int) -> str:
    return f"xx{n}xx"


def canonicalize_sentinels(text: str, pair_count: int) -> str:
    """Normalize ``xx 0 xx`` / ``XX0XX`` to ``xx{n}xx`` when ``n`` is in range."""

    def _canon(match: re.Match[str]) -> str:
        n = int(match.group(1))
        if 0 <= n < pair_count:
            return _sentinel(n)
        return match.group(0)

    return _SENTINEL_MUTATION_RE.sub(_canon, text)


def needs_nllb(protected: str) -> bool:
    """True if leftover tokens after stripping sentinels are real dialogue.

    Groan-only remainders (うっ / あっ / …) are False so speaker-tag cries
    are restored locally instead of being sent to NLLB, which drops ``xxNxx``.
    Particles such as の / と still return True.
    """
    stripped = _SENTINEL_MUTATION_RE.sub("", protected)
    for token in _SPLIT_RE.split(stripped):
        if not token:
            continue
        if _GROAN_RE.fullmatch(token):
            continue
        return True
    return False


def load_mapping(path: str | Path) -> dict[str, str]:
    """Load a flat ``source_jp: target_latin`` YAML object (UTF-8)."""
    text = Path(path).read_text(encoding="utf-8")
    mapping: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise GlossaryError(f"Invalid glossary line {lineno}: {raw!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            raise GlossaryError(f"Empty glossary key on line {lineno}")
        mapping[key] = value
    if not mapping:
        raise GlossaryError(f"Glossary is empty: {path}")
    return mapping


class Glossary:
    """Longest-first JP → sentinel protection and restore."""

    def __init__(self, mapping: dict[str, str]) -> None:
        if not mapping:
            raise GlossaryError("Glossary mapping is empty")
        self.mapping = dict(mapping)
        self.entries = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)

    @classmethod
    def load(cls, path: str | Path) -> Glossary:
        return cls(load_mapping(path))

    def source_keys(self) -> list[str]:
        """JP keys in file order (Whisper ``initial_prompt``)."""
        return list(self.mapping.keys())

    def whisper_prompt(self) -> str:
        return " ".join(self.source_keys())

    def protect(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """Replace each JP key with ``xx{n}xx``. Never inject Latin values.

        Returns ``(protected_text, [(source_jp, target_latin), ...])`` in
        replacement order. Keys are applied longest-first so
        ウラジオストク is not eaten by a shorter substring.
        """
        pairs: list[tuple[str, str]] = []
        out = text
        for key, value in self.entries:
            if not key:
                continue
            start = 0
            while True:
                idx = out.find(key, start)
                if idx < 0:
                    break
                token = _sentinel(len(pairs))
                out = out[:idx] + token + out[idx + len(key) :]
                pairs.append((key, value))
                start = idx + len(token)
        return out, pairs

    def restore(
        self,
        text: str,
        pairs: list[tuple[str, str]],
        *,
        target: Literal["value", "key"] = "value",
    ) -> str:
        """Put sentinels back. ``target='value'`` → Latin; ``'key'`` → JP."""
        out = canonicalize_sentinels(text, len(pairs))
        for i, (key, value) in enumerate(pairs):
            token = _sentinel(i)
            if token not in out:
                raise GlossaryError(
                    f"Glossary sentinel {token!r} missing after translation "
                    f"(expected {key!r} → {value!r})"
                )
            replacement = key if target == "key" else value
            out = out.replace(token, replacement)
        return out


def as_glossary(glossary: Glossary | str | Path | None) -> Glossary | None:
    if glossary is None:
        return None
    if isinstance(glossary, Glossary):
        return glossary
    return Glossary.load(glossary)
