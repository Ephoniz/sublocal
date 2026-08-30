"""Longest-first xxNxx protect-restore (original v0.3).

Names are pulled out of the source string before MT. Injecting Latin
(Drum, Nozaki) into Japanese is how ドラム became tambor — the model then
"translates" the English word.

Algorithm: longest-first replace of source keys with opaque ASCII sentinels
``xx{n}xx``, send the protected cue through the official GemmaX2 prompt,
restore by exact match. A missing sentinel fails that cue only
(retry padded/XML, then overlay leftover JP keys) — never the document.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

# Mutations a model sometimes emits around the opaque sentinel.
_SENTINEL_MUTATION_RE = re.compile(r"xx\s*(\d+)\s*xx", re.IGNORECASE)
_EXACT_SENTINEL_RE = re.compile(r"xx(\d+)xx", re.IGNORECASE)
_XML_RE = re.compile(r"<\s*g\s*(\d+)\s*>", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_SPEAKER_RE = re.compile(r"^[《＜「『]*[（(]([^）)]+)[）)]\s*")
_WRAPPERS = "《》＜＞「」『』"
_PARTICLES = ("に", "が", "を", "は", "の", "と", "へ", "おい", "よう")
_BRACKETS = "《》＜＞「」"
_ALIASES = (
    ("tambor", "Drum"),
    ("nagasaki", "Nozaki"),
)
_MOON_RE = re.compile(r"\bthe moon\b|\bla luna\b", re.IGNORECASE)


class GlossaryError(RuntimeError):
    """A glossary sentinel was lost or the mapping file is invalid."""


def _sentinel(n: int) -> str:
    return f"xx{n}xx"


def canonicalize_sentinels(text: str, pair_count: int) -> str:
    """Normalize ``xx 0 xx`` / ``XX0XX`` / ``<g0>`` to ``xx{n}xx``."""

    def _canon(match: re.Match[str]) -> str:
        n = int(match.group(1))
        if 0 <= n < pair_count:
            return _sentinel(n)
        return match.group(0)

    out = _XML_RE.sub(_canon, text)
    return _SENTINEL_MUTATION_RE.sub(_canon, out)


def _strip_sentinels(text: str) -> str:
    return _SENTINEL_MUTATION_RE.sub("", text)


def needs_nllb(protected: str) -> bool:
    """True iff a CJK ideograph remains after stripping ``xxNxx`` tokens."""
    return _CJK_RE.search(_strip_sentinels(protected)) is not None


def has_cjk(text: str) -> bool:
    return _CJK_RE.search(text) is not None


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
    """Longest-first source → ``xx{n}xx`` protection and restore."""

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

    def peel_speaker(self, text: str) -> tuple[str | None, str]:
        """If the cue starts with optional 《＜「『 then （glossary-name）.

        Returns ``(jp_name, rest)``. Rest has leftover wrapper brackets stripped.
        """
        match = _SPEAKER_RE.match(text)
        if not match:
            return None, text
        inner = match.group(1).strip()
        if inner not in self.mapping:
            return None, text
        rest = text[match.end() :].strip().strip(_WRAPPERS).strip()
        return inner, rest

    def protect(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """Replace each source key with ``xx{n}xx``. Never inject Latin values.

        Keys are applied longest-first so ウラジオストク is not eaten by a
        shorter substring. ``野崎です`` → ``xx0xxです``.
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

    def pad_sentinels(self, protected: str) -> str:
        """``いいかxx0xx`` → ``いいか xx0xx `` so a glued name is less likely dropped."""

        def _pad(match: re.Match[str]) -> str:
            start, end = match.span()
            token = _sentinel(int(match.group(1)))
            left = "" if start == 0 or protected[start - 1].isspace() else " "
            right = "" if end == len(protected) or protected[end].isspace() else " "
            return f"{left}{token}{right}"

        return _EXACT_SENTINEL_RE.sub(_pad, protected)

    def to_xml(self, protected: str, pair_count: int) -> str:
        """Retry form: ``xx{n}xx`` → ``<g{n}>``."""
        canon = canonicalize_sentinels(protected, pair_count)
        for i in range(pair_count):
            canon = canon.replace(_sentinel(i), f"<g{i}>")
        return canon

    def missing_sentinels(self, text: str, pairs: list[tuple[str, str]]) -> list[int]:
        canon = canonicalize_sentinels(text, len(pairs))
        return [i for i in range(len(pairs)) if _sentinel(i) not in canon]

    def restore_surviving(
        self,
        text: str,
        pairs: list[tuple[str, str]],
        *,
        target: Literal["value", "key"] = "value",
    ) -> tuple[str, list[int]]:
        """Restore sentinels that are present. Return ``(text, missing_indices)``."""
        out = canonicalize_sentinels(text, len(pairs))
        missing: list[int] = []
        for i, (key, value) in enumerate(pairs):
            token = _sentinel(i)
            if token not in out:
                missing.append(i)
                continue
            replacement = key if target == "key" else value
            out = out.replace(token, replacement)
        return out, missing

    def restore(
        self,
        text: str,
        pairs: list[tuple[str, str]],
        *,
        target: Literal["value", "key"] = "value",
    ) -> str:
        """Put sentinels back. ``target='value'`` → Latin; ``'key'`` → JP.

        Exact ``xx{n}xx`` after light canonicalization. Fail if a sentinel
        is missing — do not silently drop names.
        """
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

    def overlay_names(self, source: str, mt: str) -> str:
        """Legacy NLLB overlay. Not used on the GemmaX2 protect-restore path."""
        required: list[str] = []
        out = mt
        for key, value in self.entries:
            if key not in source:
                continue
            required.append(value)
            if key in out:
                out = out.replace(key, value)
        required_set = set(required)
        for alias, latin in _ALIASES:
            if latin not in required_set:
                continue
            out = re.sub(re.escape(alias), latin, out, flags=re.IGNORECASE)
        if "Bangkok" in required_set and "Bangkok" not in out:
            out = _MOON_RE.sub("Bangkok", out)
        missing = [value for value in required if value not in out]
        if missing:
            prefix = " ".join(missing)
            out = f"{prefix} {out}".strip() if out.strip() else prefix
        return out

    def cleanup_adjacent(self, text: str, latin_values: list[str]) -> str:
        """Strip leftover JP particles/brackets next to restored Latin names."""
        if not latin_values:
            return text
        particle = "|".join(re.escape(p) for p in sorted(_PARTICLES, key=len, reverse=True))
        brackets = re.escape(_BRACKETS)
        out = text
        for latin in sorted(set(latin_values), key=len, reverse=True):
            if not latin:
                continue
            name = re.escape(latin)
            out = re.sub(rf"({name})\s*(?:{particle})", r"\1", out)
            out = re.sub(rf"(?:{particle})\s*({name})", r"\1", out)
            out = re.sub(rf"[{brackets}]\s*({name})", r"\1", out)
            out = re.sub(rf"({name})\s*[{brackets}]", r"\1", out)
        out = re.sub(r"[^\S\n]+", " ", out)
        return out.strip()


def as_glossary(glossary: Glossary | str | Path | None) -> Glossary | None:
    if glossary is None:
        return None
    if isinstance(glossary, Glossary):
        return glossary
    return Glossary.load(glossary)
