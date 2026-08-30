"""Glossary placeholders for NLLB (no prompt) and Whisper canonicalization.

NLLB has no prompt, so proper names must be pulled out of the source string
before translation. Injecting Latin (Drum, Nozaki) into Japanese is how
ドラム became tambor — the model then "translates" the English word.

GPU path: send the original Japanese sentence (バンコク stays inside
バンコクに飛んだ). After MT, ``overlay_names`` writes Latin over surviving
JP keys, known aliases (tambor→Drum, Nagasaki→Nozaki), and moon/luna
when バンコク was in the source. ``protect`` / ``GLS{n}`` stay for
unit tests and the no-kanji local skip — they are not sent to NLLB.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

# Copy-stable MT token. Also accept spaced/cased mutations and XML retry tags.
_GLS_RE = re.compile(r"GLS\s*(\d+)", re.IGNORECASE)
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
    return f"GLS{n}"


def _xml_sentinel(n: int) -> str:
    return f"<g{n}>"


def canonicalize_sentinels(text: str, pair_count: int) -> str:
    """Normalize ``GLS 0`` / ``gls0`` / ``<g0>`` to ``GLS{n}`` when in range."""

    def _canon(match: re.Match[str]) -> str:
        n = int(match.group(1))
        if 0 <= n < pair_count:
            return _sentinel(n)
        return match.group(0)

    out = _XML_RE.sub(_canon, text)
    return _GLS_RE.sub(_canon, out)


def _strip_sentinels(text: str) -> str:
    return _GLS_RE.sub("", _XML_RE.sub("", text))


def needs_nllb(protected: str) -> bool:
    """True iff a CJK ideograph remains after stripping GLS / ``<gN>`` tokens."""
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
    """Longest-first JP → ``GLS{n}`` protection and restore."""

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
        """Replace each JP key with spaced ``GLS{n}``. Never inject Latin.

        ``バンコクに飛んだ`` → ``GLS0 に飛んだ`` (に飛んだ stays in the same
        string). ``おい ドラム 東条 …`` → ``おい GLS0 GLS1 …``.
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
                token = f" {_sentinel(len(pairs))} "
                out = out[:idx] + token + out[idx + len(key) :]
                pairs.append((key, value))
                start = idx + len(token)
        out = re.sub(r"[^\S\n]+", " ", out)
        out = re.sub(r" *\n *", "\n", out)
        return out.strip(), pairs

    def to_xml(self, protected: str, pair_count: int) -> str:
        """Retry form: ``GLS{n}`` → ``<g{n}>``."""
        canon = canonicalize_sentinels(protected, pair_count)
        for i in range(pair_count):
            canon = canon.replace(_sentinel(i), _xml_sentinel(i))
        return canon

    def missing_sentinels(self, text: str, pairs: list[tuple[str, str]]) -> list[int]:
        canon = canonicalize_sentinels(text, len(pairs))
        return [i for i in range(len(pairs)) if _sentinel(i) not in canon]

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

    def overlay_names(self, source: str, mt: str) -> str:
        """Write Latin names onto an MT string. Never used as an NLLB input.

        Surviving JP keys → Latin (longest-first). Known aliases
        (tambor→Drum, Nagasaki→Nozaki). If バンコク was in ``source`` and
        Bangkok is missing, ``the moon`` / ``la luna`` become Bangkok.
        Any required Latin still missing is prepended.
        """
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
