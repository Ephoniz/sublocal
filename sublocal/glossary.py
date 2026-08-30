"""File-local names for GemmaX2: in-source Hepburn (Person only).

The SRT string stays immutable. A *copy* of the cue gets longest-first
Person replacements (``野崎をマーク`` → ``Nozakiをマーク``) before the
official prompt. Opaque ``xxNxx`` sentinels are not used on the MT path —
GemmaX2 drops them or emits only those tokens.

``protect`` / ``restore`` remain for Whisper ASR canonicalize (Japanese keys
back), not for GemmaX2.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from sublocal.extract import (
    NON_PERSON_KEYS,
    extract_speakers,
    is_generic_noun,
    is_in_source_person_key,
)

# Mutations a model sometimes emits around the opaque sentinel (ASR path).
_SENTINEL_MUTATION_RE = re.compile(r"xx\s*(\d+)\s*xx", re.IGNORECASE)
_EXACT_SENTINEL_RE = re.compile(r"xx(\d+)xx", re.IGNORECASE)
_XML_RE = re.compile(r"<\s*g\s*(\d+)\s*>", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_JP_ANY_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff\uf900-\ufaff]")
_LEFTOVER_FILLER_RE = re.compile(
    r"[0-9０-９\s→\-–—\.。、，！？!?…《》【】（）()＜＞「」『』・/／]"
)
_LEFTOVER_PARTICLE_RE = re.compile(r"[はがをにでのとへもようおい]")
_NAMES_ONLY_PUNCT_RE = re.compile(
    r"[→\-–—\(\)\[\]《》【】<>\"'\s\.,;:!?。、，＋+…/／]"
)
_SPEAKER_RE = re.compile(r"^[《＜「『]*[（(]([^）)]+)[）)]\s*")
_LATIN_NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'.-]*")
_WRAPPERS = "《》＜＞「」『』"
_PARTICLES = ("に", "が", "を", "は", "の", "と", "へ", "おい", "よう")
_BRACKETS = "《》＜＞「」"
# Closed post-MT aliases. Do not prefix missing names / rewrite the cue.
_CLOSED_ALIASES = (("nagasaki", "Nozaki"),)
_JP_VERB_RE = re.compile(
    r"マーク|していた|している|した|する|して|だった|である|です|ます"
)
MIN_SENTENCE_JP_CHARS = 4


class GlossaryError(RuntimeError):
    """A glossary mapping file is invalid (or an ASR sentinel was lost)."""


def _sentinel(n: int) -> str:
    return f"xx{n}xx"


def canonicalize_sentinels(text: str, pair_count: int) -> str:
    """Normalize ``xx 0 xx`` / ``XX0XX`` / ``<g0>`` to ``xx{n}xx`` (ASR path)."""

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


def leftover_has_jp_content(text: str) -> bool:
    """False when leftover is only particles, digits, arrows, punct."""
    stripped = _strip_sentinels(text)
    stripped = _LEFTOVER_FILLER_RE.sub("", stripped)
    stripped = _LEFTOVER_PARTICLE_RE.sub("", stripped)
    return _JP_ANY_RE.search(stripped) is not None


def leftover_jp_char_count(text: str) -> int:
    """Count leftover Japanese letters; spaces/punct are ignored."""
    return len(_JP_ANY_RE.findall(text))


def peel_leading_speakers(text: str, person_keys: Iterable[str]) -> str:
    """Strip a full-cue 《》/【】 wrap, then leading Person ``（Name）`` tags.

    ``《（野崎）この中に…》`` → ``この中に…``. Does not delete a dialogue body.
    """
    rest = text.strip()
    if len(rest) >= 2:
        pairs = (("《", "》"), ("【", "】"))
        for opener, closer in pairs:
            if rest.startswith(opener) and rest.endswith(closer) and rest.count(opener) == 1:
                rest = rest[len(opener) : -len(closer)].strip()
                break
    keys = {k for k in person_keys if k}
    while True:
        match = _SPEAKER_RE.match(rest)
        if not match:
            break
        inner = match.group(1).strip()
        if keys and inner not in keys and not is_in_source_person_key(inner):
            break
        rest = rest[match.end() :].strip()
    return rest


def leftover_after_speakers_and_persons(
    text: str, person_keys: Iterable[str]
) -> str:
    rest = peel_leading_speakers(text, person_keys)
    for key in sorted({k for k in person_keys if k}, key=len, reverse=True):
        rest = rest.replace(key, "")
    return rest


def has_verb(text: str, tokens: list[str], person_latins: set[str]) -> bool:
    """True when leftover JP looks like a verb, or a Latin token is not a name."""
    if _JP_VERB_RE.search(text):
        return True
    allowed = {v.lower() for v in person_latins if v}
    return any(tok.lower() not in allowed for tok in tokens)


def is_names_only_output(
    text: str, person_latins: Iterable[str] | None = None
) -> bool:
    """Empty, or leftover JP gone and every Latin token is a Person name.

    ``(Sano)``, ``Nozaki``, ``(Sano) Nozaki`` are True when those strings are
    in the Person set. A sentence that happens to contain Nozaki is False.
    A name pile is True (caller must retry / emit ``src_original``, never keep it).
    """
    raw = text.strip()
    if not raw:
        return True
    if leftover_has_jp_content(raw):
        return False
    cleaned = _SENTINEL_MUTATION_RE.sub(" ", raw)
    cleaned = _XML_RE.sub(" ", cleaned)
    cleaned = _NAMES_ONLY_PUNCT_RE.sub(" ", cleaned)
    tokens = _LATIN_NAME_TOKEN_RE.findall(cleaned)
    if not tokens:
        return True
    allowed = {v.lower() for v in (person_latins or ()) if v}
    if not allowed:
        return False
    if has_verb(raw, tokens, allowed):
        return False
    return all(tok.lower() in allowed for tok in tokens)


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


def infer_person_keys(mapping: dict[str, str]) -> set[str]:
    """Kanji/Latin Person-like keys. Not 別班, not generics, not katakana places."""
    return {key for key in mapping if is_in_source_person_key(key)}


class Glossary:
    """Longest-first Person Hepburn in a source *copy*. Overlay is closed-set."""

    def __init__(
        self,
        mapping: dict[str, str],
        person_keys: Iterable[str] | None = None,
    ) -> None:
        if not mapping:
            raise GlossaryError("Glossary mapping is empty")
        self.mapping = dict(mapping)
        self.entries = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
        if person_keys is None:
            self.person_keys = infer_person_keys(self.mapping)
        else:
            self.person_keys = {
                key
                for key in person_keys
                if key in self.mapping and is_in_source_person_key(key)
            }

    @classmethod
    def load(cls, path: str | Path) -> Glossary:
        mapping = load_mapping(path)
        return cls(mapping, person_keys=infer_person_keys(mapping))

    def source_keys(self) -> list[str]:
        """JP keys in file order (Whisper ``initial_prompt``)."""
        return list(self.mapping.keys())

    def whisper_prompt(self) -> str:
        return " ".join(self.source_keys())

    def person_latins(self) -> set[str]:
        return {self.mapping[key] for key in self.person_keys if key in self.mapping}

    def hepburn_in_source(self, text: str) -> str:
        """Copy ``text`` and replace Person surfaces longest-first with Latin.

        ``野崎をマーク`` → ``Nozakiをマーク``. Does not touch ``src_original``.
        Does not romanize non-Person (別班 stays Japanese).
        """
        out = text
        for key, value in self.entries:
            if not key or key not in self.person_keys:
                continue
            if key in NON_PERSON_KEYS or is_generic_noun(key):
                continue
            out = out.replace(key, value)
        return out

    def speaker_tag_if_no_sentence(self, text: str) -> str | None:
        """If leftover JP after speakers+Person is < 4 chars, emit a tag.

        ``（佐野）`` → ``(Sano)``. Bare ``野崎`` → ``Nozaki``.
        ``野崎をマークしていた`` is a sentence (verb) and returns None.
        """
        speakers = extract_speakers(text, confirmed=self.person_keys)
        leftover = leftover_after_speakers_and_persons(text, self.person_keys)
        stripped_name = bool(speakers) or any(key in text for key in self.person_keys)
        if not stripped_name:
            return None
        if leftover_jp_char_count(leftover) >= MIN_SENTENCE_JP_CHARS:
            return None
        if _JP_VERB_RE.search(leftover):
            return None
        tags: list[str] = []
        seen: set[str] = set()
        for jp in speakers:
            if jp not in self.mapping:
                continue
            latin = self.mapping[jp]
            token = f"({latin})"
            if token not in seen:
                tags.append(token)
                seen.add(token)
        rest = peel_leading_speakers(text, self.person_keys)
        for key, value in self.entries:
            if key not in self.person_keys or key not in rest:
                continue
            token = value if not speakers else f"({value})"
            if speakers and value not in seen and f"({value})" not in seen:
                tags.append(value)
                seen.add(value)
            elif not speakers and value not in seen:
                tags.append(value)
                seen.add(value)
        if not tags:
            return None
        return " ".join(tags)

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
        """ASR-only: replace source keys with ``xx{n}xx``. Not the GemmaX2 path."""
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
        """ASR leftover. Not used on the GemmaX2 path."""

        def _pad(match: re.Match[str]) -> str:
            start, end = match.span()
            token = _sentinel(int(match.group(1)))
            left = "" if start == 0 or protected[start - 1].isspace() else " "
            right = "" if end == len(protected) or protected[end].isspace() else " "
            return f"{left}{token}{right}"

        return _EXACT_SENTINEL_RE.sub(_pad, protected)

    def to_xml(self, protected: str, pair_count: int) -> str:
        """ASR leftover. Not used on the GemmaX2 path."""
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
        """Restore sentinels that are present (ASR path)."""
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
        """Put sentinels back (ASR path). ``target='key'`` → Japanese."""
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
        """Closed-set alias rewrite only. Never prefix a name pile.

        If source had 野崎 and target has Nagasaki → Nozaki. Does not rewrite
        the rest of the cue.
        """
        out = mt
        required: set[str] = set()
        for key, value in self.entries:
            if key in source:
                required.add(value)
        for alias, latin in _CLOSED_ALIASES:
            if latin not in required:
                continue
            out = re.sub(alias, latin, out, flags=re.IGNORECASE)
        return out

    def cleanup_adjacent(self, text: str, latin_values: list[str]) -> str:
        """Strip leftover JP particles/brackets next to Latin names (unused on MT)."""
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
