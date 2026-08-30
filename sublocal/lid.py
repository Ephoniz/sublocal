"""Per-cue language ID and Latin/ASCII name copy-through.

Script heuristic is always local. Short Latin cues optionally use lingua-py
(``pip install .[lid]``). No cloud API.
"""

from __future__ import annotations

import re

from sublocal.languages import UnknownLanguageError, to_flores

# Hiragana / Katakana (incl. small kana, prolonged sound mark).
_HIRAGANA_KATAKANA = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_HANGUL = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")
_LATIN = re.compile(r"[A-Za-z\u00c0-\u024f]")
# Names already in the cue — do not send these through NLLB.
LATIN_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9'.-]*")

SHORT_CUE_CHARS = 24

_lingua_detector = None


def script_heuristic(text: str) -> str | None:
    """Hiragana/Katakana/CJK → ja, Hangul → ko, Latin → en."""
    if _HIRAGANA_KATAKANA.search(text):
        return "ja"
    if _CJK.search(text):
        return "ja"
    if _HANGUL.search(text):
        return "ko"
    if _LATIN.search(text):
        return "en"
    return None


def lingua_detect(text: str) -> str | None:
    """Local lingua-py LID. None if the extra is not installed."""
    global _lingua_detector
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError:
        return None
    if _lingua_detector is None:
        _lingua_detector = LanguageDetectorBuilder.from_languages(
            Language.JAPANESE,
            Language.ENGLISH,
            Language.SPANISH,
            Language.KOREAN,
        ).build()
    found = _lingua_detector.detect_language_of(text)
    if found is None:
        return None
    code = getattr(found, "iso_code_639_1", None)
    name = getattr(code, "name", None) if code is not None else None
    if not name:
        return None
    return str(name).lower()


def detect_cue_lang(text: str) -> str | None:
    """Per-cue LID: script first; short Latin uses lingua-py when installed."""
    script = script_heuristic(text)
    stripped = text.strip()
    if script in {"ja", "ko"}:
        return script
    if script == "en" and len(stripped) <= SHORT_CUE_CHARS:
        lid = lingua_detect(stripped)
        if lid:
            return lid
    return script


def same_language(a: str, b: str) -> bool:
    try:
        return to_flores(a) == to_flores(b)
    except UnknownLanguageError:
        return a.strip().lower() == b.strip().lower()


def is_latin_script(code: str) -> bool:
    try:
        return to_flores(code).endswith("_Latn")
    except UnknownLanguageError:
        return False


def protect_latin_names(text: str) -> tuple[str, list[str]]:
    """Replace Latin/ASCII tokens with ``ZX{n}`` so NLLB never sees them."""
    names: list[str] = []

    def _repl(match: re.Match[str]) -> str:
        names.append(match.group(0))
        return f" ZX{len(names) - 1} "

    return LATIN_NAME_RE.sub(_repl, text), names


def restore_latin_names(text: str, names: list[str]) -> str:
    """Put copy-through names back. Missing names are appended, not invented."""
    out = text
    for i, name in enumerate(names):
        token = f"ZX{i}"
        if token in out:
            out = out.replace(token, name)
        elif token.lower() in out.lower():
            out = re.sub(re.escape(token), name, out, flags=re.IGNORECASE)
        elif name not in out:
            out = f"{out} {name}".strip() if out.strip() else name
    out = re.sub(r"[^\S\n]+", " ", out)
    return out.strip()
