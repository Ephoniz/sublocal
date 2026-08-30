"""CPU pass-1 glossary from the current subtitle file.

GiNZA Person/Place on Japanese cues, speaker brackets, and Latin already in
non-Latin cues. Romanize JP spans with pykakasi Hepburn. No GemmaX2 NER,
no default drama.yml, no ja_ginza_electra (that model needs torch + VRAM).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version

from sublocal.lid import LATIN_NAME_RE, is_latin_script, script_heuristic

# ja_ginza NER labels (spaCy 3.8 + ginza 5.2). Not PERSON/GPE.
# Government (公安) is real but must not enter the person glossary.
GINZA_LABELS = frozenset({"Person", "Place", "N_Person"})
GINZA_MODEL = "ja_ginza"
# ja_ginza 5.2.0: bare spacy.load("ja_ginza") raises ConfigValidationError
# because compound_splitter.split_mode is None.
GINZA_LOAD_CONFIG = {
    "components": {"compound_splitter": {"split_mode": "A"}}
}

_SPEAKER_GUILLEMET_RE = re.compile(r"《([^》]+)》")
_SPEAKER_LENTICULAR_RE = re.compile(r"【([^】]+)】")
_SPEAKER_START_PAREN_RE = re.compile(r"^[《＜「『]*[（(]([^）)]+)[）)]")

_JP_SCRIPT = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff\uf900-\ufaff]")


def is_latin_token(text: str) -> bool:
    """True when the span is already Latin/ASCII (copy-through; skip romanize)."""
    stripped = text.strip()
    if not stripped or _JP_SCRIPT.search(stripped):
        return False
    return LATIN_NAME_RE.fullmatch(stripped) is not None


def pykakasi_version() -> str:
    """pykakasi 2.3.0 has no ``__version__``; read the package metadata."""
    try:
        return version("pykakasi")
    except PackageNotFoundError:
        return "unknown"


def spacy_load_ja_ginza():
    """``spacy.load("ja_ginza")`` plus the ginza 5.2.0 split_mode A config."""
    import spacy

    # Bare load raises ConfigValidationError: compound_splitter.split_mode is None.
    return spacy.load(GINZA_MODEL, config=GINZA_LOAD_CONFIG)


def load_ginza():
    """Load CPU ``ja_ginza`` with split_mode A. Never ja_ginza_electra."""
    try:
        # CPU only. Extra VRAM 0. ja_ginza_electra needs torch + ~16GB.
        return spacy_load_ja_ginza()
    except Exception:
        return None


def romanize_hepburn(text: str) -> str:
    """Hepburn romanization, first letter capitalized (野崎 → Nozaki)."""
    stripped = text.strip()
    if not stripped or is_latin_token(stripped):
        return stripped
    try:
        import pykakasi
    except ImportError:
        return stripped
    converter = pykakasi.kakasi()
    parts = converter.convert(stripped)
    roman = "".join(
        str(part.get("hepburn") or part.get("passport") or "") for part in parts
    ).strip()
    if not roman:
        return stripped
    return roman[0].upper() + roman[1:]


def _name_sized(span: str) -> bool:
    """Reject full-line 《dialogue》 wraps; keep 《野崎》 / 【佐野】."""
    stripped = span.strip()
    if not stripped or len(stripped) > 16:
        return False
    if any(mark in stripped for mark in ("。", "？", "?", "！", "!")):
        return False
    return True


def _speaker_candidates(span: str) -> list[str]:
    """Prefer inner （name）; otherwise a short 《name》 / 【name】 span."""
    match = _SPEAKER_START_PAREN_RE.match(span.strip())
    if match:
        return [match.group(1).strip()]
    if _name_sized(span):
        return [span.strip()]
    return []


def extract_speakers(text: str) -> list[str]:
    """《name》, 【name】, and leading （name） / (name)."""
    found: list[str] = []
    for span in _SPEAKER_GUILLEMET_RE.findall(text):
        found.extend(_speaker_candidates(span))
    for span in _SPEAKER_LENTICULAR_RE.findall(text):
        found.extend(_speaker_candidates(span))
    match = _SPEAKER_START_PAREN_RE.match(text)
    if match:
        found.append(match.group(1).strip())
    return [span for span in found if span]


def extract_latin(text: str) -> list[str]:
    return LATIN_NAME_RE.findall(text)


def extract_ginza_ents(text: str, nlp: object | None) -> list[str]:
    if nlp is None:
        return []
    doc = nlp(text)  # type: ignore[operator]
    ents = getattr(doc, "ents", ())
    return [
        str(ent.text).strip()
        for ent in ents
        if getattr(ent, "label_", None) in GINZA_LABELS and str(ent.text).strip()
    ]


def _add_span(mapping: dict[str, str], span: str) -> None:
    key = span.strip()
    if not key:
        return
    if key in mapping:
        return
    if is_latin_token(key):
        mapping[key] = key
        return
    mapping[key] = romanize_hepburn(key)


def extract_file_glossary(
    texts: Iterable[str],
    langs: Iterable[str] | None = None,
    *,
    nlp: object | None = None,
    load: bool = True,
) -> tuple[dict[str, str], int]:
    """Build an in-memory source→Latin map from this file only.

    Returns ``(mapping, ginza_entity_count)``. Latin tokens are taken only
    from non-Latin-script cues (v0.4 copy-through). GiNZA runs on Japanese.
    """
    if nlp is None and load:
        nlp = load_ginza()
    lang_list = list(langs) if langs is not None else []
    mapping: dict[str, str] = {}
    ginza_count = 0
    for i, text in enumerate(texts):
        lang = lang_list[i] if i < len(lang_list) else ""
        script = script_heuristic(text)
        japanese = script == "ja" or (
            bool(lang) and not is_latin_script(lang) and script != "ko"
        )
        non_latin_cue = (script in {"ja", "ko"}) or (
            bool(lang) and not is_latin_script(lang)
        )
        if japanese:
            ents = extract_ginza_ents(text, nlp)
            ginza_count += len(ents)
            for span in ents:
                _add_span(mapping, span)
        # Speakers and Latin copy-through are for JA/KO cues, not (suspiro) in ES.
        if non_latin_cue:
            for span in extract_speakers(text):
                _add_span(mapping, span)
            for span in extract_latin(text):
                mapping.setdefault(span, span)
    return mapping, ginza_count


def merge_mappings(*maps: dict[str, str] | None) -> dict[str, str]:
    """Later maps win on key conflict. Glossary applies longest-first at protect."""
    out: dict[str, str] = {}
    for mapping in maps:
        if mapping:
            out.update(mapping)
    return out
