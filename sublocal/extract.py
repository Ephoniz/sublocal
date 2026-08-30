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
# Clause particles that mean the span is not a bare name (野崎は / 野崎をマネて).
_NAME_PARTICLES = ("は", "が", "を", "に", "で")
_HONORIFICS = ("さん", "くん", "ちゃん", "様")
_TRAILING_JUNK = "）)」。.、， "
MAX_NAME_KEY_CHARS = 12
# Generic nouns / orgs / stage directions — never glossary keys.
GENERIC_NAME_KEYS = frozenset(
    {
        "諜報機関",
        "撮影開始",
        "公安",
        "政府",
        "機関",
        "開始",
    }
)
_GENERIC_SUFFIXES = ("機関", "開始", "公安部")
_KANJI_NAME = re.compile(r"^[\u3400-\u9fff]{2,4}$")
_KATAKANA_NAME = re.compile(r"^[\u30a0-\u30ffー]{2,8}$")


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


def clean_glossary_key(span: str) -> str:
    """Strip wrapping junk (trailing ``）`` / ``。``) so ``佐野）`` → ``佐野``."""
    return span.strip().strip(_TRAILING_JUNK)


def is_generic_noun(key: str) -> bool:
    """諜報機関 / 撮影開始 / 公安 and similar orgs — not person names."""
    if key in GENERIC_NAME_KEYS or "公安" in key:
        return True
    return any(key.endswith(suffix) for suffix in _GENERIC_SUFFIXES)


def is_person_place_surface(key: str, confirmed: Iterable[str] | None = None) -> bool:
    """佐野 yes, 撮影開始 no. Confirmed GiNZA Person/Place always wins."""
    if not key:
        return False
    if confirmed is not None and key in set(confirmed):
        return True
    if is_generic_noun(key):
        return False
    return bool(_KANJI_NAME.fullmatch(key) or _KATAKANA_NAME.fullmatch(key))


def is_acceptable_name_key(key: str) -> bool:
    """Keep Person/Place surfaces. Drop clause-length, generics, particle junk."""
    if not key:
        return False
    if is_latin_token(key):
        return True
    if is_generic_noun(key):
        return False
    if key in _HONORIFICS:
        return False
    if any(key.endswith(suffix) for suffix in _HONORIFICS):
        return False
    if any(particle in key for particle in _NAME_PARTICLES):
        return False
    if len(key) > MAX_NAME_KEY_CHARS:
        return False
    if any(mark in key for mark in ("。", "？", "?", "！", "!", "→")):
        return False
    return True


def honorific_stem(key: str) -> str | None:
    """``野崎さん`` → ``野崎`` when the honorific is attached to a name."""
    for suffix in _HONORIFICS:
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)]
    return None


def rest_after_speakers(text: str) -> str:
    """Cue body after stripping 《》/【】/leading （Name）."""
    out = _SPEAKER_GUILLEMET_RE.sub("", text)
    out = _SPEAKER_LENTICULAR_RE.sub("", out)
    out = _SPEAKER_START_PAREN_RE.sub("", out)
    return out.strip().strip("《》【】→ ").strip()


def is_whole_short_cue_key(key: str, text: str) -> bool:
    """Reject a key that is the entire leftover of a short cue (《撮影開始》)."""
    rest = rest_after_speakers(text)
    if rest:
        return key == rest and len(key) <= MAX_NAME_KEY_CHARS
    stripped = text.strip().strip("《》【】（）()→ ").strip()
    return key == stripped and len(key) <= MAX_NAME_KEY_CHARS


def _name_sized(span: str) -> bool:
    """Reject full-line 《dialogue》 wraps; keep 《野崎》 / 【佐野】."""
    key = clean_glossary_key(span)
    return is_acceptable_name_key(key)


def _speaker_candidates(span: str) -> list[str]:
    """Prefer inner （name）; otherwise a short 《name》 / 【name】 span."""
    match = _SPEAKER_START_PAREN_RE.match(span.strip())
    if match:
        return [match.group(1).strip()]
    if _name_sized(span):
        return [span.strip()]
    return []


def extract_speakers(
    text: str, confirmed: Iterable[str] | None = None
) -> list[str]:
    """（Name）/《Name》 only if Name is a Person/Place surface (佐野 yes)."""
    found: list[str] = []
    for span in _SPEAKER_GUILLEMET_RE.findall(text):
        found.extend(_speaker_candidates(span))
    for span in _SPEAKER_LENTICULAR_RE.findall(text):
        found.extend(_speaker_candidates(span))
    match = _SPEAKER_START_PAREN_RE.match(text)
    if match:
        found.append(match.group(1).strip())
    confirmed_set = set(confirmed) if confirmed is not None else None
    out: list[str] = []
    for span in found:
        key = clean_glossary_key(span)
        if is_generic_noun(key):
            continue
        if not is_person_place_surface(key, confirmed_set):
            continue
        if is_acceptable_name_key(key) or (confirmed_set is not None and key in confirmed_set):
            out.append(key)
    return out


def extract_latin(text: str) -> list[str]:
    return LATIN_NAME_RE.findall(text)


def extract_ginza_ents(text: str, nlp: object | None) -> list[str]:
    if nlp is None:
        return []
    doc = nlp(text)  # type: ignore[operator]
    ents = getattr(doc, "ents", ())
    out: list[str] = []
    for ent in ents:
        if getattr(ent, "label_", None) not in GINZA_LABELS:
            continue
        raw = clean_glossary_key(str(ent.text))
        stem = honorific_stem(raw)
        key = stem if stem and is_acceptable_name_key(stem) else raw
        if is_acceptable_name_key(key):
            out.append(key)
    return out


def _add_span(
    mapping: dict[str, str],
    span: str,
    *,
    confirmed: set[str] | None = None,
    source_text: str = "",
) -> None:
    key = clean_glossary_key(span)
    if is_generic_noun(key):
        return
    if not is_acceptable_name_key(key) and not (
        confirmed is not None and key in confirmed
    ):
        return
    if source_text and is_whole_short_cue_key(key, source_text) and (
        confirmed is None or key not in confirmed
    ):
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
    confirmed: set[str] = set()
    rows: list[tuple[str, str, bool, bool]] = []
    text_list = list(texts)
    for i, text in enumerate(text_list):
        lang = lang_list[i] if i < len(lang_list) else ""
        script = script_heuristic(text)
        japanese = script == "ja" or (
            bool(lang) and not is_latin_script(lang) and script != "ko"
        )
        non_latin_cue = (script in {"ja", "ko"}) or (
            bool(lang) and not is_latin_script(lang)
        )
        rows.append((text, lang, japanese, non_latin_cue))
        if japanese:
            ents = extract_ginza_ents(text, nlp)
            ginza_count += len(ents)
            confirmed.update(ents)
            for span in ents:
                _add_span(mapping, span, confirmed=confirmed, source_text=text)
    for text, _lang, _japanese, non_latin_cue in rows:
        if not non_latin_cue:
            continue
        for span in extract_speakers(text, confirmed=confirmed):
            _add_span(mapping, span, confirmed=confirmed, source_text=text)
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
