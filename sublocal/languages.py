"""Map CLI language codes to NLLB FLORES-200 codes and GemmaX English names."""

from __future__ import annotations

import re

# ISO 639-1 (and a few aliases) -> NLLB FLORES-200. Not every NLLB language
# has a two-letter code; pass the FLORES code directly when needed.
ISO639_TO_FLORES: dict[str, str] = {
    "af": "afr_Latn",
    "am": "amh_Ethi",
    "ar": "arb_Arab",
    "az": "azj_Latn",
    "be": "bel_Cyrl",
    "bg": "bul_Cyrl",
    "bn": "ben_Beng",
    "bs": "bos_Latn",
    "ca": "cat_Latn",
    "cs": "ces_Latn",
    "cy": "cym_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "eo": "epo_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "eu": "eus_Latn",
    "fa": "pes_Arab",
    "fi": "fin_Latn",
    "fil": "tgl_Latn",
    "fr": "fra_Latn",
    "ga": "gle_Latn",
    "gl": "glg_Latn",
    "gu": "guj_Gujr",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",
    "id": "ind_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "iw": "heb_Hebr",
    "ja": "jpn_Jpan",
    "jv": "jav_Latn",
    "ka": "kat_Geor",
    "kk": "kaz_Cyrl",
    "km": "khm_Khmr",
    "kn": "kan_Knda",
    "ko": "kor_Hang",
    "lo": "lao_Laoo",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "mk": "mkd_Cyrl",
    "ml": "mal_Mlym",
    "mn": "khk_Cyrl",
    "mr": "mar_Deva",
    "ms": "zsm_Latn",
    "mt": "mlt_Latn",
    "my": "mya_Mymr",
    "nb": "nob_Latn",
    "ne": "npi_Deva",
    "nl": "nld_Latn",
    "nn": "nno_Latn",
    "no": "nob_Latn",
    "pa": "pan_Guru",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "pt-br": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "si": "sin_Sinh",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sq": "als_Latn",
    "sr": "srp_Cyrl",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "th": "tha_Thai",
    "tl": "tgl_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "uz": "uzn_Latn",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
    "zh-cn": "zho_Hans",
    "zh-hans": "zho_Hans",
    "zh-hant": "zho_Hant",
    "zh-tw": "zho_Hant",
    "yue": "yue_Hant",
}

FLORES_RE = re.compile(r"^[a-z]{3}_[A-Za-z]{4}$")

# GemmaX2-28 English names (arxiv 2502.02481 / ModelSpace card). Prompt uses
# these strings, not FLORES codes.
GEMMAX_28_NAMES: tuple[str, ...] = (
    "Arabic",
    "Bengali",
    "Czech",
    "German",
    "English",
    "Spanish",
    "Persian",
    "French",
    "Hebrew",
    "Hindi",
    "Indonesian",
    "Italian",
    "Japanese",
    "Khmer",
    "Korean",
    "Lao",
    "Malay",
    "Burmese",
    "Dutch",
    "Polish",
    "Portuguese",
    "Russian",
    "Thai",
    "Tagalog",
    "Turkish",
    "Urdu",
    "Vietnamese",
    "Chinese",
)

ISO639_TO_ENGLISH: dict[str, str] = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Tagalog",
    "fr": "French",
    "ga": "Irish",
    "gl": "Galician",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "iw": "Hebrew",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "nb": "Norwegian",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Norwegian",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "zh-hant": "Chinese",
    "zh-tw": "Chinese",
    "yue": "Chinese",
}

FLORES_TO_ENGLISH: dict[str, str] = {}
for _iso, _flores in ISO639_TO_FLORES.items():
    _name = ISO639_TO_ENGLISH.get(_iso)
    if _name and _flores not in FLORES_TO_ENGLISH:
        FLORES_TO_ENGLISH[_flores] = _name


class UnknownLanguageError(ValueError):
    pass


def to_flores(code: str) -> str:
    """Accept ``en``, ``es``, ``zh-tw``, or a FLORES code like ``eng_Latn``."""
    raw = code.strip()
    if not raw:
        raise UnknownLanguageError("Empty language code.")
    if FLORES_RE.fullmatch(raw):
        return raw
    key = raw.lower().replace("_", "-")
    if key in ISO639_TO_FLORES:
        return ISO639_TO_FLORES[key]
    primary = key.split("-", 1)[0]
    if primary in ISO639_TO_FLORES:
        return ISO639_TO_FLORES[primary]
    raise UnknownLanguageError(
        f"Unknown language {code!r}. Use a short code like en / es / ja, "
        "or an NLLB FLORES code like eng_Latn."
    )


def display_code(code: str) -> str:
    """Filename-friendly code: keep short codes, strip script from FLORES."""
    raw = code.strip()
    if FLORES_RE.fullmatch(raw):
        return raw.split("_", 1)[0]
    return raw.lower().split("-", 1)[0]


def to_english_name(code: str) -> str:
    """ISO (``es``) or FLORES (``spa_Latn``) → GemmaX English name (Spanish)."""
    raw = code.strip()
    if not raw:
        raise UnknownLanguageError("Empty language code.")
    if raw in GEMMAX_28_NAMES:
        return raw
    if FLORES_RE.fullmatch(raw):
        if raw in FLORES_TO_ENGLISH:
            return FLORES_TO_ENGLISH[raw]
        raise UnknownLanguageError(
            f"Unknown language {code!r}. Use a short code like en / es / ja."
        )
    key = raw.lower().replace("_", "-")
    if key in ISO639_TO_ENGLISH:
        return ISO639_TO_ENGLISH[key]
    primary = key.split("-", 1)[0]
    if primary in ISO639_TO_ENGLISH:
        return ISO639_TO_ENGLISH[primary]
    titled = raw[:1].upper() + raw[1:].lower()
    if titled in GEMMAX_28_NAMES:
        return titled
    raise UnknownLanguageError(
        f"Unknown language {code!r}. Use a short code like en / es / ja."
    )
