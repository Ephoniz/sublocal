"""Map CLI language codes to NLLB FLORES-200 codes."""

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
