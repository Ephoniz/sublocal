import pytest

from sublocal.languages import (
    GEMMAX_28_NAMES,
    UnknownLanguageError,
    display_code,
    to_english_name,
    to_flores,
)


def test_short_codes() -> None:
    assert to_flores("en") == "eng_Latn"
    assert to_flores("es") == "spa_Latn"
    assert to_flores("ja") == "jpn_Jpan"
    assert to_flores("ko") == "kor_Hang"
    assert to_flores("zh-tw") == "zho_Hant"
    assert to_flores("pt-BR") == "por_Latn"


def test_flores_passthrough() -> None:
    assert to_flores("eng_Latn") == "eng_Latn"
    assert to_flores("jpn_Jpan") == "jpn_Jpan"


def test_unknown() -> None:
    with pytest.raises(UnknownLanguageError):
        to_flores("xx")


def test_display_code() -> None:
    assert display_code("en") == "en"
    assert display_code("eng_Latn") == "eng"


def test_english_names_from_iso_and_flores() -> None:
    assert to_english_name("es") == "Spanish"
    assert to_english_name("ja") == "Japanese"
    assert to_english_name("en") == "English"
    assert to_english_name("ko") == "Korean"
    assert to_english_name("zh") == "Chinese"
    assert to_english_name("fr") == "French"
    assert to_english_name("jpn_Jpan") == "Japanese"
    assert to_english_name("spa_Latn") == "Spanish"
    assert to_english_name("Japanese") == "Japanese"
    assert "Japanese" in GEMMAX_28_NAMES
    assert len(GEMMAX_28_NAMES) == 28
