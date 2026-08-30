import pytest

from sublocal.languages import UnknownLanguageError, display_code, to_flores


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
