from __future__ import annotations

from pathlib import Path

import pytest

from sublocal.backend import EchoBackend
from sublocal.formats import load
from sublocal.glossary import Glossary, GlossaryError
from sublocal.pipeline import translate_file

DRAMA = Path(__file__).resolve().parents[1] / "examples" / "drama.yml"


def _drama() -> Glossary:
    return Glossary.load(DRAMA)


def test_examples_drama_yml_utf8() -> None:
    text = DRAMA.read_text(encoding="utf-8")
    assert "野崎: Nozaki" in text
    assert "バンコク: Bangkok" in text
    g = _drama()
    assert g.mapping["ドラム"] == "Drum"
    assert g.mapping["バンコク"] == "Bangkok"


def test_longest_first_vladivostok() -> None:
    g = Glossary({"ストク": "STOK", "ウラジオストク": "Vladivostok"})
    protected, pairs = g.protect("ウラジオストクへ行く")
    assert "ストク" not in protected
    assert pairs[0][0] == "ウラジオストク"
    assert g.restore(protected, pairs) == "Vladivostokへ行く"


def test_drum_becomes_sentinel_never_latin() -> None:
    g = _drama()
    protected, pairs = g.protect("野崎のドラム")
    assert "Drum" not in protected
    assert "Nozaki" not in protected
    assert "D" not in protected or "Drum" not in protected
    assert "ドラム" not in protected
    assert "野崎" not in protected
    assert "xx0xx" in protected
    restored = g.restore(protected, pairs)
    assert restored == "NozakiのDrum"


def test_restore_yields_latin_names() -> None:
    g = _drama()
    src = "野崎と新庄と東条とドラムとウラジオストクとバンコク"
    protected, pairs = g.protect(src)
    out = g.restore(protected, pairs, target="value")
    assert out == "NozakiとShinjoとTojoとDrumとVladivostokとBangkok"


def test_missing_sentinel_fails() -> None:
    g = _drama()
    _protected, pairs = g.protect("ドラム")
    with pytest.raises(GlossaryError, match="xx0xx"):
        g.restore("tambor", pairs)


def test_echo_backend_glossary_restores_latin(tmp_path: Path) -> None:
    src = tmp_path / "ja.srt"
    src.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n野崎のドラムとバンコク\n",
        encoding="utf-8",
    )
    out = tmp_path / "ja.es.srt"
    seen: list[str] = []

    class Recorder(EchoBackend):
        def translate(self, texts, src_flores, tgt_flores):
            seen.extend(texts)
            return super().translate(texts, src_flores, tgt_flores)

    translate_file(
        src,
        to_code="es",
        from_code="ja",
        output_path=out,
        backend=Recorder(),
        glossary=DRAMA,
    )
    assert seen
    assert all("Drum" not in t for t in seen)
    assert all("ドラム" not in t for t in seen)
    assert any("xx" in t for t in seen)
    doc = load(out)
    assert "Drum" in doc.cues[0].text
    assert "Nozaki" in doc.cues[0].text
    assert "Bangkok" in doc.cues[0].text
    assert "ドラム" not in doc.cues[0].text


def test_restore_to_japanese_keys() -> None:
    g = _drama()
    protected, pairs = g.protect("野崎のドラム")
    assert g.restore(protected, pairs, target="key") == "野崎のドラム"
