from __future__ import annotations

from pathlib import Path

import pytest

from sublocal.backend import EchoBackend
from sublocal.formats import load
from sublocal.glossary import Glossary, GlossaryError, needs_nllb
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


def test_needs_nllb_false_for_speaker_tag_groan() -> None:
    g = _drama()
    protected, pairs = g.protect("（東条）ううっ あっ あっ…")
    assert pairs
    assert "xx0xx" in protected
    assert needs_nllb(protected) is False


def test_needs_nllb_true_for_real_dialogue() -> None:
    g = _drama()
    protected, _pairs = g.protect("おい ドラム 東条 まだ起きねえのか？")
    assert needs_nllb(protected) is True
    particles, _ = g.protect("野崎のドラムとバンコク")
    assert needs_nllb(particles) is True


def test_restore_accepts_spaced_and_cased_sentinels() -> None:
    g = _drama()
    _protected, pairs = g.protect("ドラム")
    assert g.restore("xx 0 xx", pairs) == "Drum"
    assert g.restore("XX0XX", pairs) == "Drum"


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


def test_groan_cue_skips_nllb_dialogue_still_sent(tmp_path: Path, capsys) -> None:
    src = tmp_path / "vivant.srt"
    src.write_text(
        "5\n"
        "00:00:07,000 --> 00:00:08,500\n"
        "（東条）ううっ あっ あっ…\n"
        "\n"
        "6\n"
        "00:00:10,000 --> 00:00:13,000\n"
        "《おい ドラム 東条 まだ起きねえのか？》\n",
        encoding="utf-8",
    )
    out = tmp_path / "vivant.es.srt"
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
    err = capsys.readouterr().err
    assert "Translating 1 cues" in err
    assert seen
    assert all("ううっ" not in t for t in seen)
    assert all("（xx0xx）" not in t and "（xx0xx)" not in t for t in seen)
    assert any("xx" in t for t in seen)
    assert any("起き" in t or "おい" in t for t in seen)
    doc = load(out)
    assert "Tojo" in doc.cues[0].text
    assert "ううっ" in doc.cues[0].text
    assert "Drum" in doc.cues[1].text
    assert "Tojo" in doc.cues[1].text
    assert "起き" in doc.cues[1].text
