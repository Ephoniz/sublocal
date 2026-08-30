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
    assert "GLS0" in protected
    assert "行く" in g.restore(protected, pairs)
    assert "Vladivostok" in g.restore(protected, pairs)


def test_protect_bangkok_keeps_flew_in_same_string() -> None:
    g = _drama()
    protected, pairs = g.protect("バンコクに飛んだ")
    assert "に飛んだ" in protected
    assert "GLS" in protected
    assert "Bangkok" not in protected
    assert "Drum" not in protected
    assert "バンコク" not in protected
    assert pairs


def test_protect_never_contains_latin() -> None:
    g = _drama()
    protected, _pairs = g.protect("おい ドラム 東条 まだ起きねえのか？")
    assert "Drum" not in protected
    assert "Tojo" not in protected
    assert "ドラム" not in protected
    assert "東条" not in protected
    assert "GLS0" in protected
    assert "GLS1" in protected
    assert "まだ起きねえのか" in protected


def test_restore_gls_full_sentence() -> None:
    g = _drama()
    _protected, pairs = g.protect("ドラムとバンコク")
    # Replacement order is longest-first: バンコク then ドラム.
    by_key = {k: i for i, (k, _) in enumerate(pairs)}
    drum = f"GLS{by_key['ドラム']}"
    bkk = f"GLS{by_key['バンコク']}"
    assert g.restore(f"{drum} flew to {bkk}", pairs) == "Drum flew to Bangkok"


def test_restore_accepts_spaced_and_cased_gls() -> None:
    g = _drama()
    _protected, pairs = g.protect("ドラム")
    assert g.restore("GLS 0", pairs) == "Drum"
    assert g.restore("gls0", pairs) == "Drum"
    assert g.restore("<g0>", pairs) == "Drum"


def test_missing_sentinel_fails() -> None:
    g = _drama()
    _protected, pairs = g.protect("ドラム")
    with pytest.raises(GlossaryError, match="GLS0"):
        g.restore("tambor", pairs)


def test_echo_full_cue_strips_particles(tmp_path: Path) -> None:
    src = tmp_path / "ja.srt"
    src.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n野崎に 新庄がバンコクに飛んだ\n",
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
    assert all("Bangkok" not in t for t in seen)
    assert all("バンコク" not in t for t in seen)
    assert any("に飛んだ" in t for t in seen)
    assert any("GLS" in t for t in seen)
    doc = load(out)
    text = doc.cues[0].text
    assert "Nozaki" in text
    assert "Shinjo" in text
    assert "Bangkok" in text
    assert "に" not in text
    assert "が" not in text
    assert "バンコク" not in text


def test_peel_speaker_allows_leading_bracket() -> None:
    g = _drama()
    name, rest = g.peel_speaker(
        "《（野崎）この中に直前でフライトの時間が変更した便はあるか？》"
    )
    assert name == "野崎"
    assert "野崎" not in rest
    assert "フライト" in rest
    assert rest.startswith("この中に")


def test_bracketed_speaker_plus_kanji_body(tmp_path: Path) -> None:
    src = tmp_path / "c28.srt"
    src.write_text(
        "28\n"
        "00:00:40,000 --> 00:00:44,000\n"
        "《（野崎）この中に直前でフライトの時間が変更した便はあるか？》\n"
        "\n"
        "38\n"
        "00:00:50,000 --> 00:00:54,000\n"
        "（東条）野崎に / 新庄がバンコクに飛んだ\n",
        encoding="utf-8",
    )
    out = tmp_path / "c28.es.srt"
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
    assert all("野崎" not in t for t in seen)
    assert all("東条" not in t for t in seen)
    assert any("フライト" in t or "変更" in t for t in seen)
    assert any("に飛んだ" in t for t in seen)
    doc = load(out)
    assert doc.cues[0].text.startswith("(Nozaki)")
    assert "野崎" not in doc.cues[0].text
    assert "フライト" in doc.cues[0].text or "変更" in doc.cues[0].text
    assert doc.cues[1].text.startswith("(Tojo)")
    assert "Tojo" in doc.cues[1].text
    assert "Nozaki" in doc.cues[1].text
    assert "Shinjo" in doc.cues[1].text
    assert "Bangkok" in doc.cues[1].text
    assert "東条" not in doc.cues[1].text
    assert "バンコク" not in doc.cues[1].text


def test_yoh_vocative_restores_locally(tmp_path: Path) -> None:
    src = tmp_path / "yoh.srt"
    src.write_text(
        "10\n"
        "00:00:15,000 --> 00:00:16,000\n"
        "＜よう 東条＞\n"
        "\n"
        "11\n"
        "00:00:16,000 --> 00:00:18,000\n"
        "バンコクに飛んだ\n",
        encoding="utf-8",
    )
    out = tmp_path / "yoh.es.srt"
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
    assert all("よう" not in t for t in seen)
    assert all("東条" not in t for t in seen)
    assert any("に飛んだ" in t for t in seen)
    doc = load(out)
    assert "Tojo" in doc.cues[0].text
    assert "東条" not in doc.cues[0].text
    assert "よう" not in doc.cues[0].text
    assert "Bangkok" in doc.cues[1].text


def test_groan_not_sent_output_has_tojo(tmp_path: Path) -> None:
    src = tmp_path / "groan.srt"
    src.write_text(
        "5\n"
        "00:00:07,000 --> 00:00:08,500\n"
        "（東条）ううっ あっ あっ…\n"
        "\n"
        "6\n"
        "00:00:10,000 --> 00:00:13,000\n"
        "おい ドラム 東条 まだ起きねえのか？\n",
        encoding="utf-8",
    )
    out = tmp_path / "groan.es.srt"
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
    assert all("ううっ" not in t for t in seen)
    assert all("東条" not in t for t in seen)
    assert any("起き" in t for t in seen)
    doc = load(out)
    assert "Tojo" in doc.cues[0].text
    assert "東条" not in doc.cues[0].text
    assert "Drum" in doc.cues[1].text
    assert "Tojo" in doc.cues[1].text


def test_xml_retry_when_gls_dropped(tmp_path: Path) -> None:
    src = tmp_path / "retry.srt"
    src.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nバンコクに飛んだ\n",
        encoding="utf-8",
    )
    out = tmp_path / "retry.es.srt"
    calls: list[list[str]] = []

    class DropThenXml(EchoBackend):
        def translate(self, texts, src_flores, tgt_flores):
            calls.append(list(texts))
            if any("<g" in t for t in texts):
                return list(texts)
            return ["he flew to the moon" for _ in texts]

    translate_file(
        src,
        to_code="es",
        from_code="ja",
        output_path=out,
        backend=DropThenXml(),
        glossary=DRAMA,
    )
    assert len(calls) == 2
    assert any("GLS" in t for t in calls[0])
    assert any("<g0>" in t or "<g" in t for t in calls[1])
    doc = load(out)
    assert "Bangkok" in doc.cues[0].text
    assert "バンコク" not in doc.cues[0].text


def test_missing_after_xml_retry_fails(tmp_path: Path) -> None:
    src = tmp_path / "drop.srt"
    src.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nバンコクに飛んだ\n",
        encoding="utf-8",
    )

    class AlwaysDrop(EchoBackend):
        def translate(self, texts, src_flores, tgt_flores):
            return ["the moon" for _ in texts]

    with pytest.raises(GlossaryError, match="GLS0"):
        translate_file(
            src,
            to_code="es",
            from_code="ja",
            output_path=tmp_path / "drop.es.srt",
            backend=AlwaysDrop(),
            glossary=DRAMA,
        )


def test_restore_to_japanese_keys() -> None:
    g = _drama()
    protected, pairs = g.protect("野崎のドラム")
    restored = g.restore(protected, pairs, target="key")
    assert "野崎" in restored
    assert "ドラム" in restored
