"""v0.6: completion max_tokens 256, in-source Person Hepburn. No 9B / xxNxx MT."""

from __future__ import annotations

from pathlib import Path

import pytest

from sublocal.backend import (
    DEFAULT_MAX_TOKENS,
    LENGTH_RETRY_MAX_TOKENS,
    EchoBackend,
    GemmaXBackend,
    completion_max_tokens,
    gemmax_prompt,
    gemmax_stop_sequences,
    maybe_name_instruction,
    strip_gemmax_completion,
)
from sublocal.extract import (
    GINZA_LABELS,
    GINZA_LOAD_CONFIG,
    GINZA_MODEL,
    GINZA_PERSON_LABELS,
    extract_file_glossary,
    extract_ginza_ents,
    extract_speakers,
    is_acceptable_name_key,
    merge_mappings,
    pykakasi_version,
    romanize_hepburn,
    spacy_load_ja_ginza,
)
from sublocal.formats import load
from sublocal.glossary import Glossary, GlossaryError
from sublocal.pipeline import translate_document, translate_file


def _srt(tmp: Path, name: str, cues: list[tuple[str, str]]) -> Path:
    lines: list[str] = []
    for i, (timing, text) in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(timing)
        lines.append(text)
        lines.append("")
    path = tmp / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class RecordingLlama:
    """Fake llama-cpp Llama: create_completion only, no chat, no GGUF."""

    def __init__(self, replies: list[dict] | None = None) -> None:
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []
        self.replies = list(replies or [])
        self.chat_calls = 0

    def tokenize(self, data, add_bos=True):
        raw = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
        # Rough stand-in: one token per character keeps tests deterministic.
        return list(range(max(1, len(raw))))

    def create_completion(self, prompt=None, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        if self.replies:
            return self.replies.pop(0)
        return {"choices": [{"text": "Hola", "finish_reason": "stop"}]}

    def create_chat_completion(self, *args, **kwargs):
        self.chat_calls += 1
        raise AssertionError("create_chat_completion must not be used")

    def __call__(self, *args, **kwargs):
        raise AssertionError("use create_completion, not a chat/__call__ wrapper")


def test_max_tokens_256_to_create_completion() -> None:
    llama = RecordingLlama()
    backend = GemmaXBackend(device="cpu")
    backend._llama = llama
    backend.translate(["こんにちは"], "jpn_Jpan", "spa_Latn")
    assert llama.kwargs
    assert llama.kwargs[0]["max_tokens"] == 256
    assert DEFAULT_MAX_TOKENS == 256
    assert llama.chat_calls == 0


def test_completion_api_not_chat() -> None:
    llama = RecordingLlama()
    backend = GemmaXBackend(device="cpu")
    backend._llama = llama
    backend.translate(["こんにちは"], "jpn_Jpan", "spa_Latn")
    assert llama.prompts
    assert "Translate this from Japanese to Spanish:" in llama.prompts[0]
    assert "<|im_start|>" not in llama.prompts[0]
    assert "messages" not in llama.prompts[0]


def test_stop_sequences_include_translate_and_blank() -> None:
    stops = gemmax_stop_sequences("Japanese")
    assert stops == ["\nJapanese:", "\nTranslate this", "\n\n"]
    assert "→" not in "".join(stops)
    llama = RecordingLlama()
    backend = GemmaXBackend(device="cpu")
    backend._llama = llama
    backend.translate(["hello"], "eng_Latn", "spa_Latn")
    stop = llama.kwargs[0]["stop"]
    assert stop == ["\nJapanese:", "\nTranslate this", "\n\n"]
    assert "\nEnglish:" not in stop
    assert "→" not in "".join(stop)


def test_arrow_post_strip() -> None:
    assert strip_gemmax_completion("Hola →", "Spanish") == "Hola"
    assert strip_gemmax_completion("→ Hola", "Spanish") == "Hola"
    assert strip_gemmax_completion("Spanish: Hola", "Spanish") == "Hola"
    assert strip_gemmax_completion("Japanese: leftover", "Spanish") == "leftover"


def test_finish_reason_length_bumps_once() -> None:
    llama = RecordingLlama(
        replies=[
            {"choices": [{"text": "Nozaki es", "finish_reason": "length"}]},
            {"choices": [{"text": "Nozaki es el nombre", "finish_reason": "stop"}]},
        ]
    )
    # Skip tokenize so first max_tokens is 256 and the bump can grow to 512.
    llama.tokenize = None  # type: ignore[method-assign]
    backend = GemmaXBackend(device="cpu")
    backend._llama = llama
    out = backend.translate(["野崎です"], "jpn_Jpan", "spa_Latn")
    assert out == ["Nozaki es el nombre"]
    assert len(llama.kwargs) == 2
    assert llama.kwargs[0]["max_tokens"] == 256
    assert llama.kwargs[1]["max_tokens"] == LENGTH_RETRY_MAX_TOKENS
    assert backend.finish_reason_counts["stop"] == 1


def test_max_tokens_fits_n_ctx() -> None:
    assert completion_max_tokens(2048, 2000) == 48
    assert completion_max_tokens(2048, 10) == 256
    llama = RecordingLlama()

    def many_tokens(data, add_bos=True):
        return list(range(1900))

    llama.tokenize = many_tokens  # type: ignore[method-assign]
    backend = GemmaXBackend(device="cpu")
    backend._llama = llama
    backend.n_ctx = 2048
    backend.translate(["こんにちは"], "jpn_Jpan", "spa_Latn")
    assert llama.kwargs[0]["max_tokens"] == min(256, 2048 - 1900)


def test_official_prompt_has_no_hepburn_or_name_hint() -> None:
    prompt = gemmax_prompt("Japanese", "Spanish", "野崎です")
    assert prompt == (
        "Translate this from Japanese to Spanish:\n"
        "Japanese: 野崎です\n"
        "Spanish:"
    )
    assert "Hepburn" not in prompt
    assert "Keep person names" not in prompt
    assert "Keep person and place names" not in prompt
    assert maybe_name_instruction("野崎です") is False
    llama = RecordingLlama()
    backend = GemmaXBackend(device="cpu", name_hint=True)
    backend._llama = llama
    backend.translate(["野崎です"], "jpn_Jpan", "spa_Latn")
    sent = llama.prompts[0]
    assert "Hepburn" not in sent
    assert "Keep person names" not in sent
    assert "Keep person and place names" not in sent
    assert sent.startswith("Translate this from Japanese to Spanish:")


def test_longest_first_protect_restore_nozaki_sano_liu() -> None:
    g = Glossary({"野崎": "Nozaki", "佐野": "Sano", "Liu": "Liu"})
    protected, pairs = g.protect("野崎と佐野とLiu")
    assert "野崎" not in protected
    assert "佐野" not in protected
    assert "Liu" not in protected
    assert pairs[0][0] in {"野崎", "佐野", "Liu"}
    assert g.restore(protected, pairs) == "NozakiとSanoとLiu"
    shorter = Glossary({"崎": "SAKI", "野崎": "Nozaki"})
    guarded, pairs2 = shorter.protect("野崎です")
    assert "崎" not in guarded
    assert shorter.restore(guarded, pairs2) == "Nozakiです"


def test_missing_sentinel_fails() -> None:
    g = Glossary({"野崎": "Nozaki"})
    _protected, pairs = g.protect("野崎です")
    with pytest.raises(GlossaryError, match="xx0xx"):
        g.restore("Nozaki es", pairs)


def test_speakers_and_latin_file_glossary_without_ginza() -> None:
    mapping, count, persons = extract_file_glossary(
        ["《野崎》Liuです", "【佐野】hello"],
        ["ja", "ja"],
        nlp=None,
        load=False,
    )
    assert count == 0
    assert "Liu" in mapping and mapping["Liu"] == "Liu"
    assert "野崎" in mapping
    assert "佐野" in mapping
    assert "野崎" in persons
    assert "佐野" in persons
    assert "Liu" not in persons
    assert extract_speakers("（東条）ううっ") == ["東条"]
    assert extract_speakers("《野崎さん》") == ["野崎"]


def test_ginza_ents_via_stub_nlp() -> None:
    class Ent:
        def __init__(self, text: str, label_: str) -> None:
            self.text = text
            self.label_ = label_

    class Nlp:
        def __call__(self, text: str):
            ents = []
            if "野崎" in text:
                ents.append(Ent("野崎", "Person"))
            if "バンコク" in text:
                ents.append(Ent("バンコク", "Place"))
            return type("Doc", (), {"ents": ents})()

    mapping, count, persons = extract_file_glossary(
        ["野崎がバンコクに飛んだ"],
        ["ja"],
        nlp=Nlp(),
        load=False,
    )
    assert count == 2
    assert mapping["野崎"]
    assert mapping["バンコク"]
    assert "野崎" in persons
    assert "バンコク" not in persons


def test_merge_yaml_wins_on_conflict() -> None:
    merged = merge_mappings({"野崎": "NOZAKI"}, {"野崎": "Nozaki", "ドラム": "Drum"})
    assert merged["野崎"] == "Nozaki"
    assert merged["ドラム"] == "Drum"


def test_ginza_labels_are_person_place_not_spacy_person() -> None:
    assert GINZA_LABELS == frozenset({"Person", "Place", "N_Person"})
    assert GINZA_PERSON_LABELS == frozenset({"Person", "N_Person"})
    assert "PERSON" not in GINZA_LABELS
    assert "GPE" not in GINZA_LABELS
    assert "Government" not in GINZA_LABELS
    assert GINZA_MODEL == "ja_ginza"
    assert GINZA_MODEL != "ja_ginza_electra"
    assert GINZA_LOAD_CONFIG == {
        "components": {"compound_splitter": {"split_mode": "A"}}
    }


def test_load_ginza_passes_split_mode_a(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    seen: dict = {}

    def fake_load(name, config=None, **kwargs):
        seen["name"] = name
        seen["config"] = config
        return "nlp"

    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=fake_load))
    nlp = spacy_load_ja_ginza()
    assert nlp == "nlp"
    assert seen["name"] == "ja_ginza"
    assert seen["name"] != "ja_ginza_electra"
    assert seen["config"]["components"]["compound_splitter"]["split_mode"] == "A"


def test_ginza_skips_government_keeps_person() -> None:
    class Ent:
        def __init__(self, text: str, label_: str) -> None:
            self.text = text
            self.label_ = label_

    class Nlp:
        def __call__(self, text: str):
            return type(
                "Doc",
                (),
                {
                    "ents": [
                        Ent("野崎", "Person"),
                        Ent("公安", "Government"),
                    ]
                },
            )()

    spans = extract_ginza_ents("野崎は公安だ", Nlp())
    assert "野崎" in spans
    assert "公安" not in spans
    mapping, count, _persons = extract_file_glossary(
        ["野崎は公安だ"],
        ["ja"],
        nlp=Nlp(),
        load=False,
    )
    assert count == 1
    assert "野崎" in mapping
    assert "公安" not in mapping


def test_pykakasi_version_from_metadata() -> None:
    """2.3.0 has no ``__version__``; we read importlib.metadata."""
    ver = pykakasi_version()
    assert ver != "unknown"
    assert ver[0].isdigit()
    import pykakasi

    assert getattr(pykakasi, "__version__", None) in {None, ver}


def test_file_glossary_keeps_latin_liu_in_jp_sentence(tmp_path: Path) -> None:
    src = _srt(
        tmp_path,
        "names.srt",
        [("00:00:00,000 --> 00:00:02,000", "LiuとZhangが飛んだ")],
    )
    llama = RecordingLlama()
    llama.tokenize = None  # type: ignore[method-assign]

    def echo_latin(prompt=None, **kwargs):
        llama.prompts.append(prompt)
        llama.kwargs.append(kwargs)
        return {"choices": [{"text": "Hola Liu y Zhang", "finish_reason": "stop"}]}

    llama.create_completion = echo_latin  # type: ignore[method-assign]
    backend = GemmaXBackend(device="cpu")
    backend._llama = llama
    doc = load(src)
    doc.cues[0].extra["lang"] = "ja"
    translate_document(doc, to_code="es", backend=backend)
    assert llama.prompts
    cue_line = llama.prompts[0].split("Japanese:", 1)[-1]
    assert "xx" not in cue_line
    assert "<g" not in cue_line
    assert "Liu" in cue_line
    assert "Zhang" in cue_line
    assert "Liu" in doc.cues[0].text
    assert "Zhang" in doc.cues[0].text


def test_translate_logs_ginza_and_finish(tmp_path: Path, capsys) -> None:
    src = _srt(
        tmp_path,
        "hi.srt",
        [("00:00:00,000 --> 00:00:01,000", "こんにちは")],
    )
    llama = RecordingLlama()
    llama.tokenize = None  # type: ignore[method-assign]
    backend = GemmaXBackend(device="cpu")
    backend._llama = llama
    translate_file(src, to_code="es", from_code="ja", backend=backend)
    err = capsys.readouterr().err
    assert "GiNZA entities" in err
    assert "in-source Person Hepburn" in err
    assert "finish_reason stop=" in err
    assert "leftover arrows" in err
    assert "leave JP" in err
    assert "no retry" in err
    assert "MT pass" in err


@pytest.mark.skipif(
    romanize_hepburn("野崎") == "野崎",
    reason="pykakasi not installed",
)
def test_hepburn_capitalize_nozaki() -> None:
    assert romanize_hepburn("野崎") == "Nozaki"
    assert romanize_hepburn("佐野") == "Sano"
    assert romanize_hepburn("Liu") == "Liu"


def test_junk_clause_keys_not_in_extracted_map() -> None:
    class Ent:
        def __init__(self, text: str, label_: str) -> None:
            self.text = text
            self.label_ = label_

    class Nlp:
        def __call__(self, text: str):
            return type(
                "Doc",
                (),
                {
                    "ents": [
                        Ent("野崎", "Person"),
                        Ent("佐野", "Person"),
                        Ent("別班", "Place"),
                        Ent("野崎が大きく息を吐く", "Person"),
                        Ent("野崎さん", "Person"),
                        Ent("野崎をマネて", "Person"),
                        Ent("佐野）", "Person"),
                        Ent("佐野公安部", "Person"),
                        Ent("撮影開始", "Person"),
                        Ent("諜報機関", "Person"),
                        Ent("公安", "Government"),
                    ]
                },
            )()

    texts = [
        "（佐野）5年前 野崎は北京で→",
        "当然 別班は裏であり→",
        "《撮影開始》",
        "野崎が大きく息を吐く",
        "野崎さん",
        "諜報機関だ",
    ]
    mapping, _count, persons = extract_file_glossary(
        texts, ["ja"] * len(texts), nlp=Nlp(), load=False
    )
    assert "野崎" in mapping
    assert "佐野" in mapping
    assert "別班" in mapping
    assert "野崎" in persons
    assert "佐野" in persons
    assert "別班" not in persons
    assert "撮影開始" not in mapping
    assert "諜報機関" not in mapping
    assert "野崎が大きく息を吐く" not in mapping
    assert "野崎さん" not in mapping
    assert "野崎をマネて" not in mapping
    assert "佐野公安部" not in mapping
    assert "佐野）" not in mapping
    assert "公安" not in mapping
    assert not is_acceptable_name_key("野崎が大きく息を吐く")
    assert is_acceptable_name_key("野崎")


def test_no_xxnxx_in_mt_path(tmp_path: Path) -> None:
    src = _srt(
        tmp_path,
        "mark.srt",
        [("00:00:00,000 --> 00:00:02,000", "野崎をマークしていた")],
    )
    calls: list[list[str]] = []

    class Rec(EchoBackend):
        def translate(self, texts, src_flores, tgt_flores):
            calls.append(list(texts))
            return ["Estaba marcando a Nozaki" for _ in texts]

    doc = load(src)
    translate_document(
        doc,
        to_code="es",
        from_code="ja",
        backend=Rec(),
        glossary=Glossary({"野崎": "Nozaki"}),
    )
    assert calls
    sent = calls[0][0]
    assert "xx" not in sent
    assert "<g" not in sent
    assert "Nozakiをマークしていた" == sent
    assert "野崎" not in sent


def test_hepburn_in_prompt_source_not_xxnxx() -> None:
    g = Glossary({"野崎": "Nozaki"})
    assert g.hepburn_in_source("野崎をマークしていた") == "Nozakiをマークしていた"
    protected, _pairs = g.protect("野崎をマークしていた")
    assert "xx0xx" in protected
    assert g.hepburn_in_source("野崎をマークしていた") != protected


def test_trailing_arrow_stripped_before_completion(tmp_path: Path) -> None:
    src = _srt(
        tmp_path,
        "cue7.srt",
        [("00:00:00,000 --> 00:00:02,000", "（佐野）5年前 野崎は北京で→")],
    )
    llama = RecordingLlama()
    llama.tokenize = None  # type: ignore[method-assign]

    def complete(prompt=None, **kwargs):
        llama.prompts.append(prompt)
        llama.kwargs.append(kwargs)
        return {
            "choices": [
                {"text": "Hace cinco años, Nozaki estaba en Beijing.", "finish_reason": "stop"}
            ]
        }

    llama.create_completion = complete  # type: ignore[method-assign]
    backend = GemmaXBackend(device="cpu")
    backend._llama = llama
    doc = load(src)
    translate_document(
        doc,
        to_code="es",
        from_code="ja",
        backend=backend,
        glossary=Glossary({"野崎": "Nozaki", "佐野": "Sano"}),
    )
    assert llama.prompts
    body = llama.prompts[0].split("Japanese:", 1)[-1]
    assert "→" not in body
    assert "5年前 Nozakiは北京で" in body
    assert "xx" not in body
    assert len(llama.prompts) == 1
    assert doc.cues[0].text.startswith("(Sano)")
    assert "Hace cinco años" in doc.cues[0].text


def test_latin_person_file_is_not_replaced() -> None:
    g = Glossary({"file": "File", "野崎": "Nozaki", "Liu": "Liu"})
    assert "file" not in g.person_keys
    assert "Liu" not in g.person_keys
    assert g.hepburn_in_source("file 1を") == "file 1を"
    assert g.hepburn_in_source("Liuと野崎") == "LiuとNozaki"


def test_honorific_speaker_nozaki_san_skips_mt(tmp_path: Path) -> None:
    src = _srt(
        tmp_path,
        "san.srt",
        [
            ("00:00:00,000 --> 00:00:01,000", "《野崎さん》"),
            ("00:00:01,000 --> 00:00:03,000", "野崎をマークしていた"),
        ],
    )
    calls: list[list[str]] = []

    class Rec(EchoBackend):
        def translate(self, texts, src_flores, tgt_flores):
            calls.append(list(texts))
            return ["Estaba marcando a Nozaki" for _ in texts]

    doc = load(src)
    translate_document(
        doc,
        to_code="es",
        from_code="ja",
        backend=Rec(),
        glossary=Glossary({"野崎": "Nozaki"}),
    )
    assert len(calls) == 1
    assert calls[0] == ["Nozakiをマークしていた"]
    assert "xx" not in calls[0][0]
    assert doc.cues[0].text == "(Nozaki)"
    assert "Nozakishu" not in doc.cues[0].text
    assert "Estaba marcando" in doc.cues[1].text


def test_speaker_tag_skips_gemmax(tmp_path: Path) -> None:
    src = _srt(
        tmp_path,
        "tags.srt",
        [
            ("00:00:00,000 --> 00:00:01,000", "（佐野）"),
            ("00:00:01,000 --> 00:00:04,000", "野崎をマークしていた"),
        ],
    )
    calls: list[list[str]] = []

    class Rec(EchoBackend):
        def translate(self, texts, src_flores, tgt_flores):
            calls.append(list(texts))
            return ["Estaba marcando a Nozaki" for _ in texts]

    doc = load(src)
    translate_document(
        doc,
        to_code="es",
        from_code="ja",
        backend=Rec(),
        glossary=Glossary({"野崎": "Nozaki", "佐野": "Sano"}),
    )
    assert len(calls) == 1
    assert calls[0] == ["Nozakiをマークしていた"]
    assert doc.cues[0].text == "(Sano)"
    assert "Estaba marcando" in doc.cues[1].text


def test_empty_mt_leaves_jp_without_retry(tmp_path: Path) -> None:
    src = _srt(
        tmp_path,
        "empty.srt",
        [("00:00:00,000 --> 00:00:02,000", "野崎は北京で飛んだ")],
    )
    calls: list[list[str]] = []

    class EmptyOnce(EchoBackend):
        def translate(self, texts, src_flores, tgt_flores):
            calls.append(list(texts))
            return ["" for _ in texts]

    doc = load(src)
    translate_document(
        doc,
        to_code="es",
        from_code="ja",
        backend=EmptyOnce(),
        glossary=Glossary({"野崎": "Nozaki"}),
    )
    assert len(calls) == 1
    assert "→" not in calls[0][0]
    assert "Nozakiは" in calls[0][0]
    assert doc.cues[0].text == "野崎は北京で飛んだ"


def test_nagasaki_overlay_iff_source_had_nozaki() -> None:
    g = Glossary({"野崎": "Nozaki"})
    out = g.overlay_names("野崎は北京で", "Hace cinco años, Nagasaki estaba en Beijing.")
    assert "Nozaki" in out
    assert "Nagasaki" not in out
    untouched = g.overlay_names("北京で", "Hace cinco años, Nagasaki estaba en Beijing.")
    assert "Nagasaki" in untouched
    assert not untouched.startswith("Nozaki")


def test_sentence_with_nozaki_does_not_retry(tmp_path: Path) -> None:
    src = _srt(
        tmp_path,
        "sent.srt",
        [("00:00:00,000 --> 00:00:02,000", "野崎は北京で飛んだ")],
    )
    calls: list[list[str]] = []

    class Sentence(EchoBackend):
        def translate(self, texts, src_flores, tgt_flores):
            calls.append(list(texts))
            return ["Hace cinco años, Nozaki estaba en Beijing." for _ in texts]

    doc = load(src)
    translate_document(
        doc,
        to_code="es",
        from_code="ja",
        backend=Sentence(),
        glossary=Glossary({"野崎": "Nozaki"}),
    )
    assert len(calls) == 1
    assert "xx" not in calls[0][0]
    assert "Hace cinco años, Nozaki estaba en Beijing." == doc.cues[0].text


def test_beppan_stays_japanese_in_prompt(tmp_path: Path) -> None:
    src = _srt(
        tmp_path,
        "beppan.srt",
        [("00:00:00,000 --> 00:00:02,000", "当然 別班は裏であり")],
    )
    llama = RecordingLlama()
    llama.tokenize = None  # type: ignore[method-assign]

    def complete(prompt=None, **kwargs):
        llama.prompts.append(prompt)
        llama.kwargs.append(kwargs)
        return {
            "choices": [
                {"text": "La seccion secreta opera en la sombra", "finish_reason": "stop"}
            ]
        }

    llama.create_completion = complete  # type: ignore[method-assign]
    backend = GemmaXBackend(device="cpu")
    backend._llama = llama
    doc = load(src)
    translate_document(
        doc,
        to_code="es",
        from_code="ja",
        backend=backend,
        glossary=Glossary({"別班": "Beppan", "野崎": "Nozaki"}),
    )
    sent = llama.prompts[0]
    assert "別班" in sent
    assert "Beppan" not in sent.split("Japanese:", 1)[-1].split("Spanish:", 1)[0]
    assert "xx0xx" not in sent
    assert "Hepburn" not in sent
    assert "Keep person names" not in sent
    assert doc.cues[0].text != "Beppan"
    assert "seccion" in doc.cues[0].text


def test_create_completion_max_tokens_still_256() -> None:
    llama = RecordingLlama()
    llama.tokenize = None  # type: ignore[method-assign]
    backend = GemmaXBackend(device="cpu")
    backend._llama = llama
    backend.translate(["こんにちは"], "jpn_Jpan", "spa_Latn")
    assert llama.kwargs[0]["max_tokens"] == 256
    assert llama.kwargs[0].get("add_special_tokens") is False
    assert "create_chat_completion" not in llama.kwargs[0]
