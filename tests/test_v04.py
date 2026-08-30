"""v0.4 product: mixed-language ASR → per-cue NLLB. CPU, no 3.3B download."""

from __future__ import annotations

from pathlib import Path

import pytest

from sublocal.backend import EchoBackend
from sublocal.cli import is_product_argv, main, parse_product_args
from sublocal.cues_jsonl import cues_jsonl_path, read_cues_jsonl, write_cues_jsonl
from sublocal.formats import load
from sublocal.formats.base import Block, Cue, Document
from sublocal.languages import to_flores
from sublocal.lid import (
    detect_cue_lang,
    protect_latin_names,
    restore_latin_names,
    script_heuristic,
)
from sublocal.pipeline import run_product, translate_document, translate_file
from sublocal.transcribe import (
    Segment,
    Transcript,
    Word,
    _whisper_infer,
    apply_regroup,
    stamp_langs_from_vad_chunks,
    stamp_langs_sequential,
    transcript_to_document,
    transcribe_file,
    whisper_transcribe_kwargs,
)


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


def test_per_chunk_kwargs_are_not_multilingual_true() -> None:
    kw = whisper_transcribe_kwargs(None, per_chunk=True)
    assert kw["language"] is None
    assert kw["multilingual"] is False
    assert kw["task"] == "transcribe"
    assert kw["task"] != "translate"
    assert kw["condition_on_previous_text"] is False
    assert kw["vad_filter"] is False
    assert kw["word_timestamps"] is True
    assert kw["without_timestamps"] is False


def test_full_file_language_none_is_rejected() -> None:
    with pytest.raises(ValueError, match="VAD slice"):
        whisper_transcribe_kwargs(None)


def test_multilingual_false_when_language_ja() -> None:
    kw = whisper_transcribe_kwargs("ja")
    assert kw["language"] == "ja"
    assert kw["multilingual"] is False
    assert kw["task"] == "transcribe"
    assert kw["without_timestamps"] is False


def test_whisper_infer_mono_ja_one_call() -> None:
    seen: list[dict] = []

    class Model:
        def transcribe(self, audio, **kwargs):
            seen.append(kwargs)
            return Transcript([])

    _whisper_infer(Model(), object(), "ja", lambda *_a: None)
    assert len(seen) == 1
    assert seen[0]["language"] == "ja"
    assert seen[0].get("multilingual") is False
    assert seen[0]["task"] == "transcribe"


def test_batch_still_sets_without_timestamps_false() -> None:
    kw = whisper_transcribe_kwargs(None, per_chunk=True, batched=True)
    assert kw["without_timestamps"] is False
    assert kw["multilingual"] is False


def test_mixed_path_transcribes_each_vad_slice_not_concatenated(monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    sr = 16000
    wave = np.zeros(sr * 20, dtype=np.float32)
    monkeypatch.setattr(
        "sublocal.transcribe.speech_timestamps",
        lambda audio, sampling_rate=sr: [
            {"start": 0, "end": 12 * sr},
            {"start": 13 * sr, "end": int(19.5 * sr)},
        ],
    )
    calls: list[tuple[int, dict]] = []

    class WhisperModel:
        def transcribe(self, audio, **kwargs):
            calls.append((len(audio), dict(kwargs)))
            lang = "ja" if len(calls) == 1 else "en"
            text = "こんにちは" if lang == "ja" else "hello"
            seg = Segment(words=[Word(text, 0.1, 1.0)], start=0.1, end=1.0)
            info = type("Info", (), {"language": lang})()
            return [seg], info

    _whisper_infer(WhisperModel(), wave, None, lambda *_a: None)
    assert len(calls) == 2
    assert calls[0][0] == 12 * sr
    assert calls[1][0] == int(19.5 * sr) - 13 * sr
    assert calls[0][0] + calls[1][0] < len(wave)
    for _n, kw in calls:
        assert kw["language"] is None
        assert kw["multilingual"] is False
        assert kw["task"] == "transcribe"
        assert kw["without_timestamps"] is False
        assert kw.get("vad_filter") is False


def test_chunk_info_language_and_time_offset(monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    sr = 16000
    wave = np.zeros(sr * 20, dtype=np.float32)
    monkeypatch.setattr(
        "sublocal.transcribe.speech_timestamps",
        lambda audio, sampling_rate=sr: [
            {"start": 0, "end": 12 * sr},
            {"start": 13 * sr, "end": int(19.5 * sr)},
        ],
    )

    class WhisperModel:
        n = 0

        def transcribe(self, audio, **kwargs):
            WhisperModel.n += 1
            lang = "ja" if WhisperModel.n == 1 else "en"
            text = "坂の中" if lang == "ja" else "hello there"
            seg = Segment(words=[Word(text, 0.50, 1.40)], start=0.50, end=1.40)
            return [seg], type("Info", (), {"language": lang})()

    result = _whisper_infer(WhisperModel(), wave, None, lambda *_a: None)
    assert isinstance(result, Transcript)
    langs = [s.lang for s in result.segments]
    assert langs == ["ja", "en"]
    assert result.segments[0].words[0].start == pytest.approx(0.50)
    assert result.segments[1].words[0].start == pytest.approx(13.50)
    assert result.segments[1].words[0].end == pytest.approx(14.40)


def test_jsonl_fills_null_lang_from_script_heuristic(
    tmp_path: Path, monkeypatch
) -> None:
    dummy = tmp_path / "clip.mp4"
    dummy.write_bytes(b"fake-media")
    monkeypatch.setattr("sublocal.device.cuda_device_count", lambda: 1)
    monkeypatch.setattr("sublocal.device.cuda_device_name", lambda index=0: "Fake GPU")
    monkeypatch.setattr("sublocal.transcribe.load_whisper", lambda *a, **k: object())
    monkeypatch.setattr("sublocal.transcribe.decode_audio", lambda path: object())

    def fake_infer(model, audio, language, progress_cb, **kwargs):
        progress_cb(1.0, 1.0)
        return Transcript(
            [
                Segment(words=[Word("こんにちは。", 0.5, 1.4)], start=0.5, end=1.4),
                Segment(
                    words=[Word("Good morning to all of my friends.", 13.0, 14.0)],
                    start=13.0,
                    end=14.0,
                ),
            ]
        )

    monkeypatch.setattr("sublocal.transcribe._whisper_infer", fake_infer)
    transcribe_file(dummy, output_path=tmp_path / "clip.srt")
    rows = read_cues_jsonl(cues_jsonl_path(tmp_path / "clip.srt"))
    assert [r["lang"] for r in rows] == ["ja", "en"]
    assert all(r["lang"] is not None for r in rows)


def test_from_any_copies_lang_from_fw_like_object() -> None:
    class FWSeg:
        def __init__(self) -> None:
            self.words = [Word("hello", 0.0, 1.0)]
            self.start = 0.0
            self.end = 1.0
            self.lang = "en"

    tr = Transcript.from_any(type("R", (), {"segments": [FWSeg()]})())
    assert tr.segments[0].lang == "en"
    tr2 = Transcript.from_any(
        type(
            "R",
            (),
            {"segments": [{"text": "こんにちは", "start": 0.0, "end": 1.0, "lang": "ja"}]},
        )()
    )
    assert tr2.segments[0].lang == "ja"


def test_stamp_lang_when_tokenizer_language_code_changes() -> None:
    class Tok:
        language_code = "ja"

    class Seg:
        def __init__(self, text: str, start: float) -> None:
            self.text = text
            self.start = start
            self.end = start + 1
            self.words = [Word(word=text, start=start, end=start + 1)]

    tok = Tok()

    def gen():
        tok.language_code = "ja"
        yield Seg("こんにちは", 0.0)
        yield Seg("元気ですか", 1.5)
        tok.language_code = "en"
        yield Seg("hello", 31.0)
        tok.language_code = "es"
        yield Seg("hola", 62.0)

    stamped = stamp_langs_sequential(gen(), tok)
    assert [s.lang for s in stamped] == ["ja", "ja", "en", "es"]

    tr = Transcript(
        [
            Segment(words=s.words, start=s.start, end=s.end, lang=s.lang)
            for s in stamped
        ]
    )
    doc = transcript_to_document(tr)
    assert [c.extra.get("lang") for c in doc.cues] == ["ja", "ja", "en", "es"]

    # Window lang is copied onto every cue regrouped from that window.
    window = Transcript(
        [Segment(words=stamped[0].words + stamped[1].words, lang="ja")]
    )
    regrouped = apply_regroup(window)
    assert all(s.lang == "ja" for s in regrouped.segments if s.words)


def test_batched_lang_from_vad_chunk_tokens() -> None:
    segs = [
        Segment(words=[Word("こんにちは", 0.2, 1.0)], start=0.2, end=1.0),
        Segment(words=[Word("hello", 10.1, 11.0)], start=10.1, end=11.0),
    ]
    chunks = [
        {"offset": 0.0, "duration": 5.0},
        {"offset": 10.0, "duration": 4.0},
    ]
    stamp_langs_from_vad_chunks(segs, chunks, ["<|ja|>", "<|en|>"])
    assert segs[0].lang == "ja"
    assert segs[1].lang == "en"


def test_copy_through_if_cue_lang_equals_to(tmp_path: Path) -> None:
    src = _srt(
        tmp_path,
        "mixed.srt",
        [
            ("00:00:00,000 --> 00:00:02,000", "Already Spanish"),
            ("00:00:02,000 --> 00:00:04,000", "こんにちは"),
        ],
    )
    doc = load(src)
    doc.cues[0].extra["lang"] = "es"
    doc.cues[1].extra["lang"] = "ja"

    class Mangle(EchoBackend):
        def translate(self, texts, src_flores, tgt_flores):
            return [f"MT({t})" for t in texts]

    translate_document(doc, to_code="es", backend=Mangle())
    assert doc.cues[0].text == "Already Spanish"
    assert doc.cues[1].text.startswith("MT(")


def test_script_heuristic_hiragana_hangul_latin() -> None:
    assert script_heuristic("こんにちは") == "ja"
    assert script_heuristic("カタカナ") == "ja"
    assert script_heuristic("漢字だけ") == "ja"
    assert script_heuristic("안녕하세요") == "ko"
    assert script_heuristic("hello there") == "en"
    assert detect_cue_lang("こんにちは") == "ja"
    assert detect_cue_lang("안녕하세요") == "ko"


def test_flores_whisper_iso() -> None:
    assert to_flores("ja") == "jpn_Jpan"
    assert to_flores("en") == "eng_Latn"
    assert to_flores("es") == "spa_Latn"
    assert to_flores("ko") == "kor_Hang"


def test_latin_ascii_names_in_jp_cue_not_sent_to_mt(tmp_path: Path) -> None:
    src = _srt(
        tmp_path,
        "names.srt",
        [("00:00:00,000 --> 00:00:02,000", "野崎とDrumです")],
    )
    sent: list[str] = []

    class Recorder(EchoBackend):
        def translate(self, texts, src_flores, tgt_flores):
            sent.extend(texts)
            return [t.replace("Drum", "tambor") for t in texts]

    out = tmp_path / "names.es.srt"
    doc = load(src)
    doc.cues[0].extra["lang"] = "ja"
    translate_document(doc, to_code="es", backend=Recorder())
    assert sent
    assert all("Drum" not in t for t in sent)
    assert "Drum" in doc.cues[0].text
    assert "tambor" not in doc.cues[0].text.lower()

    guarded, names = protect_latin_names("野崎とDrumです")
    assert "Drum" in names
    assert "Drum" not in guarded
    assert restore_latin_names(guarded, names).count("Drum") == 1


def test_jsonl_sidecar_roundtrip(tmp_path: Path, monkeypatch) -> None:
    dummy = tmp_path / "clip.mp4"
    dummy.write_bytes(b"fake-media")
    monkeypatch.setattr("sublocal.device.cuda_device_count", lambda: 1)
    monkeypatch.setattr("sublocal.device.cuda_device_name", lambda index=0: "Fake GPU")
    monkeypatch.setattr("sublocal.transcribe.load_whisper", lambda *a, **k: object())
    monkeypatch.setattr("sublocal.transcribe.decode_audio", lambda path: object())

    ja = Segment(
        words=[Word("こんにちは。", 0.50, 1.40)],
        start=0.50,
        end=1.40,
        lang="ja",
    )
    en = Segment(
        words=[Word("hello.", 2.00, 2.80)],
        start=2.00,
        end=2.80,
        lang="en",
    )

    def fake_infer(model, audio, language, progress_cb, **kwargs):
        progress_cb(1.0, 1.0)
        return Transcript([ja, en])

    monkeypatch.setattr("sublocal.transcribe._whisper_infer", fake_infer)
    srt = transcribe_file(dummy, output_path=tmp_path / "clip.srt")
    sidecar = cues_jsonl_path(srt)
    assert sidecar.is_file()
    rows = read_cues_jsonl(sidecar)
    assert [r["lang"] for r in rows] == ["ja", "en"]

    class Recorder(EchoBackend):
        def __init__(self) -> None:
            self.srcs: list[str] = []

        def translate(self, texts, src_flores, tgt_flores):
            self.srcs.append(src_flores)
            return [f"ES:{t}" for t in texts]

    backend = Recorder()
    out = translate_file(srt, to_code="es", output_path=tmp_path / "clip.es.srt", backend=backend)
    dst = load(out)
    assert "jpn_Jpan" in backend.srcs
    assert "eng_Latn" in backend.srcs
    assert all(c.text.startswith("ES:") for c in dst.cues)


def test_product_cli_parses_media_to() -> None:
    assert is_product_argv(["clip.mp4", "--to", "es"])
    assert not is_product_argv(["translate", "file.srt", "--to", "es"])
    assert not is_product_argv(["transcribe", "clip.mp4"])
    args = parse_product_args(["clip.mp4", "--to", "es"])
    assert args.input == "clip.mp4"
    assert args.to == "es"
    assert args.batch is False
    assert args.model is None
    assert args.glossary is None
    assert args.language is None


def test_product_does_not_require_drama_yml(tmp_path: Path, monkeypatch) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake")
    loaded: list[str] = []

    def boom(path):
        loaded.append(str(path))
        raise AssertionError("drama.yml must not load")

    monkeypatch.setattr("sublocal.glossary.Glossary.load", boom)
    monkeypatch.setattr("sublocal.device.cuda_device_count", lambda: 1)
    monkeypatch.setattr("sublocal.device.cuda_device_name", lambda index=0: "Fake GPU")
    monkeypatch.setattr("sublocal.transcribe.load_whisper", lambda *a, **k: object())
    monkeypatch.setattr("sublocal.transcribe.decode_audio", lambda path: object())

    def fake_infer(model, audio, language, progress_cb, **kwargs):
        progress_cb(1.0, 1.0)
        return Transcript(
            [Segment(words=[Word("hello.", 0.0, 1.0)], start=0.0, end=1.0, lang="en")]
        )

    monkeypatch.setattr("sublocal.transcribe._whisper_infer", fake_infer)
    rc = main([str(clip), "--to", "es", "--backend", "echo"])
    assert rc == 0
    assert loaded == []
    assert (tmp_path / "clip.es.srt").is_file()
    assert (tmp_path / "clip.cues.jsonl").is_file()


def test_product_run_writes_es_srt_and_sidecar(tmp_path: Path, monkeypatch) -> None:
    clip = tmp_path / "show.mkv"
    clip.write_bytes(b"fake")
    monkeypatch.setattr("sublocal.device.cuda_device_count", lambda: 1)
    monkeypatch.setattr("sublocal.device.cuda_device_name", lambda index=0: "Fake GPU")
    monkeypatch.setattr("sublocal.transcribe.load_whisper", lambda *a, **k: object())
    monkeypatch.setattr("sublocal.transcribe.decode_audio", lambda path: object())

    def fake_infer(model, audio, language, progress_cb, **kwargs):
        progress_cb(1.0, 1.0)
        return Transcript(
            [
                Segment(
                    words=[Word("hola.", 0.0, 1.0)], start=0.0, end=1.0, lang="es"
                )
            ]
        )

    monkeypatch.setattr("sublocal.transcribe._whisper_infer", fake_infer)
    out = run_product(clip, to_code="es", backend=EchoBackend())
    assert out == tmp_path / "show.es.srt"
    assert out.is_file()
    assert cues_jsonl_path(clip).is_file()
    assert load(out).cues[0].text == "hola."


def test_translate_reads_sidecar_langs(tmp_path: Path) -> None:
    src = _srt(
        tmp_path,
        "in.srt",
        [
            ("00:00:00,000 --> 00:00:01,000", "こんにちは"),
            ("00:00:01,000 --> 00:00:02,000", "hello"),
        ],
    )
    write_cues_jsonl(
        Document(
            format="srt",
            blocks=[
                Block(
                    kind="cue",
                    cue=Cue(
                        index="1",
                        timing="00:00:00,000 --> 00:00:01,000",
                        text="こんにちは",
                        extra={"lang": "ja"},
                    ),
                ),
                Block(
                    kind="cue",
                    cue=Cue(
                        index="2",
                        timing="00:00:01,000 --> 00:00:02,000",
                        text="hello",
                        extra={"lang": "en"},
                    ),
                ),
            ],
        ),
        cues_jsonl_path(src),
    )
    srcs: list[str] = []

    class Rec(EchoBackend):
        def translate(self, texts, src_flores, tgt_flores):
            srcs.append(src_flores)
            return list(texts)

    translate_file(src, to_code="es", backend=Rec(), output_path=tmp_path / "in.es.srt")
    assert srcs == ["jpn_Jpan", "eng_Latn"]


def test_translate_subparser_still_works(tmp_path: Path) -> None:
    src = _srt(
        tmp_path,
        "file.srt",
        [("00:00:00,000 --> 00:00:01,000", "Hola amigos")],
    )
    rc = main(
        [
            "translate",
            str(src),
            "--to",
            "en",
            "--from",
            "es",
            "--backend",
            "echo",
        ]
    )
    assert rc == 0
    assert (tmp_path / "file.en.srt").is_file()


def test_regex_pin_blocks_lid_downgrade() -> None:
    """lingua 1.4.2 pulls regex 2024.11.6; transformers 5.16.1 needs >=2025.10.22."""
    text = Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '"regex>=2025.10.22"' in text
    assert "lingua-language-detector==1.4.2" in text


def test_nllb_src_lang_set_per_encode() -> None:
    from sublocal.backend import NllbBackend, _set_src_lang

    class Tok:
        def __init__(self) -> None:
            self.src_lang = None
            self.calls: list[str] = []

        def set_src_lang_special_tokens(self, code: str) -> None:
            self.calls.append(code)

    tok = Tok()
    _set_src_lang(tok, "jpn_Jpan")
    assert tok.src_lang == "jpn_Jpan"
    assert tok.calls == ["jpn_Jpan"]
    _set_src_lang(tok, "eng_Latn")
    assert tok.src_lang == "eng_Latn"
    assert tok.calls[-1] == "eng_Latn"
    default = NllbBackend(device="cpu")
    assert "3.3B" in default.repo_id
    assert "600M" not in default.repo_id
