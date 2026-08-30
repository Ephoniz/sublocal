from __future__ import annotations

from pathlib import Path

import pytest

from sublocal.cli import main
from sublocal.formats import load
from sublocal.glossary import Glossary
from sublocal.transcribe import (
    MAX_CUE_DURATION_S,
    Segment,
    Transcript,
    Word,
    _whisper_infer,
    apply_regroup,
    decode_audio,
    default_output_path,
    format_srt_timestamp,
    transcript_to_document,
    transcribe_file,
    validate_model,
    write_transcript_srt,
)

DRAMA = Path(__file__).resolve().parents[1] / "examples" / "drama.yml"


def _ja_chars(text: str, start: float, step: float = 0.12) -> list[Word]:
    words: list[Word] = []
    t = start
    for ch in text:
        words.append(Word(word=ch, start=round(t, 3), end=round(t + step, 3)))
        t += step
    return words


def _fake_30s_window() -> Transcript:
    """One raw Whisper 0→30s window with several Japanese sentences."""
    words: list[Word] = []
    # Sentence 1. Ends 1.40; next sentence after 0.20s (punct split, no remelt).
    words.extend(_ja_chars("こんにちは。", 0.50, 0.15))
    # Sentence 2. Ends ~3.10; 0.70s gap before the next token (gap split).
    words.extend(_ja_chars("元気ですか？", 1.60, 0.15))
    # Sentence 3 after ≥0.5s silence.
    words.extend(_ja_chars("はい。", 3.80, 0.15))
    # Two short fragments with a 0.10s gap — merge_by_gap should join them.
    words.append(Word(word="あ", start=6.00, end=6.15))
    words.append(Word(word="あ。", start=6.25, end=6.40))
    # Long sentence (>64 chars) so wrap hits ~32 / two lines, not 0→30.
    long = "あ" * 70 + "。"
    assert len(long) > 64
    words.extend(_ja_chars(long, 8.00, 0.10))
    return Transcript([Segment(words=words, start=0.0, end=30.0)])


def test_default_output_is_srt_suffix() -> None:
    assert default_output_path(Path("movie.mp4")) == Path("movie.srt")
    assert default_output_path(Path("/tmp/show.mkv")) == Path("/tmp/show.srt")


def test_validate_model_rejects_bigger() -> None:
    assert validate_model("large-v3") == "large-v3"
    assert validate_model("small") == "small"
    with pytest.raises(ValueError, match="larger than large-v3"):
        validate_model("large-v4")
    with pytest.raises(ValueError, match="larger than large-v3"):
        validate_model("Systran/faster-whisper-large-v4")


def test_regroup_sentence_cues_not_30s_or_per_word() -> None:
    result = apply_regroup(_fake_30s_window())
    doc = transcript_to_document(result)
    assert len(doc.cues) >= 3
    assert len(doc.cues) < len(_fake_30s_window().segments[0].words)

    texts = [c.text.replace("\n", "") for c in doc.cues]
    assert any("こんにちは。" in t for t in texts)
    assert any("元気ですか？" in t for t in texts)
    assert any(t == "はい。" or t.startswith("はい。") for t in texts)

    timings = []
    for cue in doc.cues:
        start_s, end_s = cue.timing.split(" --> ")
        timings.append((start_s, end_s))
        assert start_s != "00:00:00,000" or end_s != "00:00:30,000"
        assert end_s != "00:00:30,000"

    # Word-derived: first cue starts at こんにちは (0.50), not the raw window.
    assert timings[0][0] == format_srt_timestamp(0.50)
    # No cue spans the fake 30s window.
    for start_s, end_s in timings:
        assert not (start_s == "00:00:00,000" and end_s == "00:00:30,000")


def test_split_by_duration_caps_8s_segment() -> None:
    words = [Word(word="あ", start=i * 0.1, end=(i + 1) * 0.1) for i in range(80)]
    tr = Transcript([Segment(words=words)])
    tr.split_by_duration(4.0)
    assert len(tr.segments) >= 2
    for seg in tr.segments:
        assert seg.words[-1].end - seg.words[0].start <= 4.0 + 1e-9
    span = tr.segments[-1].words[-1].end - tr.segments[0].words[0].start
    assert span == pytest.approx(8.0)


def test_apply_regroup_gap_0_5_still_splits_and_duration_caps() -> None:
    # Two phrases with a 0.60s gap (must stay split) plus an 8s run.
    words = [
        Word(word="あ", start=0.00, end=0.20),
        Word(word="い", start=0.20, end=0.40),
        Word(word="う", start=1.00, end=1.20),
        Word(word="え", start=1.20, end=1.40),
    ]
    words.extend(Word(word="ん", start=2.0 + i * 0.5, end=2.0 + i * 0.5 + 0.4) for i in range(16))
    result = apply_regroup(Transcript([Segment(words=words)]))
    texts = [s.cue_text() for s in result.segments if s.words]
    assert any(t == "あい" or t.startswith("あい") for t in texts)
    assert any("うえ" in t or t == "うえ" for t in texts)
    # 0.60s gap must not remelt あい and うえ.
    joined_early = [t for t in texts if "あい" in t and "うえ" in t]
    assert joined_early == []
    for seg in result.segments:
        if not seg.words:
            continue
        assert seg.words[-1].end - seg.words[0].start <= MAX_CUE_DURATION_S + 1e-9


def test_japanese_punctuation_and_gap_and_merge() -> None:
    result = apply_regroup(_fake_30s_window())
    texts = [c.text.replace("\n", "") for c in transcript_to_document(result).cues]
    # Punctuation produced separate sentence cues (gap after 。 is 0.20s > 0.15).
    konnichiwa = [t for t in texts if t.startswith("こんにちは")]
    genki = [t for t in texts if "元気ですか" in t]
    assert konnichiwa
    assert genki
    assert konnichiwa[0] != genki[0] or "元気" not in konnichiwa[0]

    # 0.70s gap after ？ splits はい。
    assert any(t == "はい。" for t in texts)

    # Tiny 0.10s gap merged into one cue, not two one-char flickers.
    aa = [t for t in texts if t == "ああ。" or t.startswith("ああ")]
    assert aa
    assert "あ。" not in texts or any(t == "ああ。" for t in texts)


def test_max_chars_two_lines(tmp_path: Path) -> None:
    result = apply_regroup(_fake_30s_window())
    out = tmp_path / "out.srt"
    n = write_transcript_srt(result, out)
    assert n == len(result.segments)
    doc = load(out)
    for cue in doc.cues:
        lines = cue.text.split("\n")
        assert 1 <= len(lines) <= 2
        for line in lines:
            assert len(line) <= 32


def test_cli_transcribe_cuda_count_zero_exits_1(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    dummy = tmp_path / "movie.mp4"
    dummy.write_bytes(b"fake-media")
    monkeypatch.setattr("sublocal.device.cuda_device_count", lambda: 0)
    loaded: list[str] = []

    def boom(*_a, **_k):
        loaded.append("load")
        raise AssertionError("transcribe must not run")

    monkeypatch.setattr("sublocal.transcribe.load_whisper", boom)
    monkeypatch.setattr("sublocal.transcribe._whisper_infer", boom)
    rc = main(["transcribe", str(dummy), "--device", "cuda"])
    assert rc == 1
    assert loaded == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    assert "get_cuda_device_count()=0" in captured.err


def test_cli_transcribe_mocked_writes_srt(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    dummy = tmp_path / "movie.mp4"
    dummy.write_bytes(b"fake-media")
    monkeypatch.setattr("sublocal.device.cuda_device_count", lambda: 1)
    monkeypatch.setattr(
        "sublocal.device.cuda_device_name",
        lambda index=0: "NVIDIA GeForce RTX 4070 Ti",
    )
    monkeypatch.setattr("sublocal.transcribe.load_whisper", lambda *a, **k: object())
    monkeypatch.setattr("sublocal.transcribe.decode_audio", lambda path: object())
    chained: list[str] = []
    monkeypatch.setattr(
        "sublocal.pipeline.translate_file",
        lambda *a, **k: chained.append("translate"),
    )

    seen: dict[str, str | None] = {}

    def fake_infer(model, path, language, progress_cb):
        seen["language"] = language
        progress_cb(12.0, 45.0)
        progress_cb(45.0, 45.0)
        return _fake_30s_window()

    monkeypatch.setattr("sublocal.transcribe._whisper_infer", fake_infer)
    rc = main(["transcribe", str(dummy), "--language", "ja"])
    assert rc == 0
    assert seen["language"] == "ja"
    assert chained == []
    out = tmp_path / "movie.srt"
    captured = capsys.readouterr()
    assert captured.out.strip() == str(out)
    assert "Using NVIDIA GeForce RTX 4070 Ti (cuda:0)" in captured.err
    assert "12/45s (26%)" in captured.err
    assert "45/45s (100%)" in captured.err
    assert "Wrote" in captured.err
    assert "cues" in captured.err
    doc = load(out)
    assert len(doc.cues) >= 3
    assert out.is_file()


def test_cli_transcribe_missing_file(tmp_path: Path, capsys) -> None:
    rc = main(["transcribe", str(tmp_path / "nope.mp4")])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    assert "not found" in captured.err.lower() or "Nope" in captured.err or "nope" in captured.err


def test_transcribe_glossary_prompt_and_jp_keys(
    tmp_path: Path, monkeypatch
) -> None:
    dummy = tmp_path / "movie.mp4"
    dummy.write_bytes(b"fake-media")
    monkeypatch.setattr("sublocal.device.cuda_device_count", lambda: 1)
    monkeypatch.setattr("sublocal.device.cuda_device_name", lambda index=0: "Fake GPU")
    monkeypatch.setattr("sublocal.transcribe.load_whisper", lambda *a, **k: object())
    monkeypatch.setattr("sublocal.transcribe.decode_audio", lambda path: object())
    seen: dict = {}

    def fake_infer(model, audio, language, progress_cb, **kwargs):
        seen["language"] = language
        seen.update(kwargs)
        progress_cb(1.0, 1.0)
        return Transcript([Segment(words=_ja_chars("野崎のドラム。", 0.50))])

    monkeypatch.setattr("sublocal.transcribe._whisper_infer", fake_infer)
    out = transcribe_file(
        dummy, language="ja", glossary=DRAMA, output_path=tmp_path / "out.srt"
    )
    g = Glossary.load(DRAMA)
    assert seen["language"] == "ja"
    assert seen["initial_prompt"] == g.whisper_prompt()
    assert "ドラム" in seen["initial_prompt"]
    assert seen["condition_on_previous_text"] is False
    doc = load(out)
    text = "".join(c.text for c in doc.cues)
    assert "ドラム" in text
    assert "野崎" in text
    assert "Drum" not in text
    assert "Nozaki" not in text


def test_transcribe_rejects_to_flag() -> None:
    with pytest.raises(SystemExit):
        main(["transcribe", "movie.mp4", "--to", "en"])


def test_regroup_from_plain_whisper_standin() -> None:
    class Seg:
        def __init__(self) -> None:
            self.start = 0.0
            self.end = 30.0
            self.words = _fake_30s_window().segments[0].words
            self.text = "raw-30s-window"

    class Result:
        segments = [Seg()]

    doc = transcript_to_document(apply_regroup(Result()))
    assert len(doc.cues) >= 3
    assert all(" --> " in c.timing for c in doc.cues)
    assert all(not c.timing.startswith("00:00:00,000 --> 00:00:30,000") for c in doc.cues)


def test_decode_audio_returns_1d_float32(monkeypatch, tmp_path) -> None:
    np = pytest.importorskip("numpy")
    import sys
    from types import SimpleNamespace

    class Frame:
        def to_ndarray(self):
            return np.array([[1000, -1000, 0, 500]], dtype=np.int16)

    class Container:
        streams = SimpleNamespace(audio=[object()])

        def decode(self, audio=0):
            yield Frame()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def make_resampler(**kwargs):
        assert kwargs.get("layout") == "mono"
        assert kwargs.get("rate") == 16000

        class _Resampler:
            def resample(self, frame):
                if frame is None:
                    return []
                return [frame]

        return _Resampler()

    fake_av = SimpleNamespace(
        audio=SimpleNamespace(resampler=SimpleNamespace(AudioResampler=make_resampler)),
        error=SimpleNamespace(InvalidDataError=type("InvalidDataError", (Exception,), {})),
        open=lambda path, mode="r", **kwargs: Container(),
    )
    monkeypatch.setitem(sys.modules, "av", fake_av)
    audio = decode_audio(tmp_path / "clip.mkv")
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert audio.size == 4
    assert abs(float(audio[0]) - (1000 / 32768.0)) < 1e-6


def test_decode_audio_from_generated_wav(tmp_path) -> None:
    pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    import struct
    import wave

    path = tmp_path / "tone.wav"
    sr = 16000
    n = 1600
    with wave.open(str(path), "w") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(struct.pack("<" + "h" * n, *([1000] * n)))
    audio = decode_audio(path)
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert abs(audio.size - n) < sr // 10


def test_transcribe_passes_ndarray_not_path(tmp_path, monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    dummy = tmp_path / "movie.mp4"
    dummy.write_bytes(b"fake-media")
    wave = np.zeros(1600, dtype=np.float32)
    monkeypatch.setattr("sublocal.transcribe.decode_audio", lambda path: wave)
    monkeypatch.setattr("sublocal.device.cuda_device_count", lambda: 1)
    monkeypatch.setattr("sublocal.device.cuda_device_name", lambda index=0: "Fake GPU")
    monkeypatch.setattr("sublocal.transcribe.load_whisper", lambda *a, **k: object())

    seen: dict = {}

    def fake_infer(model, audio, language, progress_cb):
        seen["audio"] = audio
        progress_cb(1.0, 1.0)
        return _fake_30s_window()

    monkeypatch.setattr("sublocal.transcribe._whisper_infer", fake_infer)
    transcribe_file(dummy)
    assert isinstance(seen["audio"], np.ndarray)
    assert seen["audio"].dtype == np.float32
    assert seen["audio"].ndim == 1
    assert not isinstance(seen["audio"], (str, Path))


def test_whisper_infer_calls_transcribe_with_ndarray() -> None:
    np = pytest.importorskip("numpy")
    wave = np.zeros(800, dtype=np.float32)
    seen: dict = {}

    class Model:
        def transcribe(self, audio, **kwargs):
            seen["audio"] = audio
            seen["kwargs"] = kwargs
            return _fake_30s_window()

    _whisper_infer(Model(), wave, "ja", lambda *_a: None)
    assert seen["audio"] is wave
    assert isinstance(seen["audio"], np.ndarray)
    assert seen["kwargs"]["vad_filter"] is True
    assert seen["kwargs"]["vad_parameters"]["min_silence_duration_ms"] == 500
    assert seen["kwargs"].get("vad") is True


def test_seek_progress_throttles(monkeypatch, capsys) -> None:
    from sublocal.transcribe import SeekProgress

    class Clock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            return self.t

    clock = Clock()
    monkeypatch.setattr("sublocal.transcribe.time.monotonic", clock)
    prog = SeekProgress()
    clock.t = 0.1
    prog(5.0, 100.0)
    clock.t = 0.6
    prog(6.0, 100.0)  # +1% and 0.5s — throttled
    err1 = capsys.readouterr().err
    assert "5/100s (5%)" in err1
    assert "6/100s" not in err1
    clock.t = 3.0
    prog(20.0, 100.0)  # +15% and ≥2s — prints
    prog.finish()
    err2 = capsys.readouterr().err
    assert "20/100s (20%)" in err2
    assert "100/100s (100%)" in err2
