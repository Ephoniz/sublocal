"""v0.5 product: mixed-language ASR → per-cue GemmaX2. CPU, no 9B download."""

from __future__ import annotations

from pathlib import Path

from sublocal.backend import (
    GEMMAX_Q5_FILE,
    GEMMAX_REPO,
    GemmaXBackend,
    NllbBackend,
    gemmax_prompt,
    maybe_name_instruction,
    strip_gemmax_completion,
)
from sublocal.cli import build_parser, is_product_argv, main, parse_product_args
from sublocal.cues_jsonl import cues_jsonl_path
from sublocal.formats import load
from sublocal.lid import protect_latin_names
from sublocal.pipeline import backend_from_name, run_product, translate_file
from sublocal.transcribe import Segment, Transcript, Word


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


class FakeLlama:
    """Stand-in for llama_cpp.Llama. No GGUF, no download."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        return {"choices": [{"text": "Hola"}]}


def test_product_and_translate_still_parse() -> None:
    assert is_product_argv(["clip.mp4", "--to", "es"])
    assert not is_product_argv(["translate", "file.srt", "--to", "es"])
    args = parse_product_args(["clip.mp4", "--to", "es"])
    assert args.to == "es"
    assert args.backend == "gemmax"
    assert args.model is None
    parser = build_parser()
    tr = parser.parse_args(["translate", "file.srt", "--to", "es"])
    assert tr.backend == "gemmax"
    assert tr.model is None
    assert tr.gguf is None


def test_copy_through_if_lang_equals_to(tmp_path: Path) -> None:
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
    llama = FakeLlama()
    backend = GemmaXBackend(device="cpu")
    backend._llama = llama
    from sublocal.pipeline import translate_document

    translate_document(doc, to_code="es", backend=backend)
    assert doc.cues[0].text == "Already Spanish"
    assert doc.cues[1].text == "Hola"
    assert len(llama.prompts) == 1
    assert "Already Spanish" not in llama.prompts[0]


def test_prompt_is_completion_template_english_names() -> None:
    prompt = gemmax_prompt("Japanese", "Spanish", "野崎です")
    assert prompt == (
        "Translate this from Japanese to Spanish:\n"
        "Japanese: 野崎です\n"
        "Spanish:"
    )
    assert "jpn_Jpan" not in prompt
    assert "spa_Latn" not in prompt
    assert "messages" not in prompt
    assert "<|im_start|>" not in prompt
    assert "role" not in prompt
    llama = FakeLlama()
    backend = GemmaXBackend(device="cpu")
    backend._llama = llama
    backend.translate(["野崎です"], "jpn_Jpan", "spa_Latn")
    assert llama.prompts == [prompt]
    assert llama.kwargs[0]["temperature"] == 0
    assert llama.kwargs[0]["top_k"] == 1
    assert llama.kwargs[0]["max_tokens"] == 256
    assert maybe_name_instruction("野崎です") is False
    hinted = gemmax_prompt("Japanese", "Spanish", "野崎です", name_hint=True)
    assert hinted.startswith("Keep person and place names.")
    assert "Translate this from Japanese to Spanish:" in hinted


def test_strip_completion_after_target_header() -> None:
    raw = (
        "Translate this from Japanese to Spanish:\n"
        "Japanese: こんにちは\n"
        "Spanish:\nHola amigos\n\nMore"
    )
    assert strip_gemmax_completion(raw, "Spanish") == "Hola amigos"
    assert strip_gemmax_completion("Hola amigos\nsegunda", "Spanish") == "Hola amigos"


def test_latin_copy_through_protects_liu_zhang_before_model(tmp_path: Path) -> None:
    src = _srt(
        tmp_path,
        "names.srt",
        [("00:00:00,000 --> 00:00:02,000", "LiuとZhangです")],
    )
    llama = FakeLlama()
    backend = GemmaXBackend(device="cpu")
    backend._llama = llama
    doc = load(src)
    doc.cues[0].extra["lang"] = "ja"
    from sublocal.pipeline import translate_document

    translate_document(doc, to_code="es", backend=backend)
    assert llama.prompts
    sent = llama.prompts[0]
    assert "Liu" not in sent.split("Japanese:", 1)[-1]
    assert "Zhang" not in sent.split("Japanese:", 1)[-1]
    assert "Liu" in doc.cues[0].text
    assert "Zhang" in doc.cues[0].text
    guarded, names = protect_latin_names("LiuとZhangです")
    assert names == ["Liu", "Zhang"]
    assert "Liu" not in guarded
    assert "Zhang" not in guarded


def test_default_backend_is_gemmax_not_nllb() -> None:
    default = backend_from_name("gemmax", "cpu", 1)
    assert isinstance(default, GemmaXBackend)
    assert not isinstance(default, NllbBackend)
    assert default.repo_id == GEMMAX_REPO
    assert default.filename == GEMMAX_Q5_FILE
    auto = backend_from_name("auto", "cpu", 1)
    assert isinstance(auto, GemmaXBackend)
    nllb = backend_from_name("nllb", "cpu", 32)
    assert isinstance(nllb, NllbBackend)


def test_run_product_unloads_whisper_before_gguf(
    tmp_path: Path, monkeypatch
) -> None:
    clip = tmp_path / "show.mkv"
    clip.write_bytes(b"fake")
    order: list[str] = []

    monkeypatch.setattr("sublocal.device.cuda_device_count", lambda: 1)
    monkeypatch.setattr("sublocal.device.cuda_device_name", lambda index=0: "Fake GPU")
    monkeypatch.setattr("sublocal.transcribe.load_whisper", lambda *a, **k: object())
    monkeypatch.setattr("sublocal.transcribe.decode_audio", lambda path: object())

    def fake_infer(model, audio, language, progress_cb, **kwargs):
        progress_cb(1.0, 1.0)
        return Transcript(
            [Segment(words=[Word("こんにちは", 0.0, 1.0)], start=0.0, end=1.0, lang="ja")]
        )

    monkeypatch.setattr("sublocal.transcribe._whisper_infer", fake_infer)

    real_unload = __import__(
        "sublocal.transcribe", fromlist=["unload_whisper"]
    ).unload_whisper

    def tracked_unload(model=None):
        order.append("unload")
        return real_unload(model)

    monkeypatch.setattr("sublocal.pipeline.unload_whisper", tracked_unload)
    monkeypatch.setattr("sublocal.transcribe.unload_whisper", tracked_unload)

    llama = FakeLlama()
    backend = GemmaXBackend(device="cpu")
    backend._llama = llama

    def prepare() -> None:
        order.append("gguf")

    backend.prepare = prepare  # type: ignore[method-assign]
    out = run_product(clip, to_code="es", backend=backend)
    assert out == tmp_path / "show.es.srt"
    assert "unload" in order
    assert "gguf" in order
    assert order.index("unload") < order.index("gguf")
    assert llama.prompts
    assert "Translate this from Japanese to Spanish:" in llama.prompts[0]


def test_translate_file_unloads_whisper_before_gguf(
    tmp_path: Path, monkeypatch
) -> None:
    src = _srt(
        tmp_path,
        "in.srt",
        [("00:00:00,000 --> 00:00:01,000", "こんにちは")],
    )
    order: list[str] = []

    def tracked_unload(model=None):
        order.append("unload")

    monkeypatch.setattr("sublocal.pipeline.unload_whisper", tracked_unload)
    llama = FakeLlama()
    backend = GemmaXBackend(device="cpu")
    backend._llama = llama

    def prepare() -> None:
        order.append("gguf")

    backend.prepare = prepare  # type: ignore[method-assign]
    translate_file(src, to_code="es", backend=backend, output_path=tmp_path / "in.es.srt")
    assert order.index("unload") < order.index("gguf")


def test_translate_cli_fake_llama(tmp_path: Path, monkeypatch, capsys) -> None:
    src = _srt(
        tmp_path,
        "file.srt",
        [("00:00:00,000 --> 00:00:01,000", "こんにちは")],
    )
    llama = FakeLlama()

    def fake_backend(name, device, batch_size, model=None, gguf=None, name_hint=False):
        backend = GemmaXBackend(device="cpu", gguf=gguf, name_hint=name_hint)
        backend._llama = llama
        return backend

    monkeypatch.setattr("sublocal.cli.backend_from_name", fake_backend)
    rc = main(
        [
            "translate",
            str(src),
            "--to",
            "es",
            "--from",
            "ja",
        ]
    )
    assert rc == 0
    assert (tmp_path / "file.es.srt").is_file()
    captured = capsys.readouterr()
    assert captured.out.strip() == str(tmp_path / "file.es.srt")
    assert "MT pass" in captured.err
    assert llama.prompts
    assert "Translate this from Japanese to Spanish:" in llama.prompts[0]
    assert "Japanese: こんにちは" in llama.prompts[0]


def test_gemmax_does_not_download_when_llama_injected() -> None:
    backend = GemmaXBackend(device="cpu")
    backend._llama = FakeLlama()
    out = backend.translate(["hello"], "eng_Latn", "spa_Latn")
    assert out == ["Hola"]
