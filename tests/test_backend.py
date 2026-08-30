from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from sublocal.backend import (
    DEFAULT_GGUF_FILE,
    DEFAULT_GGUF_REPO,
    DEFAULT_MODEL_REPO,
    DEFAULT_TOKENIZER_REPO,
    GEMMAX_Q5_FILE,
    GEMMAX_REPO,
    SMALL_MODEL_REPO,
    SMALL_TOKENIZER_REPO,
    GemmaXBackend,
    NllbBackend,
    _compute_type,
    _create_translator,
    gemmax_quant_spec,
    nllb_model_spec,
)
from sublocal.pipeline import backend_from_name


def test_default_size_hint_is_6gb() -> None:
    assert "~6GB+" in nllb_model_spec().size_hint


def test_default_repo_is_3_3b_not_600m() -> None:
    assert DEFAULT_MODEL_REPO == "entai2965/nllb-200-3.3B-ctranslate2-float16"
    assert DEFAULT_TOKENIZER_REPO == "facebook/nllb-200-3.3B"
    assert "600M" not in DEFAULT_MODEL_REPO
    spec = nllb_model_spec()
    assert spec.repo_id == DEFAULT_MODEL_REPO
    assert spec.cuda_compute_type == "float16"


def test_model_small_selects_600m() -> None:
    spec = nllb_model_spec("small")
    assert spec.repo_id == SMALL_MODEL_REPO
    assert spec.repo_id == "JustFrederik/nllb-200-distilled-600M-ct2-int8"
    assert spec.tokenizer_repo == SMALL_TOKENIZER_REPO
    assert nllb_model_spec("large").repo_id == DEFAULT_MODEL_REPO
    assert nllb_model_spec("3.3b").repo_id == DEFAULT_MODEL_REPO


def test_nllb_backend_default_and_small() -> None:
    default = NllbBackend(device="cpu")
    assert default.repo_id == DEFAULT_MODEL_REPO
    assert default.tokenizer_repo == DEFAULT_TOKENIZER_REPO
    small = NllbBackend(device="cpu", model="small")
    assert small.repo_id == SMALL_MODEL_REPO
    assert small.tokenizer_repo == SMALL_TOKENIZER_REPO


def test_backend_from_name_model() -> None:
    nllb = backend_from_name("nllb", "cpu", 32, model="small")
    assert isinstance(nllb, NllbBackend)
    assert nllb.repo_id == SMALL_MODEL_REPO
    echo = backend_from_name("echo", "cpu", 32, model="3.3b")
    assert echo.__class__.__name__ == "EchoBackend"
    gemmax = backend_from_name("gemmax", "cpu", 1)
    assert isinstance(gemmax, GemmaXBackend)
    assert gemmax.repo_id == GEMMAX_REPO
    assert gemmax.filename == GEMMAX_Q5_FILE
    auto = backend_from_name("auto", "cpu", 1)
    assert isinstance(auto, GemmaXBackend)


def test_default_mt_is_gemmax_q5_not_nllb() -> None:
    assert DEFAULT_GGUF_REPO == "mradermacher/GemmaX2-28-9B-v0.1-GGUF"
    assert DEFAULT_GGUF_FILE == "GemmaX2-28-9B-v0.1.Q5_K_M.gguf"
    assert "Q4" not in DEFAULT_GGUF_FILE
    spec = gemmax_quant_spec()
    assert spec.filename == GEMMAX_Q5_FILE
    assert spec.size_hint == "6.65 GB"
    q6 = gemmax_quant_spec("q6")
    assert q6.filename.endswith("Q6_K.gguf")
    default = GemmaXBackend(device="cpu")
    assert default.filename == GEMMAX_Q5_FILE
    assert default.repo_id == GEMMAX_REPO
    with pytest.raises(ValueError, match="not the product default"):
        gemmax_quant_spec("3.3b")


def test_compute_type_cuda_float16_cpu_int8() -> None:
    assert _compute_type("cuda") == "float16"
    assert _compute_type("cuda", nllb_model_spec("3.3b")) == "float16"
    assert _compute_type("cuda", nllb_model_spec("small")) == "int8_float16"
    assert _compute_type("cpu") == "int8"
    assert _compute_type("cpu", nllb_model_spec("3.3b")) == "int8"


def test_create_translator_oom_retries_int8_float16(monkeypatch, capsys) -> None:
    calls: list[str] = []

    class FakeTranslator:
        def __init__(self, model_dir, device, compute_type):
            calls.append(compute_type)
            if compute_type == "float16":
                raise RuntimeError("CUDA out of memory")

    monkeypatch.setitem(
        sys.modules, "ctranslate2", SimpleNamespace(Translator=FakeTranslator)
    )
    _create_translator("/tmp/model", "cuda", "float16")
    assert calls == ["float16", "int8_float16"]
    err = capsys.readouterr().err
    assert "int8_float16" in err
    assert "out of memory" in err.lower() or "OOM" in err or "memory" in err.lower()


def test_ensure_loaded_unloads_whisper_before_translator(monkeypatch) -> None:
    import types

    order: list[str] = []

    def unload(model=None):
        order.append("unload")

    monkeypatch.setattr("sublocal.transcribe.unload_whisper", unload)
    monkeypatch.setattr("sublocal.backend._quiet_ct2_debug", lambda: None)
    tf = types.ModuleType("transformers")
    tf_utils = types.ModuleType("transformers.utils")
    tf_logging = types.ModuleType("transformers.utils.logging")
    tf_logging.set_verbosity_error = lambda: None
    tf.utils = tf_utils
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "transformers.utils", tf_utils)
    monkeypatch.setitem(sys.modules, "transformers.utils.logging", tf_logging)
    backend = NllbBackend(device="cpu")
    monkeypatch.setattr(backend, "_snapshot", lambda *a, **k: ("/tmp/m", True))
    monkeypatch.setattr(backend, "_load_tokenizer", lambda cache: (object(), True))
    monkeypatch.setattr(
        "sublocal.backend._create_translator",
        lambda *a, **k: order.append("translator") or object(),
    )
    backend._ensure_loaded()
    assert order[0] == "unload"
    assert "translator" in order


def test_create_llama_oom_retries_n_ctx_1024(monkeypatch, capsys) -> None:
    from sublocal.backend import _create_llama

    calls: list[int] = []

    class FakeLlama:
        def __init__(self, model_path, n_gpu_layers, n_ctx, logits_all, **kwargs):
            calls.append(n_ctx)
            assert n_gpu_layers == -1
            assert logits_all is False
            self.chat_format = "gemma"
            self.chat_handler = object()
            if n_ctx == 2048:
                raise RuntimeError("CUDA out of memory")

    monkeypatch.setitem(
        sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama)
    )
    llama = _create_llama("/tmp/fake.gguf", n_ctx=2048)
    assert calls == [2048, 1024]
    assert llama.chat_format is None
    assert llama.chat_handler is None
    err = capsys.readouterr().err
    assert "1024" in err


def test_gemmax_ensure_loaded_unloads_whisper_before_llama(monkeypatch) -> None:
    order: list[str] = []

    def unload(model=None):
        order.append("unload")

    monkeypatch.setattr("sublocal.transcribe.unload_whisper", unload)
    monkeypatch.setattr(
        "sublocal.backend._create_llama",
        lambda *a, **k: order.append("llama") or object(),
    )
    backend = GemmaXBackend(device="cpu")
    monkeypatch.setattr(backend, "_download_gguf", lambda: "/tmp/fake.gguf")
    backend._ensure_loaded()
    assert order[0] == "unload"
    assert "llama" in order
    assert order.index("unload") < order.index("llama")


def test_create_translator_non_oom_does_not_retry(monkeypatch) -> None:
    class FakeTranslator:
        def __init__(self, model_dir, device, compute_type):
            raise RuntimeError("weights not found")

    monkeypatch.setitem(
        sys.modules, "ctranslate2", SimpleNamespace(Translator=FakeTranslator)
    )
    with pytest.raises(RuntimeError, match="weights not found"):
        _create_translator("/tmp/model", "cuda", "float16")
