"""Translation backends. Default is GemmaX2-28-9B-v0.1 Q5_K_M via llama-cpp."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sublocal.cache import hf_cache_dir
from sublocal.device import resolve_device
from sublocal.languages import to_english_name
from sublocal.progress import (
    BatchCounter,
    enable_download_progress,
    status,
    stderr_tqdm_class,
)
from sublocal.runtime import reject_anaconda_on_windows

# Match the CLI promise: no telemetry. Must be set before huggingface_hub import.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Product default: GemmaX2-28-9B-v0.1 Q5_K_M GGUF (mradermacher). Not Q4.
# Card: ModelSpace/GemmaX2-28-9B-v0.1. Paper: arxiv 2502.02481.
GEMMAX_REPO = "mradermacher/GemmaX2-28-9B-v0.1-GGUF"
GEMMAX_Q5_FILE = "GemmaX2-28-9B-v0.1.Q5_K_M.gguf"
GEMMAX_Q6_FILE = "GemmaX2-28-9B-v0.1.Q6_K.gguf"
GEMMAX_Q5_SIZE = "6.65 GB"
GEMMAX_Q6_SIZE = "7.59 GB"
DEFAULT_GGUF_REPO = GEMMAX_REPO
DEFAULT_GGUF_FILE = GEMMAX_Q5_FILE

# Optional --backend nllb. Not the product default.
DEFAULT_MODEL_REPO = "entai2965/nllb-200-3.3B-ctranslate2-float16"
DEFAULT_TOKENIZER_REPO = "facebook/nllb-200-3.3B"
SMALL_MODEL_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
SMALL_TOKENIZER_REPO = "facebook/nllb-200-distilled-600M"
NLLB_DEFAULT_MODEL_REPO = DEFAULT_MODEL_REPO

NAME_HINT_SENTENCE = (
    "Keep person and place names. Romanize Japanese names (Hepburn). "
    "Do not translate names by meaning."
)
DEFAULT_N_CTX = 2048
OOM_N_CTX = 1024
DEFAULT_MAX_TOKENS = 256


@dataclass(frozen=True)
class NllbModelSpec:
    name: str
    repo_id: str
    tokenizer_repo: str
    size_hint: str
    cuda_compute_type: str


LARGE_SPEC = NllbModelSpec(
    name="3.3b",
    repo_id=DEFAULT_MODEL_REPO,
    tokenizer_repo=DEFAULT_TOKENIZER_REPO,
    size_hint="~6GB+",
    cuda_compute_type="float16",
)
SMALL_SPEC = NllbModelSpec(
    name="small",
    repo_id=SMALL_MODEL_REPO,
    tokenizer_repo=SMALL_TOKENIZER_REPO,
    size_hint="CTranslate2 int8, ~600 MB",
    cuda_compute_type="int8_float16",
)

_MODEL_ALIASES = {
    "small": SMALL_SPEC,
    "600m": SMALL_SPEC,
    "3.3b": LARGE_SPEC,
    "large": LARGE_SPEC,
}


def nllb_model_spec(name: str | None = None) -> NllbModelSpec:
    """Resolve ``small`` / ``3.3b`` / ``large``. Default is 3.3B, not 600M."""
    key = (name or "3.3b").strip().lower()
    try:
        return _MODEL_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown NLLB model {name!r}. Use small, 3.3b, or large."
        ) from exc

# Skip special tokens that CTranslate2 may leave on the hypothesis.
_SKIP_TOKENS = {"</s>", "<s>", "<pad>", "<unk>"}


def _set_src_lang(tokenizer: object, src_flores: str) -> None:
    """Set NLLB source language immediately before encode (per cue / group)."""
    tokenizer.src_lang = src_flores  # type: ignore[attr-defined]
    setter = getattr(tokenizer, "set_src_lang_special_tokens", None)
    if callable(setter):
        setter(src_flores)


class TranslatorBackend(Protocol):
    def translate(
        self, texts: list[str], src_flores: str, tgt_flores: str
    ) -> list[str]: ...


class EchoBackend:
    """Returns cue text unchanged. Used by tests; no model download."""

    def prepare(self) -> None:
        return None

    def translate(
        self, texts: list[str], src_flores: str, tgt_flores: str
    ) -> list[str]:
        return list(texts)


def maybe_name_instruction(text: str) -> bool:
    """Gate for an optional second-pass name sentence. Off in v0.5.0 first-pass."""
    return False


def gemmax_prompt(
    src_name: str,
    tgt_name: str,
    cue: str,
    *,
    name_hint: bool = False,
) -> str:
    """In-distribution GemmaX2 completion prompt. Not a chat template."""
    prefix = ""
    if name_hint:
        prefix = f"{NAME_HINT_SENTENCE}\n"
    return (
        f"{prefix}Translate this from {src_name} to {tgt_name}:\n"
        f"{src_name}: {cue}\n"
        f"{tgt_name}:"
    )


def strip_gemmax_completion(text: str, tgt_name: str) -> str:
    """Drop any echoed prompt; keep the first line/paragraph after the header."""
    header = f"{tgt_name}:"
    if header in text:
        text = text.rsplit(header, 1)[-1]
    text = text.strip()
    if not text:
        return ""
    para = text.split("\n\n", 1)[0].strip()
    return para.split("\n", 1)[0].strip()


@dataclass(frozen=True)
class GemmaxQuantSpec:
    name: str
    filename: str
    size_hint: str


Q5_SPEC = GemmaxQuantSpec(name="q5", filename=GEMMAX_Q5_FILE, size_hint=GEMMAX_Q5_SIZE)
Q6_SPEC = GemmaxQuantSpec(name="q6", filename=GEMMAX_Q6_FILE, size_hint=GEMMAX_Q6_SIZE)

_GEMMAX_QUANT_ALIASES = {
    "q5": Q5_SPEC,
    "q5_k_m": Q5_SPEC,
    "q5_km": Q5_SPEC,
    "q6": Q6_SPEC,
    "q6_k": Q6_SPEC,
}
_NLLB_MODEL_KEYS = frozenset({"small", "600m", "3.3b", "large"})


def gemmax_quant_spec(name: str | None = None) -> GemmaxQuantSpec:
    """Resolve ``q5`` / ``q6``. Default is Q5_K_M, not Q4."""
    raw = (name or "q5").strip().lower().replace("-", "_")
    if raw in _NLLB_MODEL_KEYS:
        raise ValueError(
            f"NLLB model {name!r} is not the product default. "
            "Default MT is GemmaX2 Q5_K_M. Pass --backend nllb for NLLB, "
            "or --model q5 / q6 for a GemmaX2 quant."
        )
    try:
        return _GEMMAX_QUANT_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(
            f"Unknown GemmaX2 quant {name!r}. Use q5 (default Q5_K_M) or q6 (Q6_K)."
        ) from exc


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda oom" in msg
        or "cublas_status_alloc_failed" in msg
        or "ggml_cuda" in msg
        or "failed to allocate" in msg
    )


def _create_llama(model_path: str, n_ctx: int = DEFAULT_N_CTX):
    """Build llama-cpp Llama. CUDA OOM retries once with n_ctx=1024."""
    from llama_cpp import Llama

    try:
        return Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            logits_all=False,
        )
    except Exception as exc:
        if n_ctx > OOM_N_CTX and _is_oom(exc):
            status(f"CUDA out of memory; retrying with n_ctx={OOM_N_CTX}")
            return Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                n_ctx=OOM_N_CTX,
                logits_all=False,
            )
        raise


class GemmaXBackend:
    """GemmaX2-28-9B-v0.1 Q5_K_M GGUF via llama-cpp-python. Not transformers."""

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 1,
        repo_id: str | None = None,
        model: str | None = None,
        gguf: str | Path | None = None,
        name_hint: bool = False,
        n_ctx: int = DEFAULT_N_CTX,
    ) -> None:
        self.spec = gemmax_quant_spec(model)
        self.device = resolve_device(device)
        self.batch_size = max(1, batch_size)
        self.repo_id = repo_id or os.environ.get("SUBLOCAL_GGUF_REPO") or GEMMAX_REPO
        self.filename = self.spec.filename
        self.gguf = Path(gguf).expanduser() if gguf else None
        if self.gguf is None:
            env_gguf = os.environ.get("SUBLOCAL_GGUF")
            if env_gguf:
                self.gguf = Path(env_gguf).expanduser()
        self.name_hint = name_hint
        self.n_ctx = n_ctx
        self._llama = None

    def prepare(self) -> None:
        self._ensure_loaded()

    def _download_gguf(self) -> str:
        if self.gguf is not None:
            path = self.gguf
            if not path.is_file():
                raise FileNotFoundError(f"GGUF not found: {path}")
            return str(path)
        from huggingface_hub import hf_hub_download

        cache = str(hf_cache_dir())
        missing = _local_entry_error()
        try:
            return hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                cache_dir=cache,
                local_files_only=True,
            )
        except missing:
            enable_download_progress()
            status(
                f"Downloading {self.repo_id} {self.filename} ({self.spec.size_hint}) "
                f"into {cache}. No Hugging Face token required."
            )
            try:
                return hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    cache_dir=cache,
                    tqdm_class=stderr_tqdm_class(),
                )
            except TypeError:
                return hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    cache_dir=cache,
                )

    def _ensure_loaded(self) -> None:
        if self._llama is not None:
            return
        # Sequential only: never keep Whisper in VRAM beside the GGUF.
        from sublocal.transcribe import unload_whisper

        unload_whisper()
        path = self._download_gguf()
        status(f"Loading GGUF ({self.filename})")
        self._llama = _create_llama(path, n_ctx=self.n_ctx)
        status(f"Model ready (device={self.device})")

    def translate(
        self, texts: list[str], src_flores: str, tgt_flores: str
    ) -> list[str]:
        if not texts:
            return []
        self._ensure_loaded()
        llama = self._llama
        assert llama is not None
        src_name = to_english_name(src_flores)
        tgt_name = to_english_name(tgt_flores)
        out: list[str] = [""] * len(texts)
        nonempty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not nonempty:
            return out
        counter = BatchCounter(len(texts))
        empty_count = len(texts) - len(nonempty)
        for n, (orig_i, text) in enumerate(nonempty):
            hint = self.name_hint or maybe_name_instruction(text)
            prompt = gemmax_prompt(src_name, tgt_name, text, name_hint=hint)
            result = llama(
                prompt,
                temperature=0,
                top_k=1,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            raw = ""
            choices = result.get("choices") if isinstance(result, dict) else None
            if choices:
                raw = str(choices[0].get("text") or "")
            out[orig_i] = strip_gemmax_completion(raw, tgt_name)
            step = 1
            if n == 0 and empty_count:
                step += empty_count
            counter.update(step)
        return out


def _quiet_ct2_debug() -> None:
    """Keep Info (Loaded model); hide Debug (Finished batch translation)."""
    import logging

    import ctranslate2

    if os.environ.get("CTRANSLATE2_LOG_LEVEL"):
        return
    ctranslate2.set_log_level(logging.INFO)


def _compute_type(device: str, spec: NllbModelSpec | None = None) -> str:
    if device != "cuda":
        return "int8"
    return (spec or LARGE_SPEC).cuda_compute_type


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda oom" in msg
        or "cublas_status_alloc_failed" in msg
    )


def _create_translator(model_dir: str, device: str, compute_type: str):
    """Build CTranslate2 Translator. CUDA float16 OOM retries once as int8_float16."""
    import ctranslate2

    try:
        return ctranslate2.Translator(
            model_dir, device=device, compute_type=compute_type
        )
    except Exception as exc:
        if device == "cuda" and compute_type == "float16" and _is_cuda_oom(exc):
            status("CUDA out of memory; retrying with int8_float16")
            return ctranslate2.Translator(
                model_dir, device=device, compute_type="int8_float16"
            )
        raise


def _local_entry_error() -> type[Exception]:
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError

        return LocalEntryNotFoundError
    except ImportError:  # huggingface_hub < 1.0
        from huggingface_hub.utils import LocalEntryNotFoundError

        return LocalEntryNotFoundError


class NllbBackend:
    """NLLB-200 3.3B CT2 float16 by default; 600M via ``model='small'``."""

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 32,
        repo_id: str | None = None,
        model: str | None = None,
    ) -> None:
        reject_anaconda_on_windows()
        self.spec = nllb_model_spec(model)
        self.device = resolve_device(device)
        self.batch_size = max(1, batch_size)
        self.repo_id = repo_id or os.environ.get("SUBLOCAL_MODEL_REPO") or self.spec.repo_id
        self.tokenizer_repo = (
            os.environ.get("SUBLOCAL_TOKENIZER_REPO") or self.spec.tokenizer_repo
        )
        self._translator = None
        self._tokenizer = None

    def prepare(self) -> None:
        """Load weights (and refuse Anaconda on Windows) before cue work starts."""
        self._ensure_loaded()

    def _snapshot(self, repo_id: str, cache: str, size_hint: str) -> tuple[str, bool]:
        """Return (local path, was_cached). First miss uses HF tqdm byte bars."""
        from huggingface_hub import snapshot_download

        missing = _local_entry_error()
        try:
            path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache,
                local_files_only=True,
            )
            return path, True
        except missing:
            enable_download_progress()
            status(
                f"Downloading {repo_id} ({size_hint}) into {cache}. "
                "No Hugging Face token required."
            )
            path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache,
                tqdm_class=stderr_tqdm_class(),
            )
            return path, False

    def _load_tokenizer(self, cache: str) -> tuple[object, bool]:
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(
                self.tokenizer_repo,
                cache_dir=cache,
                local_files_only=True,
            )
            return tok, True
        except Exception:
            enable_download_progress()
            status(
                f"Downloading tokenizer {self.tokenizer_repo} into {cache}. "
                "No Hugging Face token required."
            )
            tok = AutoTokenizer.from_pretrained(
                self.tokenizer_repo,
                cache_dir=cache,
            )
            return tok, False

    def _ensure_loaded(self) -> None:
        if self._translator is not None:
            return
        # Again immediately before Translator(); that call AV-crashes on
        # Windows Anaconda instead of raising.
        reject_anaconda_on_windows()
        # Sequential only: never keep Whisper in VRAM beside NLLB.
        from sublocal.transcribe import unload_whisper

        unload_whisper()
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        from transformers.utils import logging as hf_logging

        _quiet_ct2_debug()
        hf_logging.set_verbosity_error()
        cache = str(hf_cache_dir())
        model_dir, model_cached = self._snapshot(
            self.repo_id, cache, size_hint=self.spec.size_hint
        )
        tokenizer, tok_cached = self._load_tokenizer(cache)

        if model_cached and tok_cached:
            status(f"Loading model from cache ({self.repo_id})")
        else:
            status("Loading model")

        self._tokenizer = tokenizer
        self._translator = _create_translator(
            model_dir,
            self.device,
            _compute_type(self.device, self.spec),
        )
        status(f"Model ready (device={self.device})")

    def translate(
        self, texts: list[str], src_flores: str, tgt_flores: str
    ) -> list[str]:
        if not texts:
            return []
        self._ensure_loaded()
        tokenizer = self._tokenizer
        translator = self._translator
        assert tokenizer is not None and translator is not None

        out: list[str] = [""] * len(texts)
        nonempty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not nonempty:
            return out

        batch_size = self.batch_size
        total = len(texts)
        empty_count = total - len(nonempty)
        counter = BatchCounter(total)
        for start in range(0, len(nonempty), batch_size):
            chunk = nonempty[start : start + batch_size]
            sources: list[list[str]] = []
            for _, text in chunk:
                # FLORES src on this cue, then encode. Never reuse another
                # cue's src_lang. Mixed files call translate() once per src.
                _set_src_lang(tokenizer, src_flores)
                token_ids = tokenizer.encode(text, add_special_tokens=True)
                sources.append(tokenizer.convert_ids_to_tokens(token_ids))
            results = translator.translate_batch(
                sources,
                target_prefix=[[tgt_flores]] * len(sources),
                beam_size=2,
                max_input_length=512,
                max_decoding_length=512,
            )
            for (orig_i, _), result in zip(chunk, results, strict=True):
                hyp = list(result.hypotheses[0]) if result.hypotheses else []
                if hyp and hyp[0] == tgt_flores:
                    hyp = hyp[1:]
                hyp = [tok for tok in hyp if tok not in _SKIP_TOKENS]
                decoded = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(hyp),
                    skip_special_tokens=True,
                )
                out[orig_i] = decoded
            n = len(chunk)
            if start == 0 and empty_count:
                n += empty_count
            counter.update(n)
        return out
