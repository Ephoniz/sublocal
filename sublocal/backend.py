"""Translation backends. Default is local NLLB-200 3.3B via CTranslate2."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from sublocal.cache import hf_cache_dir
from sublocal.device import resolve_device
from sublocal.progress import (
    BatchCounter,
    enable_download_progress,
    status,
    stderr_tqdm_class,
)
from sublocal.runtime import reject_anaconda_on_windows

# Match the CLI promise: no telemetry. Must be set before huggingface_hub import.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Default: CTranslate2 float16 conversion of Meta's NLLB-200 3.3B (CC-BY-NC-4.0).
# 600M distilled is --model small only. Do not ship 600M as the default.
DEFAULT_MODEL_REPO = "entai2965/nllb-200-3.3B-ctranslate2-float16"
DEFAULT_TOKENIZER_REPO = "facebook/nllb-200-3.3B"
SMALL_MODEL_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
SMALL_TOKENIZER_REPO = "facebook/nllb-200-distilled-600M"


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
