"""Translation backends. Default is local NLLB-200 via CTranslate2."""

from __future__ import annotations

import os
from typing import Protocol

from sublocal.cache import hf_cache_dir
from sublocal.progress import (
    BatchCounter,
    enable_download_progress,
    status,
    stderr_tqdm_class,
)
from sublocal.runtime import reject_anaconda_on_windows

# Match the CLI promise: no telemetry. Must be set before huggingface_hub import.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# CTranslate2 int8 conversion of Meta's distilled 600M NLLB-200.
# Source weights: facebook/nllb-200-distilled-600M (CC-BY-NC-4.0).
DEFAULT_MODEL_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
# Tokenizer files from the official NLLB repo (small; no model weights).
DEFAULT_TOKENIZER_REPO = "facebook/nllb-200-distilled-600M"

# Skip special tokens that CTranslate2 may leave on the hypothesis.
_SKIP_TOKENS = {"</s>", "<s>", "<pad>", "<unk>"}


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


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2

        _quiet_ct2_debug()
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def _compute_type(device: str) -> str:
    return "int8_float16" if device == "cuda" else "int8"


def _local_entry_error() -> type[Exception]:
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError

        return LocalEntryNotFoundError
    except ImportError:  # huggingface_hub < 1.0
        from huggingface_hub.utils import LocalEntryNotFoundError

        return LocalEntryNotFoundError


class NllbBackend:
    """NLLB-200 distilled 600M, CTranslate2 int8, downloaded on first use."""

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 32,
        repo_id: str | None = None,
    ) -> None:
        reject_anaconda_on_windows()
        self.device = resolve_device(device)
        self.batch_size = max(1, batch_size)
        self.repo_id = repo_id or os.environ.get(
            "SUBLOCAL_MODEL_REPO", DEFAULT_MODEL_REPO
        )
        self.tokenizer_repo = os.environ.get(
            "SUBLOCAL_TOKENIZER_REPO", DEFAULT_TOKENIZER_REPO
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
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        from transformers.utils import logging as hf_logging
        import ctranslate2

        _quiet_ct2_debug()
        hf_logging.set_verbosity_error()
        cache = str(hf_cache_dir())
        model_dir, model_cached = self._snapshot(
            self.repo_id, cache, size_hint="CTranslate2 int8, ~600 MB"
        )
        tokenizer, tok_cached = self._load_tokenizer(cache)

        if model_cached and tok_cached:
            status(f"Loading model from cache ({self.repo_id})")
        else:
            status("Loading model")

        self._tokenizer = tokenizer
        self._translator = ctranslate2.Translator(
            model_dir,
            device=self.device,
            compute_type=_compute_type(self.device),
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

        tokenizer.src_lang = src_flores
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
