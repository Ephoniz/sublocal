"""Translation backends. Default is local NLLB-200 via CTranslate2."""

from __future__ import annotations

import os
import sys
from typing import Protocol

from sublocal.cache import hf_cache_dir

# CTranslate2 int8 conversion of Meta's distilled 600M NLLB-200.
# Source weights: facebook/nllb-200-distilled-600M (CC-BY-NC-4.0).
DEFAULT_MODEL_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"

# Skip special tokens that CTranslate2 may leave on the hypothesis.
_SKIP_TOKENS = {"</s>", "<s>", "<pad>", "<unk>"}


class TranslatorBackend(Protocol):
    def translate(
        self, texts: list[str], src_flores: str, tgt_flores: str
    ) -> list[str]: ...


class EchoBackend:
    """Returns cue text unchanged. Used by tests; no model download."""

    def translate(
        self, texts: list[str], src_flores: str, tgt_flores: str
    ) -> list[str]:
        return list(texts)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def _compute_type(device: str) -> str:
    return "int8_float16" if device == "cuda" else "int8"


class NllbBackend:
    """NLLB-200 distilled 600M, CTranslate2 int8, downloaded on first use."""

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 32,
        repo_id: str | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.batch_size = max(1, batch_size)
        self.repo_id = repo_id or os.environ.get(
            "SUBLOCAL_MODEL_REPO", DEFAULT_MODEL_REPO
        )
        self._translator = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._translator is not None:
            return
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        import ctranslate2

        cache = str(hf_cache_dir())
        try:
            model_dir = snapshot_download(
                repo_id=self.repo_id,
                cache_dir=cache,
                local_files_only=True,
            )
        except Exception:
            print(
                f"First run: downloading {self.repo_id} "
                f"(CTranslate2 int8, ~600 MB) into {cache}. "
                "No Hugging Face token required.",
                file=sys.stderr,
            )
            model_dir = snapshot_download(
                repo_id=self.repo_id,
                cache_dir=cache,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._translator = ctranslate2.Translator(
            model_dir,
            device=self.device,
            compute_type=_compute_type(self.device),
        )

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
        return out
