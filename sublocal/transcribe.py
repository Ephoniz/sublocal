"""Transcribe audio/video to sentence-sized SRT. Timing is word-derived."""

from __future__ import annotations

import gc
import inspect
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from sublocal.cache import hf_cache_dir
from sublocal.cues_jsonl import cues_jsonl_path, write_cues_jsonl
from sublocal.device import resolve_device
from sublocal.formats import save
from sublocal.formats.base import Block, Cue, Document
from sublocal.glossary import Glossary, as_glossary
from sublocal.progress import enable_download_progress, format_eta, status
from sublocal.runtime import reject_anaconda_on_windows

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

DEFAULT_MODEL = "large-v3"
DEFAULT_REPO = "Systran/faster-whisper-large-v3"
MAX_CHARS = 32
GAP_SPLIT_S = 0.5
GAP_MERGE_S = 0.15
MAX_CUE_DURATION_S = 4.0
VAD_MIN_SILENCE_MS = 500
VAD_MIN_SILENCE_S = 0.5
WHISPER_SAMPLE_RATE = 16000
PUNCTUATION = ["。", "?", "？"]
_BIGGER_THAN_LARGE_V3 = re.compile(
    r"large-v(?:[4-9]\d*|[1-9]\d+)", re.IGNORECASE
)

# faster-whisper size name → HF repo (large == large-v3).
_SIZE_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": DEFAULT_REPO,
    "large": DEFAULT_REPO,
}


@dataclass
class Word:
    """One timed token. ``word`` may include a trailing newline from wrapping."""

    word: str
    start: float
    end: float


@dataclass
class Segment:
    """Words that form one cue. Raw Whisper window times are optional."""

    words: list[Word] = field(default_factory=list)
    start: float | None = None
    end: float | None = None
    lang: str | None = None

    def text(self) -> str:
        return "".join(w.word for w in self.words)

    def cue_text(self) -> str:
        lines = [ln.strip() for ln in self.text().split("\n")]
        return "\n".join(ln for ln in lines if ln)

    def char_count(self) -> int:
        return len(self.text().replace("\n", ""))


class Transcript:
    """Stand-in for ``stable_whisper.result.WhisperResult`` regrouping."""

    def __init__(self, segments: list[Segment] | None = None) -> None:
        self.segments: list[Segment] = list(segments or [])

    @classmethod
    def from_any(cls, result: Any) -> Transcript:
        if isinstance(result, Transcript):
            return result
        raw_segments = getattr(result, "segments", None)
        if raw_segments is None:
            raise TypeError(f"Cannot build Transcript from {type(result)!r}")
        out: list[Segment] = []
        for seg in raw_segments:
            if isinstance(seg, Segment):
                out.append(
                    Segment(
                        words=[
                            Word(word=w.word, start=w.start, end=w.end)
                            for w in seg.words
                        ],
                        start=seg.start,
                        end=seg.end,
                        lang=seg.lang,
                    )
                )
                continue
            if isinstance(seg, list):
                words = [_as_word(w) for w in seg]
                if words:
                    out.append(Segment(words=words))
                continue
            words_raw = getattr(seg, "words", None)
            if words_raw:
                words = [_as_word(w) for w in words_raw]
            else:
                text = getattr(seg, "text", "") or ""
                words = [
                    Word(
                        word=text,
                        start=float(getattr(seg, "start", 0.0) or 0.0),
                        end=float(getattr(seg, "end", 0.0) or 0.0),
                    )
                ]
            if words:
                out.append(
                    Segment(
                        words=words,
                        start=_opt_float(getattr(seg, "start", None)),
                        end=_opt_float(getattr(seg, "end", None)),
                        lang=_segment_lang(seg),
                    )
                )
        return cls(out)

    def split_by_punctuation(self, punctuation: list[str] | str) -> Transcript:
        marks = list(punctuation) if not isinstance(punctuation, str) else [punctuation]
        self.segments = [
            Segment(words=chunk, lang=seg.lang)
            for seg in self.segments
            for chunk in _split_words_by_punctuation(seg.words, marks)
        ]
        return self

    def split_by_gap(self, max_gap: float) -> Transcript:
        self.segments = [
            Segment(words=chunk, lang=seg.lang)
            for seg in self.segments
            for chunk in _split_words_by_gap(seg.words, max_gap)
        ]
        return self

    def split_by_duration(self, max_duration: float) -> Transcript:
        """Cap each cue at ``max_duration`` seconds (word-derived)."""
        self.segments = [
            Segment(words=chunk, lang=seg.lang)
            for seg in self.segments
            for chunk in _split_words_by_duration(seg.words, max_duration)
        ]
        return self

    def merge_by_gap(self, min_gap: float, max_chars: int | None = None) -> Transcript:
        if not self.segments:
            return self
        merged: list[Segment] = [self.segments[0]]
        for nxt in self.segments[1:]:
            prev = merged[-1]
            if not prev.words or not nxt.words:
                merged.append(nxt)
                continue
            gap = nxt.words[0].start - prev.words[-1].end
            combined = prev.char_count() + nxt.char_count()
            if gap <= min_gap and (max_chars is None or combined <= max_chars):
                merged[-1] = Segment(
                    words=prev.words + nxt.words, lang=prev.lang or nxt.lang
                )
            else:
                merged.append(nxt)
        self.segments = merged
        return self

    def split_by_length(
        self, max_chars: int = MAX_CHARS, newline: bool = False, **_: Any
    ) -> Transcript:
        new_segments: list[Segment] = []
        for seg in self.segments:
            chunks = _chunk_words_by_chars(seg.words, max_chars)
            if not chunks:
                continue
            if newline:
                new_segments.append(
                    Segment(words=_join_chunks_as_lines(chunks), lang=seg.lang)
                )
            else:
                new_segments.extend(
                    Segment(words=chunk, lang=seg.lang) for chunk in chunks
                )
        self.segments = new_segments
        return self

    def pair_lines(self) -> Transcript:
        """Join adjacent ≤32-char segments into at most two ``\\n`` lines."""
        paired: list[Segment] = []
        i = 0
        segs = self.segments
        while i < len(segs):
            if i + 1 < len(segs) and segs[i].words and segs[i + 1].words:
                first = [
                    Word(w.word, w.start, w.end) for w in segs[i].words
                ]
                first[-1] = Word(
                    first[-1].word.rstrip("\n") + "\n",
                    first[-1].start,
                    first[-1].end,
                )
                paired.append(
                    Segment(
                        words=first + list(segs[i + 1].words),
                        lang=segs[i].lang or segs[i + 1].lang,
                    )
                )
                i += 2
            else:
                paired.append(segs[i])
                i += 1
        self.segments = paired
        return self

    def enforce_two_lines(self) -> Transcript:
        """If a segment has more than two ``\\n`` lines, extra lines become cues."""
        out: list[Segment] = []
        for seg in self.segments:
            lines = _lines_from_words(seg.words)
            if len(lines) <= 2:
                if seg.words:
                    out.append(seg)
                continue
            for start in range(0, len(lines), 2):
                chunk = lines[start : start + 2]
                words: list[Word] = []
                for li, line in enumerate(chunk):
                    for w in line:
                        words.append(Word(w.word, w.start, w.end))
                    if li < len(chunk) - 1 and words:
                        last = words[-1]
                        words[-1] = Word(
                            last.word.rstrip("\n") + "\n", last.start, last.end
                        )
                if words:
                    out.append(Segment(words=words, lang=seg.lang))
        self.segments = out
        return self


class SeekProgress:
    """PowerShell-safe ``12/45s (26%) ~30s left`` lines. No ``\\r``-only signal."""

    def __init__(self, min_interval: float = 2.0, min_pct: float = 5.0) -> None:
        self.min_interval = min_interval
        self.min_pct = min_pct
        self._start = time.monotonic()
        self._last_t = 0.0
        self._last_pct = -min_pct
        self._final = False
        self.total = 0.0

    def __call__(self, seek_seconds: float, total_seconds: float) -> None:
        self.total = max(self.total, float(total_seconds or 0.0))
        if self.total <= 0:
            return
        seek = max(0.0, float(seek_seconds))
        pct = seek * 100.0 / self.total
        if seek >= self.total:
            self._emit(self.total, 100, eta="")
            self._final = True
            return
        now = time.monotonic()
        if (
            now - self._last_t < self.min_interval
            and pct - self._last_pct < self.min_pct
        ):
            return
        self._emit(seek, int(pct), eta=self._eta(seek))

    def finish(self) -> None:
        if not self._final and self.total > 0:
            self._emit(self.total, 100, eta="")
            self._final = True

    def _eta(self, seek: float) -> str:
        if seek <= 0:
            return ""
        elapsed = time.monotonic() - self._start
        if elapsed < 0.05:
            return ""
        remaining = (self.total - seek) * (elapsed / seek)
        return format_eta(remaining)

    def _emit(self, seek: float, pct: int, eta: str) -> None:
        done = int(round(min(seek, self.total)))
        total = int(round(self.total))
        line = f"{done}/{total}s ({pct}%)"
        if eta:
            line = f"{line} ~{eta} left"
        status(line)
        self._last_t = time.monotonic()
        self._last_pct = pct


def default_output_path(inp: Path) -> Path:
    return inp.with_suffix(".srt")


def validate_model(model_size: str) -> str:
    """Reject anything larger than large-v3. Known smaller sizes are allowed."""
    name = model_size.strip()
    if not name:
        raise ValueError("Model size is empty.")
    if _BIGGER_THAN_LARGE_V3.search(name):
        raise ValueError(
            f"Model {name!r} is larger than large-v3, which is the maximum."
        )
    return name


def whisper_repo_id(model_size: str) -> str:
    if "/" in model_size:
        return model_size
    return _SIZE_REPOS.get(model_size, f"Systran/faster-whisper-{model_size}")


def snapshot_cached(repo_id: str, cache: Path | None = None) -> bool:
    root = Path(cache) if cache is not None else hf_cache_dir()
    dirname = "models--" + repo_id.replace("/", "--")
    snaps = root / dirname / "snapshots"
    if not snaps.is_dir():
        return False
    for snap in snaps.iterdir():
        if snap.is_dir() and any(snap.iterdir()):
            return True
    return False


def unload_whisper(model: object | None = None) -> None:
    """Drop Whisper so a later translate in this process can use the GPU.

    Sequential only: never keep Whisper loaded beside NLLB.
    """
    del model
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    cuda = getattr(torch, "cuda", None)
    empty = getattr(cuda, "empty_cache", None) if cuda is not None else None
    if callable(empty):
        try:
            empty()
        except Exception:
            pass


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def apply_regroup(result: Any) -> Transcript:
    """Sentence-sized cues: punctuation, 0.5s gaps, ~4s caps, ~32-char wrap."""
    used_newline = False
    obj: Any = result
    if not _has_chain_methods(obj):
        obj = Transcript.from_any(obj)
    used_newline = _apply_chain(obj)
    transcript = obj if isinstance(obj, Transcript) else Transcript.from_any(obj)
    transcript.split_by_duration(MAX_CUE_DURATION_S)
    if not used_newline:
        transcript.pair_lines()
    transcript.enforce_two_lines()
    return transcript


def transcript_to_document(transcript: Transcript) -> Document:
    blocks: list[Block] = []
    index = 0
    for seg in transcript.segments:
        if not seg.words:
            continue
        text = seg.cue_text()
        if not text:
            continue
        index += 1
        start = seg.words[0].start
        end = seg.words[-1].end
        timing = f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}"
        extra: dict[str, Any] = {}
        if seg.lang:
            extra["lang"] = seg.lang
        blocks.append(
            Block(
                kind="cue",
                cue=Cue(index=str(index), timing=timing, text=text, extra=extra),
            )
        )
    return Document(format="srt", blocks=blocks)


def canonicalize_document(doc: Document, glossary: Glossary) -> None:
    """Protect JP names then restore the Japanese keys (not Latin)."""
    for cue in doc.cues:
        guarded, pairs = glossary.protect(cue.text)
        cue.text = glossary.restore(guarded, pairs, target="key")


def write_transcript_srt(
    transcript: Transcript,
    path: Path,
    glossary: Glossary | None = None,
) -> int:
    doc = transcript_to_document(transcript)
    if glossary is not None:
        canonicalize_document(doc, glossary)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(doc, path)
    return len(doc.cues)


def decode_audio(
    path: str | Path, sampling_rate: int = WHISPER_SAMPLE_RATE
) -> Any:
    """Decode audio/video to 16 kHz mono float32 with PyAV. No ffmpeg CLI."""
    import gc
    import itertools

    import av
    import numpy as np

    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=sampling_rate,
    )
    chunks: list[Any] = []
    try:
        with av.open(str(path), mode="r", metadata_errors="ignore") as container:
            audio_streams = getattr(container.streams, "audio", None)
            if not audio_streams:
                raise ValueError(f"No audio stream in {path}")
            frames = _iter_audio_frames(container)
            for frame in itertools.chain(frames, [None]):
                for resampled in resampler.resample(frame):
                    arr = resampled.to_ndarray()
                    chunks.append(np.ascontiguousarray(arr).reshape(-1))
    finally:
        del resampler
        gc.collect()
    if not chunks:
        raise ValueError(f"No audio decoded from {path}")
    audio = np.concatenate(chunks)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = np.asarray(audio, dtype=np.float32)
    return np.ascontiguousarray(audio.reshape(-1), dtype=np.float32)


def load_whisper(model_size: str, device: str) -> Any:
    reject_anaconda_on_windows()
    import stable_whisper

    cache = hf_cache_dir()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_CACHE", str(cache))

    repo = whisper_repo_id(model_size)
    if repo == DEFAULT_REPO and not snapshot_cached(repo, cache):
        status(
            f"Downloading {DEFAULT_REPO} is large (~3GB) into {cache}. "
            "No Hugging Face token required."
        )
        enable_download_progress()

    label = model_size.rsplit("/", 1)[-1]
    status(f"Loading Whisper {label}")
    compute_type = "float16" if device == "cuda" else "int8"
    model = stable_whisper.load_faster_whisper(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=str(cache),
    )
    status(f"Model ready (device={device})")
    return model


def whisper_transcribe_kwargs(
    language: str | None,
    *,
    batched: bool = False,
    initial_prompt: str | None = None,
    condition_on_previous_text: bool | None = None,
) -> dict[str, Any]:
    """Default faster-whisper kwargs. Never ``language=None`` without multilingual.

    SYSTRAN/faster-whisper#869: ``language=None`` without ``multilingual=True``
    detects language once on the first 30s and locks it for the file.
    """
    kwargs: dict[str, Any] = {
        "word_timestamps": True,
        "verbose": None,
        "regroup": False,
        "vad_filter": True,
        "vad_parameters": dict(min_silence_duration_ms=VAD_MIN_SILENCE_MS),
        # Silence-adjust uses this same ndarray (no ffmpeg reload).
        "vad": True,
        "min_silence_dur": VAD_MIN_SILENCE_S,
        "task": "transcribe",
        "condition_on_previous_text": (
            False if condition_on_previous_text is None else condition_on_previous_text
        ),
        # Sequential default is False; batched default is True and drops
        # timestamp tokens (whole-chunk SRT spans). Always keep timestamps.
        "without_timestamps": False,
    }
    if language:
        kwargs["language"] = language
        kwargs["multilingual"] = False
    else:
        kwargs["language"] = None
        kwargs["multilingual"] = True
    if initial_prompt is not None:
        kwargs["initial_prompt"] = initial_prompt
    return kwargs


def stamp_langs_sequential(segments: Any, tokenizer: Any) -> list[Any]:
    """Stamp ``tokenizer.language_code`` onto each sequential Whisper segment.

    faster-whisper updates ``tokenizer.language_code`` when multilingual
    detection runs for a new 30s window. Segment has no language field.
    """
    out: list[Any] = []
    for seg in segments:
        lang = getattr(tokenizer, "language_code", None)
        _set_seg_lang(seg, lang)
        out.append(seg)
    return out


def stamp_langs_batched(segments: Any, language_codes: list[str | None]) -> list[Any]:
    """Stamp langs from batched ``detect_language`` prompt-token patches."""
    out: list[Any] = []
    for i, seg in enumerate(segments):
        lang = language_codes[i] if i < len(language_codes) else None
        _set_seg_lang(seg, lang)
        out.append(seg)
    return out


def stamp_langs_from_vad_chunks(
    segments: Any,
    chunks_metadata: list[dict[str, Any]],
    language_tokens: list[Any],
) -> list[Any]:
    """One lang per VAD chunk: zip ``language_tokens[i]`` with ``chunks_metadata[i]``.

    Batched detect_language patches ``prompts[i][language_token_index]`` and
    does not write lang onto Segment. Map each yielded segment into the
    chunk whose ``offset``/``duration`` contains its start.
    """
    spans: list[tuple[float, float, str | None]] = []
    for meta, token in zip(chunks_metadata, language_tokens):
        offset = float(meta.get("offset", 0.0) or 0.0)
        duration = float(meta.get("duration", 0.0) or 0.0)
        spans.append((offset, offset + duration, lang_from_prompt_token(token)))
    out: list[Any] = []
    for seg in segments:
        start = _seg_start(seg)
        lang = None
        for lo, hi, chunk_lang in spans:
            if start is None or (lo - 1e-3 <= start <= hi + 1e-3):
                lang = chunk_lang
                break
        _set_seg_lang(seg, lang)
        out.append(seg)
    return out


def lang_from_prompt_token(token: Any) -> str | None:
    """``<|ja|>`` / token string from batched detect_language → ``ja``."""
    if token is None:
        return None
    if isinstance(token, str):
        raw = token.strip()
        if raw.startswith("<|") and raw.endswith("|>"):
            return raw[2:-2] or None
        return raw or None
    return None


def _whisper_infer(
    model: Any,
    audio: Any,
    language: str | None,
    progress_cb: Callable[[float, float], None],
    *,
    initial_prompt: str | None = None,
    condition_on_previous_text: bool | None = None,
    batched: bool = False,
) -> Any:
    """Run stable-ts/faster-whisper on an in-memory waveform.

    ``audio`` must be a 16 kHz mono float32 array, never a file path.
    Passing a path makes ``WhisperResult.adjust_by_silence`` reload the
    file through ``load_audio`` → ``ffmpeg``, which is not on PATH.
    """
    kwargs = whisper_transcribe_kwargs(
        language,
        batched=batched,
        initial_prompt=initial_prompt,
        condition_on_previous_text=condition_on_previous_text,
    )
    kwargs["progress_callback"] = progress_cb
    hook = _LangRecorder()
    unhook = _try_install_lang_hooks(model, hook, batched=batched)
    try:
        result = _call_transcribe(model, audio, kwargs, batched=batched, hook=hook)
        result = _apply_stamped_langs(result, hook, language)
        return result
    finally:
        unhook()


def _call_transcribe(
    model: Any,
    audio: Any,
    kwargs: dict[str, Any],
    *,
    batched: bool,
    hook: _LangRecorder | None = None,
) -> Any:
    if batched:
        fw = _find_fw_model(model)
        if fw is not None:
            from faster_whisper import BatchedInferencePipeline

            pipe = BatchedInferencePipeline(model=fw)
            orig_gen = getattr(pipe, "generate_segment_batched", None)
            if callable(orig_gen) and hook is not None:

                def _gen(features, tokenizer, chunks_metadata, options, *a, **k):
                    hook.chunks_metadata.extend(list(chunks_metadata or []))
                    return orig_gen(
                        features, tokenizer, chunks_metadata, options, *a, **k
                    )

                pipe.generate_segment_batched = _gen  # type: ignore[method-assign]
            batch_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                not in {
                    "progress_callback",
                    "verbose",
                    "regroup",
                    "vad",
                    "min_silence_dur",
                }
            }
            segs, _info = pipe.transcribe(audio, **batch_kwargs)
            return list(segs)
    transcribe = model.transcribe
    try:
        return transcribe(audio, **kwargs)
    except TypeError:
        fallback = {
            k: v
            for k, v in kwargs.items()
            if k
            not in {
                "vad_filter",
                "vad_parameters",
                "vad",
                "min_silence_dur",
                "progress_callback",
            }
        }
        # Never drop multilingual while language is None (faster-whisper#869).
        if fallback.get("language") is None:
            fallback["multilingual"] = True
        fallback["progress_callback"] = kwargs.get("progress_callback")
        try:
            return transcribe(audio, **fallback)
        except TypeError:
            fallback.pop("progress_callback", None)
            if fallback.get("language") is None:
                fallback["multilingual"] = True
            return transcribe(audio, **fallback)


def _apply_stamped_langs(result: Any, hook: "_LangRecorder", language: str | None) -> Any:
    segs = _result_segments(result)
    if not segs:
        return result
    if hook.per_segment:
        for i, seg in enumerate(segs):
            lang = hook.per_segment[i] if i < len(hook.per_segment) else hook.current
            if _segment_lang(seg) is None:
                _set_seg_lang(seg, lang)
    elif hook.batch_langs and hook.chunks_metadata:
        stamp_langs_from_vad_chunks(segs, hook.chunks_metadata, hook.batch_langs)
    elif hook.batch_langs:
        stamp_langs_batched(segs, hook.batch_langs)
    elif language:
        for seg in segs:
            if _segment_lang(seg) is None:
                _set_seg_lang(seg, language)
    return result


class _LangRecorder:
    def __init__(self) -> None:
        self.current: str | None = None
        self.per_segment: list[str | None] = []
        self.batch_langs: list[str | None] = []
        self.chunks_metadata: list[dict[str, Any]] = []


def _try_install_lang_hooks(
    model: Any, hook: _LangRecorder, *, batched: bool
) -> Callable[[], None]:
    restorers: list[Callable[[], None]] = []

    try:
        from faster_whisper.tokenizer import Tokenizer
    except ImportError:
        Tokenizer = None  # type: ignore[misc, assignment]
    if Tokenizer is not None:
        orig_sa = Tokenizer.__setattr__

        def watched(self: Any, name: str, value: Any) -> None:
            orig_sa(self, name, value)
            if name == "language_code":
                hook.current = value

        Tokenizer.__setattr__ = watched  # type: ignore[method-assign]
        restorers.append(lambda: setattr(Tokenizer, "__setattr__", orig_sa))

    fw = _find_fw_model(model)
    if fw is not None and hasattr(fw, "transcribe"):
        orig = fw.transcribe

        def wrapped(*args: Any, **kw: Any) -> Any:
            out = orig(*args, **kw)
            if isinstance(out, tuple) and len(out) == 2:
                segs, info = out

                def gen() -> Any:
                    for seg in segs:
                        lang = hook.current
                        hook.per_segment.append(lang)
                        _set_seg_lang(seg, lang)
                        yield seg

                return gen(), info
            return out

        fw.transcribe = wrapped
        restorers.append(lambda: setattr(fw, "transcribe", orig))

    if batched and fw is not None:
        ct2 = getattr(fw, "model", None)
        detect = getattr(ct2, "detect_language", None) if ct2 is not None else None
        if callable(detect):

            def wrapped_detect(*args: Any, **kw: Any) -> Any:
                results = detect(*args, **kw)
                try:
                    for item in results:
                        tok = item[0][0]
                        hook.batch_langs.append(lang_from_prompt_token(tok))
                except Exception:
                    pass
                return results

            ct2.detect_language = wrapped_detect
            restorers.append(lambda: setattr(ct2, "detect_language", detect))

    def unhook() -> None:
        for restore in reversed(restorers):
            try:
                restore()
            except Exception:
                pass

    return unhook


def _find_fw_model(model: Any) -> Any | None:
    if model is None:
        return None
    names = {"WhisperModel", "FasterWhisperModel"}
    if type(model).__name__ in names:
        return model
    inner = getattr(model, "model", None)
    if inner is not None and type(inner).__name__ in names:
        return inner
    deeper = getattr(inner, "model", None) if inner is not None else None
    if deeper is not None and type(deeper).__name__ in names:
        return deeper
    return None


def _seg_start(seg: Any) -> float | None:
    start = getattr(seg, "start", None)
    if start is not None:
        return float(start)
    words = getattr(seg, "words", None) or []
    if words:
        return float(getattr(words[0], "start", 0.0) or 0.0)
    return None


def _result_segments(result: Any) -> list[Any]:
    if result is None:
        return []
    if isinstance(result, list):
        return result
    segs = getattr(result, "segments", None)
    if segs is None:
        return []
    if isinstance(segs, list):
        return segs
    try:
        return list(segs)
    except TypeError:
        return []


def _segment_lang(seg: Any) -> str | None:
    if isinstance(seg, Segment):
        return seg.lang
    for attr in ("lang", "language"):
        value = getattr(seg, attr, None)
        if value:
            return str(value)
    extra = getattr(seg, "extra", None)
    if isinstance(extra, dict) and extra.get("lang"):
        return str(extra["lang"])
    return None


def _set_seg_lang(seg: Any, lang: str | None) -> None:
    if lang is None:
        return
    if isinstance(seg, Segment):
        seg.lang = lang
        return
    try:
        seg.lang = lang
    except Exception:
        pass
    extra = getattr(seg, "extra", None)
    if isinstance(extra, dict):
        extra["lang"] = lang


def transcribe_document(
    input_path: str | Path,
    language: str | None = None,
    model_size: str = DEFAULT_MODEL,
    device: str = "auto",
    glossary: Glossary | str | Path | None = None,
    batched: bool = False,
) -> Document:
    """Decode → Whisper → regroup → Document with per-cue ``extra['lang']``.

    Unloads Whisper before returning so NLLB can take the GPU.
    """
    inp = Path(input_path)
    if not inp.is_file():
        raise FileNotFoundError(f"Input file not found: {inp}")
    model_size = validate_model(model_size)
    resolved = resolve_device(device)
    gloss = as_glossary(glossary)
    waveform = decode_audio(inp)
    model = load_whisper(model_size, resolved)
    progress = SeekProgress()
    infer_kw: dict[str, Any] = {
        "condition_on_previous_text": False,
        "batched": batched,
    }
    if gloss is not None:
        infer_kw["initial_prompt"] = gloss.whisper_prompt()
    try:
        raw = _whisper_infer(model, waveform, language, progress, **infer_kw)
        progress.finish()
        transcript = apply_regroup(raw)
        if language:
            for seg in transcript.segments:
                if not seg.lang:
                    seg.lang = language
        if not any(seg.words for seg in transcript.segments):
            raise ValueError("No speech found in the input file.")
        doc = transcript_to_document(transcript)
        if gloss is not None:
            canonicalize_document(doc, gloss)
        return doc
    finally:
        unload_whisper(model)


def transcribe_file(
    input_path: str | Path,
    language: str | None = None,
    model_size: str = DEFAULT_MODEL,
    output_path: str | Path | None = None,
    device: str = "auto",
    glossary: Glossary | str | Path | None = None,
    batched: bool = False,
) -> Path:
    inp = Path(input_path)
    if not inp.is_file():
        raise FileNotFoundError(f"Input file not found: {inp}")
    out = Path(output_path) if output_path else default_output_path(inp)
    doc = transcribe_document(
        inp,
        language=language,
        model_size=model_size,
        device=device,
        glossary=glossary,
        batched=batched,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save(doc, out)
    sidecar = cues_jsonl_path(out)
    write_cues_jsonl(doc, sidecar)
    status(f"Wrote {len(doc.cues)} cues")
    return out


def _iter_audio_frames(container: Any) -> Any:
    import av

    frames = container.decode(audio=0)
    while True:
        try:
            yield next(frames)
        except StopIteration:
            return
        except av.error.InvalidDataError:
            continue


def _as_word(w: Any) -> Word:
    if isinstance(w, Word):
        return Word(word=w.word, start=w.start, end=w.end)
    if isinstance(w, dict):
        return Word(
            word=str(w.get("word", w.get("text", ""))),
            start=float(w.get("start", 0.0) or 0.0),
            end=float(w.get("end", 0.0) or 0.0),
        )
    return Word(
        word=str(getattr(w, "word", getattr(w, "text", ""))),
        start=float(getattr(w, "start", 0.0) or 0.0),
        end=float(getattr(w, "end", 0.0) or 0.0),
    )


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _has_chain_methods(obj: Any) -> bool:
    return all(
        callable(getattr(obj, name, None))
        for name in (
            "split_by_punctuation",
            "split_by_gap",
            "merge_by_gap",
            "split_by_length",
        )
    )


def _call(obj: Any, name: str, *args: Any, **kwargs: Any) -> None:
    fn = getattr(obj, name)
    try:
        fn(*args, **kwargs)
        return
    except TypeError:
        pass
    try:
        sig = inspect.signature(fn)
        accepted = {
            k: v
            for k, v in kwargs.items()
            if k in sig.parameters
            or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        }
    except (TypeError, ValueError):
        accepted = {}
    fn(*args, **accepted)


def _apply_chain(obj: Any) -> bool:
    _call(obj, "split_by_punctuation", PUNCTUATION)
    _call(obj, "split_by_gap", GAP_SPLIT_S)
    _call(obj, "merge_by_gap", GAP_MERGE_S, max_chars=MAX_CHARS)
    sl = getattr(obj, "split_by_length")
    used_newline = False
    try:
        sig = inspect.signature(sl)
        if "newline" in sig.parameters:
            sl(max_chars=MAX_CHARS, newline=True)
            used_newline = True
        else:
            sl(max_chars=MAX_CHARS)
    except (TypeError, ValueError):
        try:
            sl(max_chars=MAX_CHARS, newline=True)
            used_newline = True
        except TypeError:
            sl(max_chars=MAX_CHARS)
    return used_newline


def _ends_with_punct(text: str, marks: list[str]) -> bool:
    stripped = text.rstrip("\n").rstrip()
    return any(stripped.endswith(mark) for mark in marks)


def _split_words_by_punctuation(
    words: list[Word], marks: list[str]
) -> list[list[Word]]:
    if not words:
        return []
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for i, w in enumerate(words):
        current.append(w)
        is_last = i == len(words) - 1
        if current and _ends_with_punct(w.word, marks) and not is_last:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def _split_words_by_gap(words: list[Word], max_gap: float) -> list[list[Word]]:
    if not words:
        return []
    chunks: list[list[Word]] = []
    current = [words[0]]
    for prev, nxt in zip(words, words[1:]):
        gap = nxt.start - prev.end
        if gap >= max_gap:
            chunks.append(current)
            current = [nxt]
        else:
            current.append(nxt)
    chunks.append(current)
    return chunks


def _split_words_by_duration(
    words: list[Word], max_duration: float
) -> list[list[Word]]:
    if not words:
        return []
    chunks: list[list[Word]] = []
    current = [words[0]]
    origin = words[0].start
    for w in words[1:]:
        if w.end - origin > max_duration:
            chunks.append(current)
            current = [w]
            origin = w.start
        else:
            current.append(w)
    chunks.append(current)
    return chunks


def _word_display_len(word: str) -> int:
    return len(word.replace("\n", ""))


def _chunk_words_by_chars(words: list[Word], max_chars: int) -> list[list[Word]]:
    if not words:
        return []
    chunks: list[list[Word]] = []
    current: list[Word] = []
    width = 0
    for w in words:
        n = _word_display_len(w.word)
        if current and width + n > max_chars:
            chunks.append(current)
            current = [w]
            width = n
        else:
            current.append(w)
            width += n
    if current:
        chunks.append(current)
    return chunks


def _join_chunks_as_lines(chunks: list[list[Word]]) -> list[Word]:
    merged: list[Word] = []
    for i, chunk in enumerate(chunks):
        if i > 0 and merged:
            last = merged[-1]
            merged[-1] = Word(last.word.rstrip("\n") + "\n", last.start, last.end)
        merged.extend(Word(w.word, w.start, w.end) for w in chunk)
    return merged


def _lines_from_words(words: list[Word]) -> list[list[Word]]:
    lines: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        current.append(w)
        if "\n" in w.word:
            lines.append(current)
            current = []
    if current:
        lines.append(current)
    return lines
