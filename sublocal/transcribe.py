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
                    )
                )
        return cls(out)

    def split_by_punctuation(self, punctuation: list[str] | str) -> Transcript:
        marks = list(punctuation) if not isinstance(punctuation, str) else [punctuation]
        self.segments = [
            Segment(words=chunk)
            for seg in self.segments
            for chunk in _split_words_by_punctuation(seg.words, marks)
        ]
        return self

    def split_by_gap(self, max_gap: float) -> Transcript:
        self.segments = [
            Segment(words=chunk)
            for seg in self.segments
            for chunk in _split_words_by_gap(seg.words, max_gap)
        ]
        return self

    def split_by_duration(self, max_duration: float) -> Transcript:
        """Cap each cue at ``max_duration`` seconds (word-derived)."""
        self.segments = [
            Segment(words=chunk)
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
                merged[-1] = Segment(words=prev.words + nxt.words)
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
                new_segments.append(Segment(words=_join_chunks_as_lines(chunks)))
            else:
                new_segments.extend(Segment(words=chunk) for chunk in chunks)
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
                paired.append(Segment(words=first + list(segs[i + 1].words)))
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
                    out.append(Segment(words=words))
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
        blocks.append(
            Block(kind="cue", cue=Cue(index=str(index), timing=timing, text=text))
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


def _whisper_infer(
    model: Any,
    audio: Any,
    language: str | None,
    progress_cb: Callable[[float, float], None],
    *,
    initial_prompt: str | None = None,
    condition_on_previous_text: bool | None = None,
) -> Any:
    """Run stable-ts/faster-whisper on an in-memory waveform.

    ``audio`` must be a 16 kHz mono float32 array, never a file path.
    Passing a path makes ``WhisperResult.adjust_by_silence`` reload the
    file through ``load_audio`` → ``ffmpeg``, which is not on PATH.
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
        "progress_callback": progress_cb,
    }
    if language:
        kwargs["language"] = language
    if initial_prompt is not None:
        kwargs["initial_prompt"] = initial_prompt
    if condition_on_previous_text is not None:
        kwargs["condition_on_previous_text"] = condition_on_previous_text
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
                "language",
            }
        }
        if language:
            fallback["language"] = language
        fallback["progress_callback"] = progress_cb
        try:
            return transcribe(audio, **fallback)
        except TypeError:
            fallback.pop("progress_callback", None)
            return transcribe(audio, **fallback)


def transcribe_file(
    input_path: str | Path,
    language: str | None = None,
    model_size: str = DEFAULT_MODEL,
    output_path: str | Path | None = None,
    device: str = "auto",
    glossary: Glossary | str | Path | None = None,
) -> Path:
    inp = Path(input_path)
    if not inp.is_file():
        raise FileNotFoundError(f"Input file not found: {inp}")
    model_size = validate_model(model_size)
    out = Path(output_path) if output_path else default_output_path(inp)
    resolved = resolve_device(device)
    gloss = as_glossary(glossary)
    waveform = decode_audio(inp)
    model = load_whisper(model_size, resolved)
    progress = SeekProgress()
    infer_kw: dict[str, Any] = {}
    if gloss is not None:
        infer_kw["initial_prompt"] = gloss.whisper_prompt()
        infer_kw["condition_on_previous_text"] = False
    try:
        raw = _whisper_infer(model, waveform, language, progress, **infer_kw)
        progress.finish()
        transcript = apply_regroup(raw)
        if not any(seg.words for seg in transcript.segments):
            raise ValueError("No speech found in the input file.")
        n = write_transcript_srt(transcript, out, glossary=gloss)
        status(f"Wrote {n} cues")
        return out
    finally:
        unload_whisper(model)


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
