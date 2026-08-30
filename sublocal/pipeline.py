"""Parse → detect → translate cue text → write. Timestamps are never sent to the model."""

from __future__ import annotations

import time
from pathlib import Path

from sublocal.backend import (
    EchoBackend,
    GemmaXBackend,
    NllbBackend,
    TranslatorBackend,
    leftover_arrow_count,
    strip_caption_arrows,
)
from sublocal.cues_jsonl import cues_jsonl_path, langs_from_sidecar, write_cues_jsonl
from sublocal.detect import DetectionError, detect_iso639
from sublocal.extract import extract_file_glossary, merge_mappings, pykakasi_version
from sublocal.formats import load, save
from sublocal.formats.base import Cue, Document
from sublocal.glossary import Glossary, as_glossary, is_majority_jp
from sublocal.languages import display_code, to_flores
from sublocal.lid import (
    detect_cue_lang,
    same_language,
    script_heuristic,
)
from sublocal.progress import status
from sublocal.transcribe import unload_whisper


def default_output_path(inp: Path, to_code: str) -> Path:
    tag = display_code(to_code)
    return inp.with_name(f"{inp.stem}.{tag}{inp.suffix}")


def apply_translations(doc: Document, translated: list[str]) -> None:
    cues = doc.cues
    if len(translated) != len(cues):
        raise RuntimeError(
            f"Translator returned {len(translated)} texts for {len(cues)} cues."
        )
    for cue, text in zip(cues, translated, strict=True):
        cue.text = text


def resolve_cue_langs(
    cues: list[Cue],
    from_code: str | None = None,
    cue_langs: list[str | None] | None = None,
) -> list[str]:
    """Per-cue ISO: sidecar / extra, then ``--from``, then script / lingua LID."""
    out: list[str] = []
    for i, cue in enumerate(cues):
        lang: str | None = None
        if cue_langs is not None and i < len(cue_langs) and cue_langs[i]:
            lang = cue_langs[i]
        elif cue.extra.get("lang"):
            lang = str(cue.extra["lang"])
        elif from_code:
            lang = from_code
        else:
            lang = detect_cue_lang(cue.text)
        out.append(lang or "")
    missing = [i for i, lang in enumerate(out) if not lang]
    if missing:
        try:
            blob = detect_iso639([cues[i].text for i in missing])
        except DetectionError:
            try:
                blob = detect_iso639([c.text for c in cues])
            except DetectionError:
                blob = "en"
        for i in missing:
            out[i] = blob
    # All-Latin file without ASR sidecar: script heuristic says "en" for
    # Spanish/French too. Use concatenated langdetect (existing v0.3 path).
    used_sidecar = cue_langs is not None and any(cue_langs)
    if not from_code and not used_sidecar:
        scripts = [script_heuristic(c.text) for c in cues]
        if scripts and all(s in {None, "en"} for s in scripts):
            try:
                blob = detect_iso639([c.text for c in cues])
            except DetectionError:
                blob = None
            if blob:
                for i, lang in enumerate(out):
                    if lang == "en":
                        refined = detect_cue_lang(cues[i].text)
                        out[i] = refined if refined and refined != "en" else blob
    return out


def translate_document(
    doc: Document,
    to_code: str,
    from_code: str | None = None,
    backend: TranslatorBackend | None = None,
    glossary: Glossary | str | Path | None = None,
    cue_langs: list[str | None] | None = None,
) -> tuple[Document, str, str]:
    """Translate cue text in-place. Returns (doc, src_flores, tgt_flores).

    Each cue is encoded with that cue's source language. Cues already in
    ``to_code`` are copied through. One official completion per cue after
    peel + caption-arrow strip + CJK Person Hepburn. No retry happy path.
    """
    cues = doc.cues
    if not cues:
        raise ValueError("No subtitle cues found in the input file.")

    langs = resolve_cue_langs(cues, from_code=from_code, cue_langs=cue_langs)
    tgt = to_flores(to_code)
    if backend is None:
        backend = GemmaXBackend()
    yaml_gloss = as_glossary(glossary)
    texts = [c.text for c in cues]
    src_original = list(texts)
    file_map, ginza_count, file_persons = extract_file_glossary(
        texts, langs, load=True
    )
    yaml_map = dict(yaml_gloss.mapping) if yaml_gloss is not None else {}
    yaml_persons = set(yaml_gloss.person_keys) if yaml_gloss is not None else set()
    merged = merge_mappings(file_map, yaml_map)
    person_keys = set(file_persons) | yaml_persons
    gloss = Glossary(merged, person_keys=person_keys) if merged else None
    status(f"GiNZA entities {ginza_count} (pykakasi {pykakasi_version()})")

    translated: list[str | None] = [None] * len(texts)
    groups: dict[str, list[int]] = {}
    speaker_prefix: list[str | None] = [None] * len(texts)
    left_jp: list[bool] = [False] * len(texts)
    hepburn_count = 0

    for i, text in enumerate(texts):
        src_iso = langs[i]
        if same_language(src_iso, to_code):
            translated[i] = text
            continue
        src = to_flores(src_iso)
        if gloss is not None:
            prefix, body = gloss.prepare_mt_body(text)
            speaker_prefix[i] = prefix
            if not body.strip():
                translated[i] = prefix or ""
                continue
            send = body
            if send != text:
                hepburn_count += 1
        else:
            send = strip_caption_arrows(text)
        groups.setdefault(src, []).append(i)
        texts[i] = send

    status(f"in-source Person Hepburn {hepburn_count}")

    send_count = sum(len(idx) for idx in groups.values())
    prepare = getattr(backend, "prepare", None)
    if callable(prepare) and send_count:
        prepare()

    leave_jp = 0

    if not groups:
        finals = [t if t is not None else "" for t in translated]
        apply_translations(doc, finals)
        _log_mt_stats(backend, finals, elapsed=None, leave_jp=leave_jp, left_jp=left_jp)
        src_display = to_flores(langs[0]) if langs else tgt
        return doc, src_display, tgt

    if len(groups) == 1:
        only_src = next(iter(groups))
        status(f"Translating {send_count} cues ({only_src} → {tgt})")
    else:
        status(f"Translating {send_count} cues (per-cue src → {tgt})")

    started = time.monotonic()
    for src, idxs in groups.items():
        to_send = [texts[i] for i in idxs]
        batch = backend.translate(to_send, src, tgt)
        if len(batch) != len(to_send):
            raise RuntimeError(
                f"Translator returned {len(batch)} texts for {len(to_send)} cues."
            )
        for j, i in enumerate(idxs):
            body = batch[j]
            original = src_original[i]
            if gloss is not None:
                body = gloss.overlay_names(original, body)
            if not body.strip() or is_majority_jp(body):
                status(f"Cue {i + 1}: empty or majority-JP; leave JP (no retry)")
                leave_jp += 1
                left_jp[i] = True
                translated[i] = original
                continue
            if speaker_prefix[i]:
                body = (
                    speaker_prefix[i]
                    if not body
                    else f"{speaker_prefix[i]} {body}"
                )
            translated[i] = body
    elapsed = time.monotonic() - started
    finals = [t if t is not None else "" for t in translated]
    apply_translations(doc, finals)
    _log_mt_stats(
        backend, finals, elapsed=elapsed, leave_jp=leave_jp, left_jp=left_jp
    )
    first_src = to_flores(langs[0]) if langs else tgt
    return doc, first_src, tgt


def _log_mt_stats(
    backend: object,
    texts: list[str],
    *,
    elapsed: float | None,
    leave_jp: int = 0,
    left_jp: list[bool] | None = None,
) -> None:
    counts = getattr(backend, "finish_reason_counts", None)
    stop = 0
    length = 0
    if counts is not None:
        stop = int(counts.get("stop", 0))
        length = int(counts.get("length", 0))
    status(f"finish_reason stop={stop} length={length}")
    flags = left_jp or [False] * len(texts)
    arrows = 0
    for text, copied in zip(texts, flags, strict=False):
        if copied:
            continue
        arrows += leftover_arrow_count(text)
    status(f"leftover arrows {arrows}")
    status(f"leave JP {leave_jp} (empty or majority-JP; no retry)")
    if elapsed is not None:
        status(f"MT pass {elapsed:.1f}s")


def translate_file(
    input_path: str | Path,
    to_code: str,
    from_code: str | None = None,
    output_path: str | Path | None = None,
    backend: TranslatorBackend | None = None,
    glossary: Glossary | str | Path | None = None,
) -> Path:
    inp = Path(input_path)
    if not inp.is_file():
        raise FileNotFoundError(f"Input file not found: {inp}")
    out = Path(output_path) if output_path else default_output_path(inp, to_code)
    doc = load(inp)
    cue_langs: list[str | None] | None = None
    sidecar = cues_jsonl_path(inp)
    if sidecar.is_file():
        cue_langs = langs_from_sidecar(sidecar, doc)
    unload_whisper()
    translate_document(
        doc,
        to_code=to_code,
        from_code=from_code,
        backend=backend,
        glossary=glossary,
        cue_langs=cue_langs,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save(doc, out)
    return out


def backend_from_name(
    name: str,
    device: str,
    batch_size: int,
    model: str | None = None,
    gguf: str | Path | None = None,
    name_hint: bool = False,
) -> TranslatorBackend:
    if name in {"echo", "identity"}:
        return EchoBackend()
    if name in {"nllb"}:
        return NllbBackend(device=device, batch_size=batch_size, model=model)
    if name in {"gemmax", "gemmax2", "auto", "llama", "llama-cpp"}:
        return GemmaXBackend(
            device=device,
            batch_size=batch_size,
            model=model,
            gguf=gguf,
            name_hint=name_hint,
        )
    raise ValueError(f"Unknown backend {name!r}. Use gemmax, nllb, or echo.")


def product_output_path(inp: Path, to_code: str) -> Path:
    tag = display_code(to_code)
    return inp.with_name(f"{inp.stem}.{tag}.srt")


def run_product(
    input_path: str | Path,
    to_code: str,
    *,
    output_path: str | Path | None = None,
    device: str = "auto",
    glossary: Glossary | str | Path | None = None,
    model: str | None = None,
    language: str | None = None,
    batched: bool = False,
    backend: TranslatorBackend | None = None,
    whisper_model: str = "large-v3",
    gguf: str | Path | None = None,
    name_hint: bool = False,
) -> Path:
    """Transcribe mixed-language media, unload Whisper, GemmaX2 per cue.

    Writes ``INPUT.cues.jsonl`` and ``INPUT.<to>.srt``. No default glossary.
    Whisper ``task=translate`` is never used.
    """
    from sublocal.transcribe import transcribe_document

    inp = Path(input_path)
    if not inp.is_file():
        raise FileNotFoundError(f"Input file not found: {inp}")
    out = Path(output_path) if output_path else product_output_path(inp, to_code)
    doc = transcribe_document(
        inp,
        language=language,
        model_size=whisper_model,
        device=device,
        glossary=glossary,
        batched=batched,
    )
    write_cues_jsonl(doc, cues_jsonl_path(inp))
    unload_whisper()
    if backend is None:
        backend = GemmaXBackend(
            device=device, model=model, gguf=gguf, name_hint=name_hint
        )
    cue_langs = [c.extra.get("lang") for c in doc.cues]
    translate_document(
        doc,
        to_code=to_code,
        backend=backend,
        glossary=glossary,
        cue_langs=cue_langs,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save(doc, out)
    return out
