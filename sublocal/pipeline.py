"""Parse → detect → translate cue text → write. Timestamps are never sent to the model."""

from __future__ import annotations

import time
from pathlib import Path

from sublocal.backend import (
    EchoBackend,
    GemmaXBackend,
    NllbBackend,
    TranslatorBackend,
)
from sublocal.cues_jsonl import cues_jsonl_path, langs_from_sidecar, write_cues_jsonl
from sublocal.detect import DetectionError, detect_iso639
from sublocal.formats import dumps, load, save
from sublocal.formats.base import Cue, Document
from sublocal.glossary import (
    Glossary,
    as_glossary,
    has_cjk,
    needs_nllb,
)
from sublocal.languages import display_code, to_flores
from sublocal.lid import (
    detect_cue_lang,
    is_latin_script,
    protect_latin_names,
    restore_latin_names,
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
    ``to_code`` are copied through. Latin/ASCII tokens in non-Latin cues
    are not sent through the MT backend.
    """
    cues = doc.cues
    if not cues:
        raise ValueError("No subtitle cues found in the input file.")

    langs = resolve_cue_langs(cues, from_code=from_code, cue_langs=cue_langs)
    tgt = to_flores(to_code)
    if backend is None:
        backend = GemmaXBackend()
    gloss = as_glossary(glossary)
    texts = [c.text for c in cues]
    translated: list[str | None] = [None] * len(texts)
    # src_flores → list of cue indices to send
    groups: dict[str, list[int]] = {}
    source_by_i: list[str] = [""] * len(texts)
    latin_by_i: list[list[str]] = [[] for _ in texts]
    speaker_prefix: list[str | None] = [None] * len(texts)

    for i, text in enumerate(texts):
        src_iso = langs[i]
        if same_language(src_iso, to_code):
            translated[i] = text
            continue
        src = to_flores(src_iso)
        work = text
        if gloss is not None:
            speaker_jp, rest = gloss.peel_speaker(work)
            if speaker_jp is not None:
                speaker_prefix[i] = f"({gloss.mapping[speaker_jp]})"
                if not has_cjk(rest):
                    translated[i] = speaker_prefix[i] or ""
                    continue
                work = rest
            guarded, pairs = gloss.protect(work)
            if pairs and not needs_nllb(guarded):
                restored = gloss.restore(guarded, pairs, target="value")
                body = gloss.cleanup_adjacent(restored, [v for _, v in pairs])
                translated[i] = _with_speaker(speaker_prefix[i], body)
                continue
        source_by_i[i] = work
        send = work
        if not is_latin_script(src):
            send, latin_by_i[i] = protect_latin_names(work)
        groups.setdefault(src, []).append(i)
        texts[i] = send  # what we actually send

    send_count = sum(len(idx) for idx in groups.values())
    prepare = getattr(backend, "prepare", None)
    if callable(prepare) and send_count:
        prepare()

    if not groups:
        apply_translations(doc, [t if t is not None else "" for t in translated])
        src_display = to_flores(langs[0]) if langs else tgt
        return doc, src_display, tgt

    # One status line; keep the old shape when every cue shares a source.
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
            mt = batch[j]
            if latin_by_i[i]:
                mt = restore_latin_names(mt, latin_by_i[i])
            if gloss is not None:
                source = source_by_i[i]
                body = gloss.overlay_names(source, mt)
                latins = [v for k, v in gloss.entries if k in source]
                if latins:
                    body = gloss.cleanup_adjacent(body, latins)
                translated[i] = _with_speaker(speaker_prefix[i], body)
            else:
                translated[i] = mt
    status(f"MT pass {time.monotonic() - started:.1f}s")

    apply_translations(doc, [t if t is not None else "" for t in translated])
    first_src = to_flores(langs[0]) if langs else tgt
    return doc, first_src, tgt


def _with_speaker(prefix: str | None, body: str) -> str:
    if not prefix:
        return body
    if not body:
        return prefix
    return f"{prefix} {body}"


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
