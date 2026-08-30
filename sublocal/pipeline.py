"""Parse → detect → translate cue text → write. Timestamps are never sent to the model."""

from __future__ import annotations

from pathlib import Path

from sublocal.backend import EchoBackend, NllbBackend, TranslatorBackend
from sublocal.detect import detect_iso639
from sublocal.formats import dumps, load, save
from sublocal.formats.base import Document
from sublocal.glossary import (
    Glossary,
    GlossaryError,
    as_glossary,
    has_cjk,
    needs_nllb,
)
from sublocal.languages import display_code, to_flores
from sublocal.progress import status


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


def translate_document(
    doc: Document,
    to_code: str,
    from_code: str | None = None,
    backend: TranslatorBackend | None = None,
    glossary: Glossary | str | Path | None = None,
) -> tuple[Document, str, str]:
    """Translate cue text in-place. Returns (doc, src_flores, tgt_flores)."""
    cues = doc.cues
    if not cues:
        raise ValueError("No subtitle cues found in the input file.")

    if from_code:
        src = to_flores(from_code)
        src_display = from_code
    else:
        src_display = detect_iso639([c.text for c in cues])
        src = to_flores(src_display)
    tgt = to_flores(to_code)
    if backend is None:
        backend = NllbBackend()
    gloss = as_glossary(glossary)
    texts = [c.text for c in cues]
    prepare = getattr(backend, "prepare", None)
    if gloss is None:
        if callable(prepare):
            prepare()
        status(f"Translating {len(texts)} cues ({src} → {tgt})")
        translated = backend.translate(texts, src, tgt)
        apply_translations(doc, translated)
        return doc, src, tgt

    translated = [""] * len(texts)
    send_idx: list[int] = []
    to_send: list[str] = []
    guarded_by_i: list[str] = [""] * len(texts)
    pairs_by_i: list[list[tuple[str, str]]] = [[] for _ in texts]
    speaker_prefix: list[str | None] = [None] * len(texts)

    for i, text in enumerate(texts):
        speaker_jp, rest = gloss.peel_speaker(text)
        if speaker_jp is not None:
            speaker_prefix[i] = f"({gloss.mapping[speaker_jp]})"
            if not has_cjk(rest):
                translated[i] = speaker_prefix[i] or ""
                continue
            text = rest
        guarded, pairs = gloss.protect(text)
        guarded_by_i[i] = guarded
        pairs_by_i[i] = pairs
        if pairs and not needs_nllb(guarded):
            restored = gloss.restore(guarded, pairs, target="value")
            body = gloss.cleanup_adjacent(restored, [v for _, v in pairs])
            translated[i] = _with_speaker(speaker_prefix[i], body)
            continue
        send_idx.append(i)
        to_send.append(guarded)

    if callable(prepare) and to_send:
        prepare()
    status(f"Translating {len(to_send)} cues ({src} → {tgt})")
    batch: list[str] = []
    if to_send:
        batch = backend.translate(to_send, src, tgt)
        if len(batch) != len(to_send):
            raise RuntimeError(
                f"Translator returned {len(batch)} texts for {len(to_send)} cues."
            )

    for j, i in enumerate(send_idx):
        pairs = pairs_by_i[i]
        mt = batch[j]
        if not pairs:
            translated[i] = _with_speaker(speaker_prefix[i], mt)
            continue
        try:
            if gloss.missing_sentinels(mt, pairs):
                xml = gloss.to_xml(guarded_by_i[i], len(pairs))
                retried = backend.translate([xml], src, tgt)
                if len(retried) != 1:
                    raise RuntimeError("XML glossary retry returned the wrong count.")
                mt = retried[0]
            restored = gloss.restore(mt, pairs, target="value")
        except GlossaryError as exc:
            raise GlossaryError(f"Cue {i + 1}: {exc}") from exc
        body = gloss.cleanup_adjacent(restored, [v for _, v in pairs])
        translated[i] = _with_speaker(speaker_prefix[i], body)
    apply_translations(doc, translated)
    return doc, src, tgt


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
    translate_document(
        doc,
        to_code=to_code,
        from_code=from_code,
        backend=backend,
        glossary=glossary,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save(doc, out)
    return out


def backend_from_name(
    name: str,
    device: str,
    batch_size: int,
    model: str | None = None,
) -> TranslatorBackend:
    if name in {"echo", "identity"}:
        return EchoBackend()
    if name in {"nllb", "auto"}:
        return NllbBackend(device=device, batch_size=batch_size, model=model)
    raise ValueError(f"Unknown backend {name!r}. Use nllb or echo.")
