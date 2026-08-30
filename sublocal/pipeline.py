"""Parse → detect → translate cue text → write. Timestamps are never sent to the model."""

from __future__ import annotations

from pathlib import Path

from sublocal.backend import EchoBackend, NllbBackend, TranslatorBackend
from sublocal.detect import detect_iso639
from sublocal.formats import dumps, load, save
from sublocal.formats.base import Document
from sublocal.glossary import Glossary, GlossaryError, as_glossary
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
    protected_pairs: list[list[tuple[str, str]]] = [[] for _ in texts]
    if gloss is not None:
        protected: list[str] = []
        for i, text in enumerate(texts):
            guarded, pairs = gloss.protect(text)
            protected.append(guarded)
            protected_pairs[i] = pairs
        texts = protected
    prepare = getattr(backend, "prepare", None)
    if callable(prepare):
        prepare()
    status(f"Translating {len(texts)} cues ({src} → {tgt})")
    translated = backend.translate(texts, src, tgt)
    if gloss is not None:
        restored: list[str] = []
        for i, text in enumerate(translated):
            try:
                restored.append(gloss.restore(text, protected_pairs[i], target="value"))
            except GlossaryError as exc:
                raise GlossaryError(f"Cue {i + 1}: {exc}") from exc
        translated = restored
    apply_translations(doc, translated)
    return doc, src, tgt


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
