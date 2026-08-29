# sublocal

Local-only subtitle translation with small models. No cloud APIs, no API keys, no telemetry.

v0.1 translates an existing subtitle file and writes a new one. Timestamps stay as they were; only cue text is sent to the model.

## Install

Use **official CPython 3.11+** from [python.org](https://www.python.org/downloads/) or [uv](https://docs.astral.sh/uv/). Python 3.12 from those sources is fine.

**Do not use Anaconda or Miniconda Python on Windows.** CTranslate2 hits a native access violation (`0xC0000005`) with no traceback on Anaconda 3.12.x. The same cached NLLB model loads on official CPython 3.11.4 (CPU). `sublocal` refuses to load the translator on that combo instead of crashing silently.

GPU is optional (CUDA via CTranslate2). CPU works.

```bash
git clone https://github.com/Ephoniz/sublocal.git
cd sublocal
pip install .
```

Dev / tests:

```bash
pip install -e ".[dev]"
pytest
```

## Translate

```bash
sublocal translate input.srt --to en
sublocal translate input.srt --to en --from es --out input.en.srt
```

`--from` is optional; source language is detected from cue text when omitted.

Progress (first-run download bars, cache/load, cue counts) goes to stderr. The output path is printed on stdout when the file is written.

`.srt` is the supported format. `.vtt` and `.ass` are best-effort: timings are kept, styling may not be perfect.

`extract` and `transcribe` exist as stubs and print `not in v0.1`.

## Models

Default: **NLLB-200 distilled 600M** through CTranslate2 (int8). Fits a 12 GB card easily; also runs on CPU.

First run downloads ~600 MB into the user cache (no Hugging Face token):

- Linux / macOS: `~/.cache/sublocal/huggingface/`
- Windows: `%LOCALAPPDATA%\sublocal\huggingface\`
- Override: `SUBLOCAL_CACHE`

Weights used: [`JustFrederik/nllb-200-distilled-600M-ct2-int8`](https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-int8), a CTranslate2 conversion of [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M). That model is **CC-BY-NC-4.0** (non-commercial). The CLI itself is MIT.

`--device auto` uses CUDA when CTranslate2 sees a GPU, otherwise CPU.

## Roadmap

1. **Translate existing subtitle files** — this release.
2. **Extract** soft subtitle tracks from video (`ffmpeg` / `mkvextract`). Not in v0.1.
3. **Transcribe** with faster-whisper `small` or `base`, then the same translate pipeline. Not in v0.1.
4. **Burned-in OCR** — later. Not faked.

Helsinki-NLP Opus-MT (when a pair exists) may land later as a smaller per-pair option. Not in v0.1.
