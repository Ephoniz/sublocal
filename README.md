# sublocal

Local-only subtitle translation with small models. No cloud APIs, no API keys, no telemetry.

v0.1 translates an existing subtitle file and writes a new one. Timestamps stay as they were; only cue text is sent to the model.

## Install

Use **official CPython 3.11+** from [python.org](https://www.python.org/downloads/) or [uv](https://docs.astral.sh/uv/). Python 3.12 from those sources is fine.

**Do not use Anaconda or Miniconda Python on Windows.** CTranslate2 hits a native access violation (`0xC0000005`) with no traceback on Anaconda 3.12.x. Official CPython 3.11 loads the same cached NLLB model. `sublocal` refuses that combo instead of crashing silently.

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

`--device auto` (the default) uses CUDA when `get_cuda_device_count()` is greater than 0 and prints `Using NVIDIA GeForce RTX 4070 Ti (cuda:0)` (or the real `nvidia-smi` name). Official CTranslate2 4.8 Windows wheels dynamically load CUDA 12 (`cublas64_12.dll`, `cudart64_12.dll`); a leftover `CUDA_VISIBLE_DEVICES=-1` or empty value is dropped in this process only. If the count is still 0, stderr prints why (`CUDA_PATH` unset or pointing at v13.x, missing `cublas64_12.dll` on PATH / `CUDA_PATH\bin`) and the run continues so a file still gets written. `--device cuda` exits 1 with the same diagnostic instead of faking CPU.

Progress goes to stderr: download (tqdm, extra), cache/load, `Model ready (device=cuda)` only after `Translator` is constructed, then `Translating N cues` and one **new line per batch** (`128/599 cues (21%) ~2m left`). PowerShell eats tqdm `\r` bars, so the line counter is the real signal. The output path is the only stdout line.

`.srt` is the supported format. `.vtt` and `.ass` are best-effort: timings are kept, styling may not be perfect.

`extract` and `transcribe` exist as stubs and print `not in v0.1`.

## Models

Default: **NLLB-200 distilled 600M** through CTranslate2 (int8 on CPU, `int8_float16` on CUDA). Fits a 12 GB card.

First run downloads ~600 MB into the user cache (no Hugging Face token):

- Linux / macOS: `~/.cache/sublocal/huggingface/`
- Windows: `%LOCALAPPDATA%\sublocal\huggingface\`
- Override: `SUBLOCAL_CACHE`

Weights used: [`JustFrederik/nllb-200-distilled-600M-ct2-int8`](https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-int8), a CTranslate2 conversion of [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M). That model is **CC-BY-NC-4.0** (non-commercial). The CLI itself is MIT.

## Roadmap

1. **Translate existing subtitle files** — this release.
2. **Extract** soft subtitle tracks from video (`ffmpeg` / `mkvextract`). Not in v0.1.
3. **Transcribe** with faster-whisper `small` or `base`, then the same translate pipeline. Not in v0.1.
4. **Burned-in OCR** — later. Not faked.

Helsinki-NLP Opus-MT (when a pair exists) may land later as a smaller per-pair option. Not in v0.1.
