# sublocal

Local-only subtitle translation and transcription. No cloud APIs, no API keys, no telemetry.

v0.4 is one command: transcribe mixed-language audio/video, stamp a language on each cue, unload Whisper, then NLLB-200 **3.3B** per cue.

```bash
sublocal clip.mp4 --to es
```

Writes `clip.es.srt` and a sidecar `clip.cues.jsonl` (`start`, `end`, `text`, `lang`). No `--language` required. Whisper `task=translate` is never the Spanish path.

`--glossary` is opt-in. There is no default `drama.yml`. Latin/ASCII tokens already in a Japanese (or other non-Latin) cue are copied through and not sent to NLLB. **Kanji/kana names may still mangle.**

`sublocal transcribe` and `sublocal translate in.srt --to es` stay as debug commands.

## Install

Use **official CPython 3.11+** from [python.org](https://www.python.org/downloads/) or [uv](https://docs.astral.sh/uv/). Python 3.12 from those sources is fine.

**Do not use Anaconda or Miniconda Python on Windows.** CTranslate2 hits a native access violation (`0xC0000005`) with no traceback on Anaconda 3.12.x. Official CPython 3.11 loads the same cached models. `sublocal` refuses that combo instead of crashing silently.

```bash
git clone https://github.com/Ephoniz/sublocal.git
cd sublocal
pip install .
```

Short-cue Latin LID when ASR did not stamp lang (optional, local, no API). Package pin is `lingua-language-detector==1.4.2` (2.x needs Python 3.12; this project supports 3.11). Restricted to JA/EN/ES/KO. Main deps pin `regex>=2025.10.22` (transformers 5.16); lingua 1.4.2 asks for `regex<2025`, so a strict resolver will refuse `.[lid]` rather than downgrade regex. Install lingua without its regex pin:

```bash
pip install ".[lid]"
# if pip reports a regex conflict:
pip install --no-deps lingua-language-detector==1.4.2
```

Dev / tests:

```bash
pip install -e ".[dev]"
pytest
```

## Product command

```bash
sublocal movie.mp4 --to es
sublocal interview.wav --to es --out interview.es.srt
sublocal clip.mkv --to es --language ja
sublocal clip.mp4 --to es --model small
sublocal clip.mp4 --to es --glossary examples/drama.yml
sublocal clip.mp4 --to es --batch
```

`INPUT` is audio or video. `--to` is required. Optional: `--out`, `--device`, `--glossary`, `--model` (NLLB size), `--language` (mono ASR override), `--batch` (batched Whisper; `without_timestamps` is False).

Default ASR iterates Silero speech timestamps and calls faster-whisper **once per slice** (`language=None`, `multilingual=False`, `task=transcribe`, `without_timestamps=False`). That is first-30s LID on that chunk only — mixed JA+EN is not glued into one 30s window. Sequential, one slice at a time (fits a 12 GB card). `--language ja` is a mono override: one full-file call, `multilingual=False`. `--batch` is optional and still passes `without_timestamps=False`.

After the source SRT would be written, Whisper is unloaded so it never shares VRAM with NLLB. Each cue is encoded with that cue's Whisper ISO → FLORES source (`ja`→`jpn_Jpan`, `en`→`eng_Latn`, `es`→`spa_Latn`). Cues already in `--to` are copied through.

## Debug: transcribe / translate

```bash
sublocal transcribe movie.mp4
sublocal transcribe movie.mp4 --language ja --out movie.srt
sublocal translate movie.srt --to es
sublocal translate input.srt --to es --from ja --glossary examples/drama.yml
sublocal translate input.srt --to es --model small
```

`transcribe` writes source-language SRT plus `INPUT.cues.jsonl`. `translate` reads that sidecar for per-cue lang when present; otherwise it uses a script heuristic (Hiragana/Katakana/CJK → ja, Hangul → ko, Latin → en) and lingua-py for short Latin cues if `[lid]` is installed.

`extract` still prints `not in v0.1` and exits 2.

## Translate details

Default model is **NLLB-200 3.3B** through CTranslate2 on CUDA with float16 (`int8_float16` on CUDA OOM). `--model small` is the distilled 600M. `--model large` is an alias for 3.3b.

`--from` is optional. `--glossary` is a UTF-8 YAML map of source names → Latin (see `examples/drama.yml`). It is never loaded unless you pass `--glossary`. The original Japanese sentence is sent to NLLB (バンコク stays inside バンコクに飛んだ). After MT, Latin names are overlaid and leftover particles next to those names are stripped.

Latin/ASCII already in a non-Latin cue (`Drum`, `Nozaki`) is copied through and not sent to NLLB. Kanji/kana names without `--glossary` may still mangle.

`--device auto` (the default) uses CUDA when `get_cuda_device_count()` is greater than 0 and prints `Using NVIDIA GeForce RTX 4070 Ti (cuda:0)` (or the real `nvidia-smi` name). Official CTranslate2 4.8 Windows wheels dynamically load CUDA 12 (`cublas64_12.dll`, `cudart64_12.dll`); a leftover `CUDA_VISIBLE_DEVICES=-1` or empty value is dropped in this process only. If the count is still 0, stderr prints why (`CUDA_PATH` unset or pointing at v13.x, missing `cublas64_12.dll` on PATH / `CUDA_PATH\bin`) and the run continues so a file still gets written. `--device cuda` exits 1 with the same diagnostic instead of faking CPU.

Progress goes to stderr: download (tqdm, extra), cache/load, `Model ready (device=cuda)` only after `Translator` is constructed, then `Translating N cues` and one **new line per batch** (`128/599 cues (21%) ~2m left`). PowerShell eats tqdm `\r` bars, so the line counter is the real signal. The output path is the only stdout line.

`.srt` is the supported format. `.vtt` and `.ass` are best-effort: timings are kept, styling may not be perfect.

## Transcribe details

Default model is **faster-whisper large-v3** (`Systran/faster-whisper-large-v3`) on CUDA with float16 (int8 on CPU). Audio is decoded in-process with PyAV to 16 kHz mono and passed into Whisper as an array. `ffmpeg` is not required for `transcribe` (it is only for optional `extract` later). First run downloads ~3 GB into the same Hugging Face cache as translate if that snapshot is not already there.

Cues are sentence-sized: word timestamps, Silero VAD at ~500 ms silence, then regroup (Japanese `。` / `？`, 0.5 s gaps — do not merge across ≥0.5 s — ~4 s duration cap, tiny-gap merge, ~32 characters / two lines). Timestamps are first-word start → last-word end, not Whisper's raw 30 s windows. Cue length is capped at ~4 s.

Progress on stderr is new flushed lines (`12/45s (26%) ~30s left`). The output path is the only stdout line. The Whisper model is unloaded after the SRT (and sidecar) are written so a later `translate` in the same process can use the GPU.

When using `--glossary` for Japanese, pass `--language ja`. Glossary keys are given to Whisper as `initial_prompt` (`condition_on_previous_text=False`); after ASR they are restored as the **Japanese** spellings, never Latin.

## Models

**Translate.** Default: **NLLB-200 3.3B** through CTranslate2 (`float16` on CUDA; if `Translator()` raises CUDA OOM it retries once with `int8_float16`). `--model small` is the distilled 600M.

Default weights: [`entai2965/nllb-200-3.3B-ctranslate2-float16`](https://huggingface.co/entai2965/nllb-200-3.3B-ctranslate2-float16), tokenizer [`facebook/nllb-200-3.3B`](https://huggingface.co/facebook/nllb-200-3.3B). Small: [`JustFrederik/nllb-200-distilled-600M-ct2-int8`](https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-int8), tokenizer [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M). Both NLLB weight families are **CC-BY-NC-4.0** (non-commercial). The CLI itself is MIT.

The first 3.3B download is large (~6GB+). The 600M default turned 野崎/ドラム/東条/新庄 into Nagasaki/tambor/hacienda; 3.3B is the default so those names survive with `--glossary`.

**Transcribe.** Default: **faster-whisper large-v3** through CTranslate2 (float16 on CUDA, int8 on CPU). First-run weights: [`Systran/faster-whisper-large-v3`](https://huggingface.co/Systran/faster-whisper-large-v3).

First-run downloads go into the user cache (no Hugging Face token):

- Linux / macOS: `~/.cache/sublocal/huggingface/`
- Windows: `%LOCALAPPDATA%\sublocal\huggingface\`
- Override: `SUBLOCAL_CACHE`

## Roadmap

1. **Translate existing subtitle files** — v0.1.
2. **Extract** soft subtitle tracks from video (`ffmpeg` / `mkvextract`). Not auto-chained. Still a stub.
3. **Transcribe** audio/video to sentence-sized SRT — v0.2.
4. **Translate default 3.3B + glossary** — v0.3.
5. **One-command mixed-language → target SRT** — this release.
6. **Burned-in OCR** — later. Not faked.

Helsinki-NLP Opus-MT (when a pair exists) may land later as a smaller per-pair option.
