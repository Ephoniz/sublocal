# sublocal

Local-only subtitle translation and transcription. No cloud APIs, no API keys, no telemetry.

v0.5 is one command: transcribe mixed-language audio/video, stamp a language on each cue, unload Whisper, then **GemmaX2-28-9B-v0.1 Q5_K_M** GGUF per cue via llama-cpp-python CUDA.

```bash
sublocal clip.mp4 --to es
```

The end result is `clip.es.srt` (the overlay file); `clip.cues.jsonl` is an optional lang sidecar for debug (`start`, `end`, `text`, `lang`), not the overlay. No `--language` required. Whisper `task=translate` is never the Spanish path.

NLLB is no longer the default. `--glossary` is still opt-in. There is no default `drama.yml`. Latin/ASCII tokens already in a Japanese (or other non-Latin) cue are copied through and not sent to the model (Liu/Zhang stay Latin). Kanji names: first-pass official GemmaX prompt only; no NER.

`sublocal transcribe` and `sublocal translate in.srt --to es` stay as debug commands.

## Install

Use **official CPython 3.11+** from [python.org](https://www.python.org/downloads/) or [uv](https://docs.astral.sh/uv/). Python 3.12 from those sources is fine.

**Do not use Anaconda or Miniconda Python on Windows.** CTranslate2 (Whisper) hits a native access violation (`0xC0000005`) with no traceback on Anaconda 3.12.x. Official CPython 3.11 loads the same cached models. `sublocal` refuses that combo instead of crashing silently.

```bash
git clone https://github.com/Ephoniz/sublocal.git
cd sublocal
pip install . --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

GemmaX2 inference needs the **CUDA 12.4** wheel of **`llama-cpp-python==0.3.4`**. The pin is exact so pip does not pull 0.3.35 from that extra index.

**0.3.35 cu124 win_amd64** raises `OSError 0xc000001d` (illegal instruction) while constructing `llama_context`. That wheel's `ggml-cpu.dll` is built with AVX512/AMX; an i7-13700K reports AVX512=false. **0.3.4 cu124 cp311** loads, offloads 43/43 layers, and completed the 4070 Ti prove.

If `llama-cpp-python` is already installed (including 0.3.35):

```bash
pip install --force-reinstall --no-cache-dir "llama-cpp-python==0.3.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Do not `pip install llama-cpp-python` unpinned from the extra index — it resolves 0.3.35.

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
sublocal clip.mp4 --to es --model q6
sublocal clip.mp4 --to es --gguf /path/to/GemmaX2-28-9B-v0.1.Q5_K_M.gguf
sublocal clip.mp4 --to es --glossary examples/drama.yml
sublocal clip.mp4 --to es --batch
```

`INPUT` is audio or video. `--to` is required. Optional: `--out`, `--device`, `--glossary`, `--model` (GemmaX2 quant: `q5` default, `q6` fallback), `--gguf` (local GGUF path), `--language` (mono ASR override), `--batch` (batched Whisper; `without_timestamps` is False).

Default ASR iterates Silero speech timestamps and calls faster-whisper **once per slice** (`language=None`, `multilingual=False`, `task=transcribe`, `without_timestamps=False`). That is first-30s LID on that chunk only — mixed JA+EN is not glued into one 30s window. Sequential, one slice at a time (fits a 12 GB card). `--language ja` is a mono override: one full-file call, `multilingual=False`. `--batch` is optional and still passes `without_timestamps=False`.

After the source SRT would be written, Whisper is unloaded so it never shares VRAM with the GGUF. Each cue uses that cue's Whisper ISO → English source name (`ja`→Japanese, `en`→English, `es`→Spanish) in the official GemmaX2 completion prompt. Cues already in `--to` are copied through.

## Debug: transcribe / translate

```bash
sublocal transcribe movie.mp4
sublocal transcribe movie.mp4 --language ja --out movie.srt
sublocal translate movie.srt --to es
sublocal translate input.srt --to es --from ja --glossary examples/drama.yml
sublocal translate input.srt --to es --model q6
```

`transcribe` writes source-language SRT plus `INPUT.cues.jsonl`. `translate` reads that sidecar for per-cue lang when present; otherwise it uses a script heuristic (Hiragana/Katakana/CJK → ja, Hangul → ko, Latin → en) and lingua-py for short Latin cues if `[lid]` is installed.

`extract` still prints `not in v0.1` and exits 2.

## Translate details

Default model is **GemmaX2-28-9B-v0.1 Q5_K_M** GGUF through llama-cpp-python (`n_gpu_layers=-1`, `n_ctx=2048`; CUDA OOM retries once with `n_ctx=1024` before considering `--model q6`). Generation is greedy (`temperature=0`, `top_k=1`, `max_tokens=256`). Sequential: one cue at a time. The MT pass logs elapsed seconds on stderr (hundreds of cues take minutes).

`--from` is optional. `--glossary` is a UTF-8 YAML map of source names → Latin (see `examples/drama.yml`). It is never loaded unless you pass `--glossary`. There is no default `drama.yml` and no genre-tuned prompt.

Latin/ASCII already in a non-Latin cue (`Liu`, `Zhang`, `Drum`) is copied through and not sent to the model. Kanji/kana names are left to the official GemmaX2 prompt. No NER, no cutlet/pykakasi. An optional name sentence (`Keep person and place names. …`) exists as `name_hint=False` on the backend and is off in v0.5.0.

`--device auto` (the default) uses CUDA when `get_cuda_device_count()` is greater than 0 and prints `Using NVIDIA GeForce RTX 4070 Ti (cuda:0)` (or the real `nvidia-smi` name). Official CTranslate2 4.8 Windows wheels dynamically load CUDA 12 (`cublas64_12.dll`, `cudart64_12.dll`); a leftover `CUDA_VISIBLE_DEVICES=-1` or empty value is dropped in this process only. If the count is still 0, stderr prints why (`CUDA_PATH` unset or pointing at v13.x, missing `cublas64_12.dll` on PATH / `CUDA_PATH\bin`) and the run continues so a file still gets written. `--device cuda` exits 1 with the same diagnostic instead of faking CPU.

Progress goes to stderr: download (tqdm, extra), cache/load, `Model ready (device=cuda)` only after `Llama` is constructed, then `Translating N cues`, per-cue/batch lines, and `MT pass Xs`. PowerShell eats tqdm `\r` bars, so the line counter is the real signal. The output path is the only stdout line.

`.srt` is the supported format. `.vtt` and `.ass` are best-effort: timings are kept, styling may not be perfect.

## Transcribe details

Default model is **faster-whisper large-v3** (`Systran/faster-whisper-large-v3`) on CUDA with float16 (int8 on CPU). Audio is decoded in-process with PyAV to 16 kHz mono and passed into Whisper as an array. `ffmpeg` is not required for `transcribe` (it is only for optional `extract` later). First run downloads ~3 GB into the same Hugging Face cache as translate if that snapshot is not already there.

Cues are sentence-sized: word timestamps, Silero VAD at ~500 ms silence, then regroup (Japanese `。` / `？`, 0.5 s gaps — do not merge across ≥0.5 s — ~4 s duration cap, tiny-gap merge, ~32 characters / two lines). Timestamps are first-word start → last-word end, not Whisper's raw 30 s windows. Cue length is capped at ~4 s.

Progress on stderr is new flushed lines (`12/45s (26%) ~30s left`). The output path is the only stdout line. The Whisper model is unloaded after the SRT (and sidecar) are written so a later `translate` in the same process can use the GPU.

When using `--glossary` for Japanese, pass `--language ja`. Glossary keys are given to Whisper as `initial_prompt` (`condition_on_previous_text=False`); after ASR they are restored as the **Japanese** spellings, never Latin.

## Models

**Translate.** Default: **GemmaX2-28-9B-v0.1 Q5_K_M** GGUF through **llama-cpp-python==0.3.4** (cu124 extra index). Not 0.3.35. Not transformers `generate`. Not NLLB-200 3.3B.

Default weights: [`mradermacher/GemmaX2-28-9B-v0.1-GGUF`](https://huggingface.co/mradermacher/GemmaX2-28-9B-v0.1-GGUF) file `GemmaX2-28-9B-v0.1.Q5_K_M.gguf` (6.65 GB). Card: [`ModelSpace/GemmaX2-28-9B-v0.1`](https://huggingface.co/ModelSpace/GemmaX2-28-9B-v0.1). Paper: [arxiv 2502.02481](https://arxiv.org/abs/2502.02481). Fallback quant: `--model q6` (`GemmaX2-28-9B-v0.1.Q6_K.gguf`, 7.59 GB). Do not default Q4.

The official completion prompt (28-language English names, not FLORES, not chat):

```
Translate this from Japanese to Spanish:
Japanese: <cue>
Spanish:
```

`--to es` → Spanish. Cue `lang` from the v0.4 stamp → source English name (`ja`→Japanese). If source equals target, the cue is copied through.

NLLB-200 via CTranslate2 remains available only as a hidden `--backend nllb` (not the product default). NLLB weight families are **CC-BY-NC-4.0** (non-commercial). GemmaX2 inherits the Gemma terms of the base model. The CLI itself is MIT.

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
5. **One-command mixed-language → target SRT** — v0.4.
6. **GemmaX2-28-9B GGUF default MT** — this release.
7. **Burned-in OCR** — later. Not faked.

Helsinki-NLP Opus-MT (when a pair exists) may land later as a smaller per-pair option.
