from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sublocal import __version__
from sublocal.detect import DetectionError
from sublocal.formats import UnsupportedFormatError
from sublocal.languages import UnknownLanguageError
from sublocal.device import CudaUnavailableError, unhide_cuda_env
from sublocal.glossary import GlossaryError
from sublocal.pipeline import backend_from_name, run_product, translate_file
from sublocal.runtime import UnsupportedPythonError
from sublocal.transcribe import transcribe_file

NOT_IN_V01_EXTRACT = (
    "not in v0.1: extract existing subtitle tracks from video "
    "(ffmpeg/mkvextract). Soft subs only."
)

COMMANDS = frozenset({"translate", "transcribe", "extract"})
MEDIA_SUFFIXES = frozenset(
    {
        ".mp4",
        ".mkv",
        ".mov",
        ".avi",
        ".webm",
        ".m4v",
        ".wmv",
        ".mpg",
        ".mpeg",
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".ogg",
        ".aac",
        ".wma",
        ".opus",
        ".aiff",
        ".aif",
        ".ts",
        ".mts",
        ".m2ts",
    }
)

_EPILOG = (
    "product:  sublocal INPUT --to LANG   "
    "transcribe mixed-language audio/video, then NLLB per cue "
    "(writes INPUT.<lang>.srt and INPUT.cues.jsonl). "
    "transcribe / translate remain as debug commands."
)


def looks_like_media(value: str) -> bool:
    if not value or value in COMMANDS or value.startswith("-"):
        return False
    return Path(value).suffix.lower() in MEDIA_SUFFIXES


def is_product_argv(argv: list[str]) -> bool:
    if not argv:
        return False
    if argv[0] in COMMANDS or argv[0] in {"-h", "--help", "--version"}:
        return False
    return looks_like_media(argv[0]) and "--to" in argv


def build_product_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sublocal",
        description=(
            "Transcribe mixed-language media and translate per cue. "
            "No cloud APIs, no API keys, no telemetry."
        ),
    )
    parser.add_argument("input", help="Path to an audio or video file")
    parser.add_argument(
        "--to",
        required=True,
        metavar="LANG",
        help="Target language (en, es, ja, or a FLORES code like spa_Latn)",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output SRT path (default: input.<to>.srt)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto (default) uses CUDA when CTranslate2 sees a GPU; cuda requires a GPU",
    )
    parser.add_argument(
        "--glossary",
        default=None,
        metavar="PATH",
        help="Optional YAML map of source names to keep (opt-in; no default file)",
    )
    parser.add_argument(
        "--model",
        default="3.3b",
        choices=("small", "3.3b", "large"),
        help="NLLB size: 3.3b (default) or small (600M). large aliases 3.3b.",
    )
    parser.add_argument(
        "--language",
        default=None,
        metavar="LANG",
        help="Mono ASR override (e.g. ja). Default is mixed-language Whisper.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batched Whisper (without_timestamps=False). Sequential is the default.",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("SUBLOCAL_BACKEND", "nllb"),
        help=argparse.SUPPRESS,
    )
    return parser


def parse_product_args(argv: list[str]) -> argparse.Namespace:
    return build_product_parser().parse_args(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sublocal",
        description=(
            "Local-only subtitle translation and transcription. "
            "No cloud APIs, no API keys, no telemetry."
        ),
        epilog=_EPILOG,
    )
    parser.add_argument(
        "--version", action="version", version=f"sublocal {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser(
        "translate",
        help="Translate an existing subtitle file; keep timestamps. Debug.",
    )
    tr.add_argument("input", help="Path to a .srt, .vtt, or .ass file")
    tr.add_argument(
        "--to",
        required=True,
        metavar="LANG",
        help="Target language (en, es, ja, or a FLORES code like eng_Latn)",
    )
    tr.add_argument(
        "--from",
        dest="from_lang",
        default=None,
        metavar="LANG",
        help="Source language. Per-cue sidecar or text LID if omitted.",
    )
    tr.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output path (default: input.<to>.<ext>)",
    )
    tr.add_argument(
        "--glossary",
        default=None,
        metavar="PATH",
        help="Optional YAML map of source names to keep (e.g. examples/drama.yml)",
    )
    tr.add_argument(
        "--model",
        default="3.3b",
        choices=("small", "3.3b", "large"),
        help="NLLB size: 3.3b (default) or small (600M). large aliases 3.3b.",
    )
    tr.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto (default) uses CUDA when CTranslate2 sees a GPU; cuda requires a GPU",
    )
    tr.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Cues per CTranslate2 batch (default: 32)",
    )
    tr.add_argument(
        "--backend",
        default=os.environ.get("SUBLOCAL_BACKEND", "nllb"),
        help=argparse.SUPPRESS,
    )

    extract = sub.add_parser(
        "extract",
        help="Extract soft subtitle tracks from video (not in v0.1).",
    )
    extract.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    transcribe = sub.add_parser(
        "transcribe",
        help="Transcribe audio or video to source-language SRT. Debug.",
    )
    transcribe.add_argument("input", help="Path to an audio or video file")
    transcribe.add_argument(
        "--language",
        default=None,
        metavar="LANG",
        help="Whisper language code (e.g. ja). Mixed-language if omitted.",
    )
    transcribe.add_argument(
        "--glossary",
        default=None,
        metavar="PATH",
        help="YAML names for Whisper prompt and JP canonical spelling",
    )
    transcribe.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model size (default: large-v3). Larger models are rejected.",
    )
    transcribe.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output SRT path (default: input with .srt suffix)",
    )
    transcribe.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto (default) uses CUDA when CTranslate2 sees a GPU; cuda requires a GPU",
    )
    transcribe.add_argument(
        "--batch",
        action="store_true",
        help="Batched Whisper (without_timestamps=False). Sequential is the default.",
    )

    return parser


def _cmd_translate(args: argparse.Namespace) -> int:
    try:
        backend = backend_from_name(
            args.backend, args.device, args.batch_size, model=args.model
        )
        out = translate_file(
            args.input,
            to_code=args.to,
            from_code=args.from_lang,
            output_path=args.out,
            backend=backend,
            glossary=args.glossary,
        )
    except (
        FileNotFoundError,
        ValueError,
        UnknownLanguageError,
        DetectionError,
        UnsupportedFormatError,
        UnsupportedPythonError,
        CudaUnavailableError,
        GlossaryError,
        OSError,
        RuntimeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(out)
    return 0


def _cmd_transcribe(args: argparse.Namespace) -> int:
    try:
        out = transcribe_file(
            args.input,
            language=args.language,
            model_size=args.model,
            output_path=args.out,
            device=args.device,
            glossary=args.glossary,
            batched=args.batch,
        )
    except (
        FileNotFoundError,
        ValueError,
        UnsupportedPythonError,
        CudaUnavailableError,
        GlossaryError,
        OSError,
        RuntimeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(out)
    return 0


def _cmd_product(args: argparse.Namespace) -> int:
    try:
        backend = backend_from_name(
            args.backend, args.device, 32, model=args.model
        )
        out = run_product(
            args.input,
            to_code=args.to,
            output_path=args.out,
            device=args.device,
            glossary=args.glossary,
            model=args.model,
            language=args.language,
            batched=args.batch,
            backend=backend,
        )
    except (
        FileNotFoundError,
        ValueError,
        UnknownLanguageError,
        DetectionError,
        UnsupportedFormatError,
        UnsupportedPythonError,
        CudaUnavailableError,
        GlossaryError,
        OSError,
        RuntimeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(out)
    return 0


def _cmd_stub(message: str) -> int:
    print(message)
    return 2


def main(argv: list[str] | None = None) -> int:
    # Before any ctranslate2 import in this process.
    unhide_cuda_env()
    argv = list(sys.argv[1:] if argv is None else argv)
    if is_product_argv(argv):
        return _cmd_product(parse_product_args(argv))
    if argv and looks_like_media(argv[0]) and "--to" not in argv:
        print(
            "error: sublocal INPUT --to LANG transcribes then translates. "
            "Pass --to, or use: sublocal transcribe INPUT",
            file=sys.stderr,
        )
        return 1
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "translate":
        return _cmd_translate(args)
    if args.command == "extract":
        return _cmd_stub(NOT_IN_V01_EXTRACT)
    if args.command == "transcribe":
        return _cmd_transcribe(args)
    parser.print_help()
    return 1
