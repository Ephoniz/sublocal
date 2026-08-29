from __future__ import annotations

import argparse
import os
import sys

from sublocal import __version__
from sublocal.detect import DetectionError
from sublocal.formats import UnsupportedFormatError
from sublocal.languages import UnknownLanguageError
from sublocal.pipeline import backend_from_name, translate_file

NOT_IN_V01_EXTRACT = (
    "not in v0.1: extract existing subtitle tracks from video "
    "(ffmpeg/mkvextract). Soft subs only."
)
NOT_IN_V01_TRANSCRIBE = (
    "not in v0.1: transcribe with faster-whisper, then translate "
    "with the same pipeline."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sublocal",
        description=(
            "Local-only subtitle translation. No cloud APIs, no API keys, "
            "no telemetry."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"sublocal {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser(
        "translate",
        help="Translate an existing subtitle file; keep timestamps.",
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
        help="Source language. Detected from cue text if omitted.",
    )
    tr.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output path (default: input.<to>.<ext>)",
    )
    tr.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: cuda if available, else cpu)",
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
    extract.add_argument("args", nargs="*", help=argparse.SUPPRESS)

    transcribe = sub.add_parser(
        "transcribe",
        help="Transcribe audio then translate (not in v0.1).",
    )
    transcribe.add_argument("args", nargs="*", help=argparse.SUPPRESS)

    return parser


def _cmd_translate(args: argparse.Namespace) -> int:
    try:
        backend = backend_from_name(args.backend, args.device, args.batch_size)
        out = translate_file(
            args.input,
            to_code=args.to,
            from_code=args.from_lang,
            output_path=args.out,
            backend=backend,
        )
    except (
        FileNotFoundError,
        ValueError,
        UnknownLanguageError,
        DetectionError,
        UnsupportedFormatError,
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "translate":
        return _cmd_translate(args)
    if args.command == "extract":
        return _cmd_stub(NOT_IN_V01_EXTRACT)
    if args.command == "transcribe":
        return _cmd_stub(NOT_IN_V01_TRANSCRIBE)
    parser.print_help()
    return 1
