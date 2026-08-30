from pathlib import Path

from sublocal.cli import NOT_IN_V01_EXTRACT, build_parser, main
from sublocal.formats import load

DRAMA = Path(__file__).resolve().parents[1] / "examples" / "drama.yml"


def test_cli_translate_echo(
    sample_srt: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("SUBLOCAL_BACKEND", "echo")
    out = tmp_path / "cli.en.srt"
    rc = main(
        [
            "translate",
            str(sample_srt),
            "--to",
            "en",
            "--from",
            "es",
            "--out",
            str(out),
            "--backend",
            "echo",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    printed = captured.out.strip()
    assert printed == str(out)
    assert "Translating 12 cues" in captured.err
    # Progress belongs on stderr; do not dump cue text.
    assert "Hola, bienvenidos" not in captured.err
    assert "Hola, bienvenidos" not in captured.out
    src = load(sample_srt)
    dst = load(out)
    assert len(dst.cues) == len(src.cues)
    assert [c.timing for c in dst.cues] == [c.timing for c in src.cues]


def test_cli_default_out_name(
    sample_srt: Path, tmp_path: Path, monkeypatch
) -> None:
    copied = tmp_path / "input.srt"
    copied.write_text(sample_srt.read_text(encoding="utf-8"), encoding="utf-8")
    rc = main(
        ["translate", str(copied), "--to", "en", "--from", "es", "--backend", "echo"]
    )
    assert rc == 0
    assert (tmp_path / "input.en.srt").is_file()


def test_extract_stub(capsys) -> None:
    rc = main(["extract", "movie.mkv"])
    assert rc == 2
    assert capsys.readouterr().out.strip() == NOT_IN_V01_EXTRACT


def test_cli_translate_model_default_and_small() -> None:
    parser = build_parser()
    default = parser.parse_args(["translate", "in.srt", "--to", "es"])
    assert default.model is None
    assert default.backend == "gemmax"
    q6 = parser.parse_args(["translate", "in.srt", "--to", "es", "--model", "q6"])
    assert q6.model == "q6"
    small = parser.parse_args(["translate", "in.srt", "--to", "es", "--model", "small"])
    assert small.model == "small"


def test_cli_passes_model_small_to_backend(
    sample_srt: Path, tmp_path: Path, monkeypatch
) -> None:
    from sublocal.backend import EchoBackend

    seen: dict[str, str | None] = {}

    def fake(name, device, batch_size, model=None, gguf=None, name_hint=False):
        seen["model"] = model
        return EchoBackend()

    monkeypatch.setattr("sublocal.cli.backend_from_name", fake)
    rc = main(
        [
            "translate",
            str(sample_srt),
            "--to",
            "en",
            "--from",
            "es",
            "--model",
            "small",
            "--out",
            str(tmp_path / "out.srt"),
        ]
    )
    assert rc == 0
    assert seen["model"] == "small"


def test_cli_translate_glossary_echo_restores_latin(
    tmp_path: Path, capsys
) -> None:
    src = tmp_path / "drama.srt"
    src.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n野崎とドラムとバンコク\n",
        encoding="utf-8",
    )
    out = tmp_path / "drama.es.srt"
    rc = main(
        [
            "translate",
            str(src),
            "--to",
            "es",
            "--from",
            "ja",
            "--glossary",
            str(DRAMA),
            "--out",
            str(out),
            "--backend",
            "echo",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == str(out)
    doc = load(out)
    text = doc.cues[0].text
    assert "Nozaki" in text
    assert "野崎" not in text
    assert "ドラム" in text or "Drum" in text
    assert "xx0xx" not in text


def test_missing_file(tmp_path: Path) -> None:
    rc = main(
        [
            "translate",
            str(tmp_path / "nope.srt"),
            "--to",
            "en",
            "--backend",
            "echo",
        ]
    )
    assert rc == 1
