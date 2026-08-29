from pathlib import Path

from sublocal.cli import NOT_IN_V01_EXTRACT, main
from sublocal.formats import load


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
