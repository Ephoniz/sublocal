from pathlib import Path

from sublocal.backend import EchoBackend
from sublocal.cli import main
from sublocal.pipeline import translate_file
from sublocal.progress import enable_download_progress, status, stderr_tqdm_class


def test_status_goes_to_stderr(capsys) -> None:
    status("Downloading example/repo (~600 MB)")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Downloading example/repo (~600 MB)" in captured.err


def test_cli_echo_progress_does_not_break_stdout(
    sample_srt: Path, tmp_path: Path, capsys
) -> None:
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
    assert captured.out.strip() == str(out)
    assert "Translating 12 cues" in captured.err
    assert "-->" not in captured.err


def test_echo_pipeline_progress_on_stderr(
    sample_srt: Path, tmp_path: Path, capsys
) -> None:
    out = tmp_path / "echo.en.srt"
    translate_file(
        sample_srt,
        to_code="en",
        from_code="es",
        output_path=out,
        backend=EchoBackend(),
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Translating 12 cues (spa_Latn → eng_Latn)" in captured.err


def test_stderr_tqdm_class_updates(capsys) -> None:
    enable_download_progress()
    cls = stderr_tqdm_class()
    bar = cls(total=100, desc="Downloading", unit="B", unit_scale=True)
    bar.update(40)
    bar.update(60)
    bar.close()
    err = capsys.readouterr().err
    assert "Downloading" in err
    # Real tqdm reports a finished 100% bar, not a stuck spinner.
    assert "100%" in err
