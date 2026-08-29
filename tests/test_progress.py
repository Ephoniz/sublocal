from pathlib import Path

import pytest

from sublocal.backend import EchoBackend
from sublocal.cli import main
from sublocal.pipeline import translate_file
from sublocal.progress import (
    BatchCounter,
    enable_download_progress,
    format_eta,
    status,
    stderr_tqdm_class,
)


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


def test_batch_counter_newlines(capsys) -> None:
    counter = BatchCounter(599)
    counter.update(32)
    counter.update(32)
    counter.update(535)
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln]
    assert lines[0].startswith("32/599 cues (5%)")
    assert lines[1].startswith("64/599 cues (10%)")
    assert lines[-1] == "599/599 cues (100%)"
    # New lines, not a single carriage-return bar.
    assert "\r" not in err


def test_format_eta() -> None:
    assert format_eta(9) == "9s"
    assert format_eta(60) == "1m"
    assert format_eta(90) == "1m30s"
    assert format_eta(3600) == "1h00m"


def test_translating_line_after_prepare(sample_srt: Path, tmp_path: Path, capsys) -> None:
    events: list[str] = []

    class BoomThenOk:
        def prepare(self) -> None:
            events.append("prepare")

        def translate(self, texts, src_flores, tgt_flores):
            events.append("translate")
            return list(texts)

    translate_file(
        sample_srt,
        to_code="en",
        from_code="es",
        output_path=tmp_path / "ok.srt",
        backend=BoomThenOk(),
    )
    err = capsys.readouterr().err
    assert events == ["prepare", "translate"]
    assert "Translating 12 cues" in err


def test_no_translating_line_if_prepare_fails(
    sample_srt: Path, tmp_path: Path, capsys
) -> None:
    class GuardFail:
        def prepare(self) -> None:
            raise RuntimeError("anaconda guard")

        def translate(self, texts, src_flores, tgt_flores):
            raise AssertionError("translate must not run")

    with pytest.raises(RuntimeError, match="anaconda guard"):
        translate_file(
            sample_srt,
            to_code="en",
            from_code="es",
            output_path=tmp_path / "nope.srt",
            backend=GuardFail(),
        )
    assert "Translating" not in capsys.readouterr().err


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
