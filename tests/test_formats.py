from pathlib import Path

from sublocal.backend import EchoBackend
from sublocal.formats import dumps, load
from sublocal.pipeline import translate_file


def test_vtt_preserves_timestamps(sample_vtt: Path, tmp_path: Path) -> None:
    out = tmp_path / "sample.en.vtt"
    translate_file(
        sample_vtt,
        to_code="en",
        from_code="es",
        output_path=out,
        backend=EchoBackend(),
    )
    src = load(sample_vtt)
    dst = load(out)
    assert len(dst.cues) == len(src.cues)
    assert [c.timing for c in dst.cues] == [c.timing for c in src.cues]
    dumped = dumps(dst)
    assert dumped.startswith("WEBVTT")
    assert "NOTE sample file" in dumped


def test_ass_preserves_start_end(sample_ass: Path, tmp_path: Path) -> None:
    out = tmp_path / "sample.en.ass"
    translate_file(
        sample_ass,
        to_code="en",
        from_code="es",
        output_path=out,
        backend=EchoBackend(),
    )
    src = load(sample_ass)
    dst = load(out)
    assert len(dst.cues) == 3
    assert [c.timing for c in dst.cues] == [c.timing for c in src.cues]
    text = out.read_text(encoding="utf-8")
    assert "0:00:01.00" in text
    assert "0:00:03.24" in text
    assert "{\\an8}" in text
    assert "[Events]" in text
