from pathlib import Path

from sublocal.detect import detect_iso639
from sublocal.formats import load
from sublocal.pipeline import translate_file
from sublocal.backend import EchoBackend


def test_detect_spanish_fixture(sample_srt: Path) -> None:
    doc = load(sample_srt)
    assert detect_iso639([c.text for c in doc.cues]) == "es"


def test_translate_autodetect(sample_srt: Path, tmp_path: Path) -> None:
    out = tmp_path / "auto.en.srt"
    translate_file(sample_srt, to_code="en", output_path=out, backend=EchoBackend())
    assert load(out).cues
