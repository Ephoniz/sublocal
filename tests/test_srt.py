from pathlib import Path

from sublocal.backend import EchoBackend
from sublocal.formats import dumps, load
from sublocal.formats.srt import parse_srt
from sublocal.pipeline import translate_file


def _timings(doc) -> list[str]:
    return [c.timing for c in doc.cues]


def test_fixture_has_enough_cues(sample_srt: Path) -> None:
    doc = load(sample_srt)
    assert len(doc.cues) >= 10
    texts = [c.text for c in doc.cues]
    assert any(t.startswith("(") and t.endswith(")") for t in texts)
    assert any("\n" in t for t in texts)


def test_srt_roundtrip_preserves_timestamps(sample_srt: Path) -> None:
    original = load(sample_srt)
    again = parse_srt(dumps(original))
    assert _timings(again) == _timings(original)
    assert [c.index for c in again.cues] == [c.index for c in original.cues]
    assert [c.text for c in again.cues] == [c.text for c in original.cues]


def test_translate_preserves_cue_count_and_timestamps(
    sample_srt: Path, tmp_path: Path
) -> None:
    out = tmp_path / "sample.es.en.srt"
    translate_file(
        sample_srt,
        to_code="en",
        from_code="es",
        output_path=out,
        backend=EchoBackend(),
    )
    src = load(sample_srt)
    dst = load(out)
    assert len(dst.cues) == len(src.cues)
    assert _timings(dst) == _timings(src)
    assert [c.index for c in dst.cues] == [c.index for c in src.cues]


def test_translate_changes_only_text(
    sample_srt: Path, tmp_path: Path
) -> None:
    class Upper:
        def translate(self, texts, src_flores, tgt_flores):
            assert src_flores == "spa_Latn"
            assert tgt_flores == "eng_Latn"
            for text in texts:
                assert "-->" not in text
            return [t.upper() for t in texts]

    out = tmp_path / "upper.srt"
    translate_file(
        sample_srt,
        to_code="en",
        from_code="es",
        output_path=out,
        backend=Upper(),
    )
    src = load(sample_srt)
    dst = load(out)
    assert _timings(dst) == _timings(src)
    assert [c.text for c in dst.cues] == [c.text.upper() for c in src.cues]


def test_batch_600_cues(tmp_path: Path) -> None:
    lines = []
    for i in range(1, 601):
        start_s = i
        end_s = i + 1
        lines.append(f"{i}")
        lines.append(
            f"00:{start_s // 60:02d}:{start_s % 60:02d},000 --> "
            f"00:{end_s // 60:02d}:{end_s % 60:02d},000"
        )
        lines.append(f"Cue number {i}")
        lines.append("")
    src = tmp_path / "big.srt"
    src.write_text("\n".join(lines), encoding="utf-8")

    class ChunkRecorder:
        def __init__(self) -> None:
            self.sizes: list[int] = []

        def translate(self, texts, src_flores, tgt_flores):
            # Mirror NllbBackend chunking without loading weights.
            batch_size = 32
            out = [""] * len(texts)
            nonempty = [(i, t) for i, t in enumerate(texts) if t.strip()]
            for start in range(0, len(nonempty), batch_size):
                chunk = nonempty[start : start + batch_size]
                self.sizes.append(len(chunk))
                for orig_i, text in chunk:
                    out[orig_i] = text
            return out

    recorder = ChunkRecorder()
    out = tmp_path / "big.en.srt"
    translate_file(src, to_code="en", from_code="en", output_path=out, backend=recorder)
    dst = load(out)
    assert len(dst.cues) == 600
    assert [c.timing for c in dst.cues] == [c.timing for c in load(src).cues]
    assert recorder.sizes[0] == 32
    assert sum(recorder.sizes) == 600
