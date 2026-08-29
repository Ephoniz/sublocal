"""Optional live check against cached NLLB weights. Skipped if not downloaded."""

from __future__ import annotations

from pathlib import Path

import pytest

from sublocal.backend import DEFAULT_MODEL_REPO, NllbBackend
from sublocal.cache import hf_cache_dir
from sublocal.formats import load
from sublocal.pipeline import translate_file


def _nllb_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=DEFAULT_MODEL_REPO,
            cache_dir=str(hf_cache_dir()),
            local_files_only=True,
        )
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _nllb_cached(), reason="NLLB CTranslate2 weights not cached")
def test_nllb_translate_preserves_timestamps(
    sample_srt: Path, tmp_path: Path
) -> None:
    out = tmp_path / "live.en.srt"
    translate_file(
        sample_srt,
        to_code="en",
        from_code="es",
        output_path=out,
        backend=NllbBackend(device="cpu", batch_size=8),
    )
    src = load(sample_srt)
    dst = load(out)
    assert len(dst.cues) == len(src.cues)
    assert [c.timing for c in dst.cues] == [c.timing for c in src.cues]
    assert [c.index for c in dst.cues] == [c.index for c in src.cues]
    # Translation should change at least some Spanish cue text.
    assert any(a.text != b.text for a, b in zip(src.cues, dst.cues))
