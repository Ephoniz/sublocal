"""Offline source-language detection from cue text."""

from __future__ import annotations

from langdetect import DetectorFactory, LangDetectException, detect

# Deterministic results for the same subtitle file.
DetectorFactory.seed = 0


class DetectionError(RuntimeError):
    pass


def detect_iso639(texts: list[str]) -> str:
    """Detect ISO 639-1 from concatenated cue text.

    Short cues are unreliable on their own, so we join many of them.
    """
    blob = " ".join(t.strip() for t in texts if t.strip())
    if len(blob) < 8:
        raise DetectionError(
            "Not enough text to detect the source language; pass --from."
        )
    try:
        return detect(blob)
    except LangDetectException as exc:
        raise DetectionError(
            "Could not detect the source language; pass --from."
        ) from exc
