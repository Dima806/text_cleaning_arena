"""Named pipelines that bundle the cleaning steps into the choices practitioners make.

- ``none``       raw text, the baseline everyone skips.
- ``minimal``    whitespace/encoding normalization only — the modern recommendation.
- ``standard``   lowercase + stopwords + lemmatize — the common tutorial pipeline.
- ``aggressive`` everything, plus stemming and number removal — the kitchen sink.
"""

from __future__ import annotations

from collections.abc import Callable

from src.cleaning.steps import (
    lemmatize,
    lowercase,
    normalize_whitespace,
    remove_numbers,
    remove_stopwords,
    stem,
    strip_punctuation,
)

#: Ordered step lists for each named pipeline.
PIPELINES: dict[str, list[Callable[[str], str]]] = {
    "none": [],
    "minimal": [normalize_whitespace],
    "standard": [lowercase, remove_stopwords, lemmatize],
    "aggressive": [
        normalize_whitespace,
        lowercase,
        remove_numbers,
        strip_punctuation,
        remove_stopwords,
        stem,
    ],
}

#: Pipeline names in spectrum order, from least to most destructive.
PIPELINE_NAMES: tuple[str, ...] = tuple(PIPELINES)


def clean(text: str, pipeline: str = "none") -> str:
    """Apply a named pipeline to ``text``.

    This is the public entry point the notebooks and the app rely on.
    """
    if pipeline not in PIPELINES:
        msg = f"Unknown pipeline {pipeline!r}; choose from {PIPELINE_NAMES}"
        raise ValueError(msg)
    for step in PIPELINES[pipeline]:
        text = step(text)
    return text
