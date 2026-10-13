"""Detecting the specific, label-flipping damage that cleaning causes.

The headline failure: stopword removal deletes ``not``/``no``/``never``, so a
negative sentence loses its negation and reads positive. This module finds those
cases on real data; ``tests/test_negation.py`` asserts they happen.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from src.cleaning.pipelines import clean
from src.cleaning.steps import NEGATIONS

_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _negations_in(text: str) -> set[str]:
    return {tok for tok in _TOKEN_RE.findall(text.lower()) if tok in NEGATIONS}


def negations_removed(text: str, pipeline: str) -> set[str]:
    """Negation words present in ``text`` but gone after cleaning with ``pipeline``."""
    return _negations_in(text) - _negations_in(clean(text, pipeline))


@dataclass(frozen=True)
class NegationFlip:
    """A sentence whose negation a pipeline deleted."""

    text: str
    cleaned: str
    label: int
    removed: frozenset[str]


def find_negation_flips(
    texts: Sequence[str],
    labels: Sequence[int],
    pipeline: str = "standard",
) -> list[NegationFlip]:
    """Every example containing a negation that ``pipeline`` strips out."""
    flips: list[NegationFlip] = []
    for text, label in zip(texts, labels, strict=True):
        removed = negations_removed(text, pipeline)
        if removed:
            flips.append(
                NegationFlip(
                    text=text,
                    cleaned=clean(text, pipeline),
                    label=label,
                    removed=frozenset(removed),
                )
            )
    return flips
