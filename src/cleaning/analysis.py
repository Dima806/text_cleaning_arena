"""Vocabulary and information-loss analysis for cleaning steps and pipelines.

The historical justification for cleaning is vocabulary reduction; its hidden cost
is information loss. This module quantifies both so every step can be judged on
what it buys and what it destroys.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from src.cleaning.pipelines import PIPELINES
from src.cleaning.steps import NEGATIONS, STEPS

_TOKEN_RE = re.compile(r"[A-Za-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercased alphabetic tokens, for vocabulary counting."""
    return _TOKEN_RE.findall(text.lower())


def vocabulary(texts: Iterable[str]) -> set[str]:
    """The set of distinct lowercased tokens across a corpus."""
    vocab: set[str] = set()
    for text in texts:
        vocab.update(tokenize(text))
    return vocab


def vocab_size(texts: Iterable[str]) -> int:
    """Number of distinct tokens in a corpus."""
    return len(vocabulary(texts))


def _negations_in(texts: Iterable[str]) -> int:
    return sum(tok in NEGATIONS for text in texts for tok in tokenize(text))


@dataclass(frozen=True)
class TransformReport:
    """What a transform did to a corpus."""

    name: str
    vocab_before: int
    vocab_after: int
    tokens_before: int
    tokens_after: int
    negations_before: int
    negations_after: int

    @property
    def vocab_reduction_pct(self) -> float:
        if self.vocab_before == 0:
            return 0.0
        return 100.0 * (self.vocab_before - self.vocab_after) / self.vocab_before

    @property
    def negations_removed(self) -> int:
        return self.negations_before - self.negations_after


def _report(name: str, before: list[str], after: list[str]) -> TransformReport:
    return TransformReport(
        name=name,
        vocab_before=vocab_size(before),
        vocab_after=vocab_size(after),
        tokens_before=sum(len(tokenize(t)) for t in before),
        tokens_after=sum(len(tokenize(t)) for t in after),
        negations_before=_negations_in(before),
        negations_after=_negations_in(after),
    )


def analyze_step(texts: Iterable[str], step: str) -> TransformReport:
    """Measure what a single named step does to a corpus."""
    materialized = list(texts)
    fn: Callable[[str], str] = STEPS[step]
    return _report(step, materialized, [fn(t) for t in materialized])


def analyze_pipeline(texts: Iterable[str], pipeline: str) -> TransformReport:
    """Measure what a named pipeline does to a corpus."""
    from src.cleaning.pipelines import clean

    materialized = list(texts)
    return _report(pipeline, materialized, [clean(t, pipeline) for t in materialized])


def step_table(texts: Iterable[str]) -> list[TransformReport]:
    """One report per cleaning step, cheapest-to-most-destructive by vocab reduction."""
    materialized = list(texts)
    return [analyze_step(materialized, name) for name in STEPS]


def pipeline_table(texts: Iterable[str]) -> list[TransformReport]:
    """One report per named pipeline."""
    materialized = list(texts)
    return [analyze_pipeline(materialized, name) for name in PIPELINES]
