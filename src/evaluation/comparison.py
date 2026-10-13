"""The arena harness.

``evaluate_cleaning`` is the reader-facing utility: give it texts, labels, and a way
to build a model, and it runs every pipeline and reports which cleaning wins — so you
measure rather than assume. ``classification_arena`` stacks several models into one
scoreboard.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.cleaning.pipelines import PIPELINE_NAMES
from src.evaluation.classification import ModelFactory, evaluate_classification


def evaluate_cleaning(
    texts: Sequence[str],
    labels: Sequence[int],
    model_factory: ModelFactory,
    pipelines: Sequence[str] = PIPELINE_NAMES,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Run every pipeline for one model and return a scoreboard sorted by macro-F1."""
    rows = []
    for pipeline in pipelines:
        result = evaluate_classification(
            model_factory, texts, labels, pipeline=pipeline, n_splits=n_splits, seed=seed
        )
        rows.append(
            {
                "model": result.model,
                "pipeline": result.pipeline,
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "n_splits": result.n_splits,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def classification_arena(
    model_factories: dict[str, ModelFactory],
    texts: Sequence[str],
    labels: Sequence[int],
    pipelines: Sequence[str] = PIPELINE_NAMES,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Stack ``evaluate_cleaning`` across several models into one long-form table."""
    frames = [
        evaluate_cleaning(texts, labels, factory, pipelines, n_splits, seed)
        for factory in model_factories.values()
    ]
    return pd.concat(frames, ignore_index=True)
