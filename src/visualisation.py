"""Plotting helpers for the notebooks and the app.

Thin functions that turn the evaluation DataFrames and analysis reports into
figures. Kept free of any app/notebook state so they are easy to reuse and test.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.cleaning.analysis import TransformReport
from src.config import FIGURES_DIR, RESULTS_DIR


def plot_pipeline_scores(frame: pd.DataFrame, metric: str = "macro_f1") -> Figure:
    """Grouped bar chart of a metric per pipeline, one group of bars per model."""
    pivot = frame.pivot_table(index="pipeline", columns="model", values=metric)
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot.bar(ax=ax)
    ax.set_ylabel(metric)
    ax.set_xlabel("cleaning pipeline")
    ax.set_title(f"{metric} by pipeline and model")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def plot_vocab_reduction(reports: Sequence[TransformReport]) -> Figure:
    """Bar chart of the vocabulary reduction each step or pipeline buys."""
    names = [r.name for r in reports]
    reductions = [r.vocab_reduction_pct for r in reports]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, reductions, color="#4c72b0")
    ax.set_ylabel("vocabulary reduction (%)")
    ax.set_title("What each step removes from the vocabulary")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, name: str) -> None:
    """Save a figure under ``outputs/figures/`` as a PNG."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")


def save_json(obj: Any, name: str) -> None:
    """Save textual/numerical results under ``outputs/data/`` as JSON.

    A pandas DataFrame is serialized as a list of row records; everything else is
    dumped as-is, so notebooks can persist every table, number, and string they show.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = obj.to_dict(orient="records") if isinstance(obj, pd.DataFrame) else obj
    (RESULTS_DIR / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
