"""Classification evaluation: cross-validated accuracy and macro-F1.

Cross-validation rather than a single split, because the corpora are small enough
that one 70/30 draw is far too noisy to compare pipelines fairly. A fresh model is
fit on every fold via the ``model_factory``, so nothing leaks between folds or
pipelines.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Protocol

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.cleaning.pipelines import clean


class Classifier(Protocol):
    """Minimal fit/predict interface shared by the arena's classifiers."""

    name: str

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> Classifier: ...

    def predict(self, texts: Sequence[str]) -> list[int]: ...


ModelFactory = Callable[[], Classifier]


@dataclass(frozen=True)
class ClassificationResult:
    """Cross-validated scores for one (model, pipeline) cell."""

    model: str
    pipeline: str
    accuracy: float
    macro_f1: float
    n_splits: int


def evaluate_classification(
    model_factory: ModelFactory,
    texts: Sequence[str],
    labels: Sequence[int],
    pipeline: str = "none",
    n_splits: int = 5,
    seed: int = 42,
) -> ClassificationResult:
    """Clean with ``pipeline``, then report mean accuracy / macro-F1 over k folds."""
    cleaned = [clean(text, pipeline) for text in texts]
    y = list(labels)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accuracies: list[float] = []
    macro_f1s: list[float] = []
    model_name = "model"
    for train_idx, test_idx in splitter.split(cleaned, y):
        model = model_factory()
        model_name = model.name
        model.fit([cleaned[i] for i in train_idx], [y[i] for i in train_idx])
        predictions = model.predict([cleaned[i] for i in test_idx])
        truth = [y[i] for i in test_idx]
        accuracies.append(float(accuracy_score(truth, predictions)))
        macro_f1s.append(float(f1_score(truth, predictions, average="macro")))
    return ClassificationResult(
        model=model_name,
        pipeline=pipeline,
        accuracy=fmean(accuracies),
        macro_f1=fmean(macro_f1s),
        n_splits=n_splits,
    )
