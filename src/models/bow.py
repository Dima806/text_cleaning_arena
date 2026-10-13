"""TF-IDF + logistic regression: the classic sparse bag-of-words classifier.

This is the model the cleaning ritual was designed for, where shrinking the sparse
vocabulary genuinely helps. Having it in the same arena as the transformer is the
point: the same cleaning step moves the two in opposite directions.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class BowClassifier:
    """TF-IDF features into a logistic-regression head."""

    name = "tfidf+logreg"

    def __init__(self, seed: int = 42, max_features: int = 20000) -> None:
        # A realistic classic TF-IDF setup: the vectorizer lowercases (casing is the
        # transformer's story, not bag-of-words'), while the pipeline's stopword
        # removal, lemmatization, and stemming still shape the feature space.
        self.pipeline: Pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=max_features)),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> BowClassifier:
        self.pipeline.fit(list(texts), list(labels))
        return self

    def predict(self, texts: Sequence[str]) -> list[int]:
        return [int(p) for p in self.pipeline.predict(list(texts))]
