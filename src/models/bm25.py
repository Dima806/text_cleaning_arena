"""BM25: the sparse retrieval workhorse.

A compact, dependency-free Okapi BM25 so retrieval results are deterministic and
run offline on a 2-CPU box. BM25 matches on surface tokens, which is exactly why
stemming — merging ``run``/``runs``/``running`` — tends to help it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 over a fixed document collection."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._idf: dict[str, float] = {}
        self._avg_len: float = 0.0

    def index(self, documents: Sequence[str]) -> BM25Retriever:
        self._docs = [tokenize(doc) for doc in documents]
        n_docs = len(self._docs)
        self._avg_len = sum(len(d) for d in self._docs) / n_docs if n_docs else 0.0
        self._doc_freq = Counter()
        for doc in self._docs:
            self._doc_freq.update(set(doc))
        self._idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in self._doc_freq.items()
        }
        return self

    def scores(self, query: str) -> list[float]:
        """BM25 score of every indexed document for ``query``."""
        query_terms = tokenize(query)
        results: list[float] = []
        for doc in self._docs:
            counts = Counter(doc)
            doc_len = len(doc)
            score = 0.0
            for term in query_terms:
                freq = counts.get(term, 0)
                if freq == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self._avg_len)
                score += idf * freq * (self.k1 + 1) / denom
            results.append(score)
        return results

    def retrieve(self, query: str, k: int = 10) -> list[int]:
        """Return the indices of the top-``k`` documents, best first."""
        scores = self.scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:k]
