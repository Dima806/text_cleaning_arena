"""Retrieval evaluation: nDCG@k and recall@k over ranked results."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from src.datasets.corpora import RetrievalCorpus


class Retriever(Protocol):
    """Index-then-retrieve interface shared by the arena's retrievers."""

    name: str

    def index(self, documents: Sequence[str]) -> Retriever: ...

    def retrieve(self, query: str, k: int) -> list[int]: ...


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant documents found in the top ``k``."""
    if not relevant:
        return 0.0
    hits = sum(1 for doc in ranked[:k] if doc in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Normalized discounted cumulative gain with binary relevance."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 2) for rank, doc in enumerate(ranked[:k]) if doc in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


@dataclass(frozen=True)
class RetrievalResult:
    """Mean scores for one (retriever, pipeline) cell."""

    model: str
    pipeline: str
    ndcg: float
    recall: float
    k: int


def evaluate_retrieval(
    retriever: Retriever,
    corpus: RetrievalCorpus,
    clean_fn: Callable[[str], str],
    k: int = 10,
) -> RetrievalResult:
    """Clean queries and documents with ``clean_fn``, then score mean nDCG/recall."""
    retriever.index([clean_fn(doc) for doc in corpus.documents])
    ndcgs: list[float] = []
    recalls: list[float] = []
    for q_idx, query in enumerate(corpus.queries):
        relevant = corpus.qrels.get(q_idx, set())
        ranked = retriever.retrieve(clean_fn(query), k)
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
        recalls.append(recall_at_k(ranked, relevant, k))
    n = len(corpus.queries)
    return RetrievalResult(
        model=retriever.name,
        pipeline="custom",
        ndcg=sum(ndcgs) / n if n else 0.0,
        recall=sum(recalls) / n if n else 0.0,
        k=k,
    )
