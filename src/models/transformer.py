"""Transformer sentence embeddings: the modern model cleaning tends to hurt.

Uses ``all-MiniLM-L6-v2`` on CPU. ``sentence-transformers`` and ``torch`` are
imported lazily so that importing this module (and the rest of ``src``) stays cheap
and the test suite never has to download a model. Embeddings are cached to
``data/emb/`` keyed by (model, pipeline, corpus) so notebooks rerun in seconds.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from src.cleaning.pipelines import PIPELINE_NAMES, clean
from src.config import EMB_DIR, get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def _cache_key(model_name: str, pipeline: str, texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode())
    digest.update(pipeline.encode())
    for text in texts:
        digest.update(text.encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


class TransformerEmbedder:
    """Lazy, cached wrapper around a sentence-transformer encoder."""

    name = "transformer"

    def __init__(self, model_name: str | None = None, revision: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.model.name
        self.revision = revision if revision is not None else settings.model.revision
        self.batch_size = settings.model.batch_size
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name, revision=self.revision, device="cpu"
            )
        return self._model

    def encode(
        self,
        texts: Sequence[str],
        pipeline: str = "none",
        use_cache: bool = True,
    ) -> np.ndarray:
        """Clean ``texts`` with ``pipeline``, then embed, caching the result."""
        cleaned = [clean(text, pipeline) for text in texts]
        key = _cache_key(self.model_name, pipeline, cleaned)
        cache_path = EMB_DIR / f"{key}.npy"
        if use_cache and cache_path.exists():
            return np.load(cache_path)
        model = self._load()
        embeddings = np.asarray(
            model.encode(cleaned, batch_size=self.batch_size, show_progress_bar=False)
        )
        if use_cache:
            EMB_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
        return embeddings


def _encode_all() -> None:
    """Pre-compute embeddings for every pipeline on every corpus (``make embed``)."""
    from src.datasets.corpora import load_retrieval, load_sentiment, load_topic

    embedder = TransformerEmbedder()
    sentiment_texts, _ = load_sentiment()
    topic_texts, _ = load_topic()
    retrieval = load_retrieval()
    corpora = {
        "sentiment": sentiment_texts,
        "topic": topic_texts,
        "retrieval-docs": retrieval.documents,
        "retrieval-queries": retrieval.queries,
    }
    for pipeline in PIPELINE_NAMES:
        for corpus_name, texts in corpora.items():
            embedder.encode(texts, pipeline=pipeline)
            print(f"  cached {corpus_name} x {pipeline}")
    print(f"✅ Embeddings cached in {EMB_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer embedding cache")
    parser.add_argument("--encode-all", action="store_true", help="Encode every pipeline")
    args = parser.parse_args()
    if args.encode_all:
        _encode_all()
    else:
        parser.print_help()
