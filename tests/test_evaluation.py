"""Evaluation harnesses: metrics, retrieval, classification, and the arena."""

from __future__ import annotations

from src.cleaning.analysis import analyze_pipeline, analyze_step, vocab_size
from src.cleaning.pipelines import PIPELINE_NAMES, clean
from src.datasets.corpora import load_retrieval, load_sentiment
from src.evaluation.classification import evaluate_classification
from src.evaluation.comparison import evaluate_cleaning
from src.evaluation.retrieval import evaluate_retrieval, ndcg_at_k, recall_at_k
from src.models.bm25 import BM25Retriever
from src.models.bow import BowClassifier

# --- Metrics ---------------------------------------------------------------


def test_ndcg_perfect_and_discounted():
    assert ndcg_at_k([0, 1, 2], {0}, 3) == 1.0
    assert ndcg_at_k([2, 1, 0], {0}, 3) == 0.5
    assert ndcg_at_k([1, 2], {0}, 2) == 0.0


def test_recall_at_k():
    assert recall_at_k([0, 1, 2], {0, 1}, 3) == 1.0
    assert recall_at_k([2, 1, 0], {0}, 2) == 0.0
    assert recall_at_k([0], set(), 1) == 0.0


# --- BM25 retrieval --------------------------------------------------------


def test_bm25_with_stemming_ranks_relevant_documents():
    # The corpus is stemming-sensitive: relevant docs use different word forms than
    # the query, so BM25 only finds them once stemming merges the forms.
    corpus = load_retrieval()
    documents = [clean(doc, "aggressive") for doc in corpus.documents]
    retriever = BM25Retriever().index(documents)
    top = retriever.retrieve(clean(corpus.queries[0], "aggressive"), k=3)
    assert set(top) & corpus.qrels[0]


def test_evaluate_retrieval_in_range():
    corpus = load_retrieval()
    result = evaluate_retrieval(BM25Retriever(), corpus, clean_fn=lambda text: text, k=5)
    assert 0.0 <= result.ndcg <= 1.0
    assert 0.0 <= result.recall <= 1.0


def test_stemming_helps_bm25_retrieval():
    # The retrieval thesis: aggressive normalization lifts a lexical retriever by
    # merging query and document word forms.
    corpus = load_retrieval()
    raw = evaluate_retrieval(BM25Retriever(), corpus, clean_fn=lambda text: text, k=5)
    stemmed = evaluate_retrieval(
        BM25Retriever(), corpus, clean_fn=lambda text: clean(text, "aggressive"), k=5
    )
    assert stemmed.ndcg > raw.ndcg


# --- Classification --------------------------------------------------------


def test_evaluate_classification_learns_sentiment():
    texts, labels = load_sentiment()
    result = evaluate_classification(BowClassifier, texts, labels, pipeline="none")
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    # Cross-validated, the classic model clears chance comfortably on this corpus.
    assert result.accuracy > 0.6
    assert result.model == "tfidf+logreg"


def test_evaluate_cleaning_scoreboard_covers_every_pipeline():
    texts, labels = load_sentiment()
    frame = evaluate_cleaning(texts, labels, BowClassifier)
    assert len(frame) == len(PIPELINE_NAMES)
    assert set(frame["pipeline"]) == set(PIPELINE_NAMES)
    assert {"model", "pipeline", "accuracy", "macro_f1", "n_splits"} <= set(frame.columns)


# --- Vocabulary / information-loss analysis --------------------------------


def test_stopword_removal_shrinks_vocabulary():
    texts, _ = load_sentiment()
    report = analyze_step(texts, "remove_stopwords")
    assert report.vocab_after < report.vocab_before
    assert report.vocab_reduction_pct > 0


def test_standard_pipeline_removes_negations():
    texts, _ = load_sentiment()
    report = analyze_pipeline(texts, "standard")
    assert report.negations_removed > 0


def test_none_pipeline_preserves_vocabulary():
    texts, _ = load_sentiment()
    report = analyze_pipeline(texts, "none")
    assert report.vocab_after == report.vocab_before == vocab_size(texts)
