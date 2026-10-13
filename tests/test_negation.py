"""The asserted showpiece: stopword removal deletes negation and flips meaning.

If someone 'fixes' the stopword list so negations survive, these tests fail loudly
— which is the point. The destructive behaviour is the thing being measured.
"""

from __future__ import annotations

from src.cleaning.pipelines import clean
from src.cleaning.steps import NEGATIONS, STOPWORDS, remove_stopwords
from src.datasets.corpora import load_sentiment
from src.evaluation.failure_cases import find_negation_flips, negations_removed


def test_negations_are_in_the_stopword_list():
    # This is what makes the damage happen.
    assert NEGATIONS <= STOPWORDS
    for word in ("not", "no", "never"):
        assert word in STOPWORDS


def test_not_good_becomes_good():
    out = remove_stopwords("the movie was not good")
    assert "not" not in out.split()
    assert "good" in out.split()


def test_never_recommend_loses_the_negation():
    out = clean("I would never recommend this", "standard")
    assert "never" not in out.split()
    assert "recommend" in out.split()


def test_negations_removed_reports_the_deleted_words():
    removed = negations_removed("This is not good and I would never buy it", "standard")
    assert "not" in removed
    assert "never" in removed


def test_standard_cleaning_flips_negative_sentiment_examples():
    texts, labels = load_sentiment()
    flips = find_negation_flips(texts, labels, pipeline="standard")
    # At least a few negative reviews lose their negation under standard cleaning.
    assert len(flips) >= 3
    assert all(flip.removed for flip in flips)
    # The flips land on negative (label 0) reviews, which is where the harm is.
    assert any(flip.label == 0 for flip in flips)


def test_minimal_cleaning_keeps_negations():
    # The modern recommendation preserves the signal transformers were trained on.
    assert negations_removed("this is not good", "minimal") == set()
    assert negations_removed("this is not good", "none") == set()
