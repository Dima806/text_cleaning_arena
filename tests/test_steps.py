"""Each cleaning step does what it claims."""

from __future__ import annotations

from src.cleaning import steps


def test_lowercase_collapses_casing():
    assert steps.lowercase("Apple US March") == "apple us march"


def test_normalize_whitespace_collapses_runs():
    assert steps.normalize_whitespace("  hello\t\nworld   ") == "hello world"


def test_remove_stopwords_keeps_content_words():
    out = steps.remove_stopwords("this is a very good movie")
    assert "movie" in out
    assert "good" in out
    assert "the" not in out.split()
    assert "is" not in out.split()


def test_strip_punctuation_removes_symbols():
    out = steps.strip_punctuation("let's eat, grandma! right?")
    assert "," not in out
    assert "!" not in out
    assert "?" not in out
    assert "grandma" in out


def test_remove_numbers_drops_digits():
    out = steps.remove_numbers("i bought 3 apples and 25 pears")
    assert not any(ch.isdigit() for ch in out)
    assert "apples" in out


def test_stem_produces_shorter_or_nonword_forms():
    assert steps.stem("running") == "run"
    # Porter stemming over-chops into the non-word "studi".
    assert steps.stem("studies").startswith("studi")


def test_lemmatize_returns_valid_base_forms():
    assert steps.lemmatize("cats") == "cat"
    assert steps.lemmatize("studies") == "study"


def test_stem_and_lemmatize_differ_on_studies():
    # The contrast the project leans on: stemming mangles, lemmatization preserves.
    assert steps.stem("studies") != steps.lemmatize("studies")
