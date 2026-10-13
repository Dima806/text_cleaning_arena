"""Named pipelines compose the steps correctly."""

from __future__ import annotations

import pytest

from src.cleaning.pipelines import PIPELINE_NAMES, clean


def test_pipeline_names_in_spectrum_order():
    assert PIPELINE_NAMES == ("none", "minimal", "standard", "aggressive")


def test_none_is_identity():
    text = "The US economy grew, and Apple's stock rose 3%!"
    assert clean(text, "none") == text


def test_minimal_only_normalizes_whitespace():
    assert clean("  Hello   World  ", "minimal") == "Hello World"
    # Casing and punctuation survive minimal cleaning.
    assert clean("Hello, World!", "minimal") == "Hello, World!"


def test_standard_lowercases_and_drops_stopwords():
    out = clean("This is NOT a Good Movie", "standard")
    assert out == out.lower()
    assert "is" not in out.split()


def test_aggressive_stems_and_strips():
    out = clean("The 2 cats were running quickly!", "aggressive")
    assert out == out.lower()
    assert not any(ch.isdigit() for ch in out)
    assert "!" not in out
    assert "run" in out  # "running" -> "run"


def test_unknown_pipeline_raises():
    with pytest.raises(ValueError, match="Unknown pipeline"):
        clean("hello", "turbo")


@pytest.mark.parametrize("pipeline", PIPELINE_NAMES)
def test_every_pipeline_returns_a_string(pipeline: str):
    assert isinstance(clean("A quick Test, with 1 number.", pipeline), str)
