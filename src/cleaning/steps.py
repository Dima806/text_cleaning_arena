"""Individual, toggleable cleaning steps.

Each step is a ``str -> str`` function doing exactly one thing, so the damage of
each can be measured in isolation. Stopword removal is deliberately configured to
delete negations (``not``/``no``/``never``) — that is the destructive behaviour the
project exists to expose, not a bug.

nltk is used for stemming and lemmatization when its data is present; both fall
back to pure-Python approximations so the notebooks still run on a bare Codespace.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections.abc import Callable
from functools import lru_cache

# --- Word lists ------------------------------------------------------------

#: Negation words. Removing any of these can flip the meaning of a sentence.
NEGATIONS: frozenset[str] = frozenset(
    {"no", "not", "nor", "none", "never", "neither", "nothing", "nobody", "nowhere"}
)

#: Curated English stopword list. It intentionally includes the negations above
#: and excludes modal verbs (``would``, ``could``) so the documented failure
#: cases are reproducible without depending on an external download.
STOPWORDS: frozenset[str] = (
    frozenset(
        {
            "a",
            "an",
            "the",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "myself",
            "you",
            "your",
            "yours",
            "he",
            "him",
            "his",
            "she",
            "her",
            "hers",
            "it",
            "its",
            "we",
            "us",
            "our",
            "ours",
            "they",
            "them",
            "their",
            "theirs",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "do",
            "does",
            "did",
            "doing",
            "have",
            "has",
            "had",
            "having",
            "and",
            "or",
            "but",
            "if",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "to",
            "from",
            "in",
            "on",
            "off",
            "over",
            "under",
            "again",
            "then",
            "once",
            "here",
            "there",
            "all",
            "any",
            "both",
            "each",
            "more",
            "most",
            "other",
            "some",
            "such",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
        }
    )
    | NEGATIONS
)


# --- Tokenization helpers --------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z]+")


def _map_words(text: str, fn: Callable[[str], str]) -> str:
    """Apply ``fn`` to each alphabetic run, leaving punctuation and spacing intact."""
    return _WORD_RE.sub(lambda m: fn(m.group()), text)


# --- Steps -----------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """Unicode-normalize (NFKC) and collapse runs of whitespace to single spaces."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def lowercase(text: str) -> str:
    """Lowercase everything, collapsing ``US``/``us`` and ``Apple``/``apple``."""
    return text.lower()


def remove_stopwords(text: str) -> str:
    """Drop whitespace tokens whose bare form is a stopword (negations included)."""
    kept = []
    for tok in text.split():
        bare = tok.strip(string.punctuation).lower()
        if bare and bare in STOPWORDS:
            continue
        kept.append(tok)
    return " ".join(kept)


def strip_punctuation(text: str) -> str:
    """Remove punctuation, erasing structure such as the comma in a direct address."""
    return normalize_whitespace(re.sub(r"[^\w\s]", " ", text))


def remove_numbers(text: str) -> str:
    """Remove digit sequences."""
    return normalize_whitespace(re.sub(r"\d+", " ", text))


def stem(text: str) -> str:
    """Aggressive suffix chopping (Porter) that can produce non-words (``studi``)."""
    stemmer = _get_stemmer()
    fn = stemmer if stemmer is not None else _fallback_stem_word
    return _map_words(text, fn)


def lemmatize(text: str) -> str:
    """Dictionary-based reduction to valid base forms (``studies`` -> ``study``)."""
    lemmatizer = _get_lemmatizer()
    fn = lemmatizer if lemmatizer is not None else _fallback_lemmatize_word
    return _map_words(text, fn)


#: Registry of all steps, used by the analysis module.
STEPS: dict[str, Callable[[str], str]] = {
    "normalize_whitespace": normalize_whitespace,
    "lowercase": lowercase,
    "remove_stopwords": remove_stopwords,
    "strip_punctuation": strip_punctuation,
    "remove_numbers": remove_numbers,
    "stem": stem,
    "lemmatize": lemmatize,
}


# --- nltk with pure-Python fallbacks ---------------------------------------


@lru_cache(maxsize=1)
def _get_stemmer() -> Callable[[str], str] | None:
    try:
        from nltk.stem import PorterStemmer
    except ImportError:
        return None
    return PorterStemmer().stem


@lru_cache(maxsize=1)
def _get_lemmatizer() -> Callable[[str], str] | None:
    try:
        from nltk.stem import WordNetLemmatizer
    except ImportError:
        return None
    lemmatizer = WordNetLemmatizer()
    try:
        lemmatizer.lemmatize("tests")  # probe forces the wordnet data to load
    except LookupError:
        return None
    return lemmatizer.lemmatize


def _fallback_stem_word(word: str) -> str:
    lowered = word.lower()
    for suffix in ("ingly", "edly", "ing", "ied", "ies", "ed", "ly", "es", "s"):
        if lowered.endswith(suffix) and len(lowered) - len(suffix) >= 2:
            return lowered[: -len(suffix)]
    return lowered


def _fallback_lemmatize_word(word: str) -> str:
    lowered = word.lower()
    if lowered.endswith("ies") and len(lowered) > 4:
        return lowered[:-3] + "y"
    if lowered.endswith("es") and len(lowered) > 3:
        return lowered[:-2]
    if lowered.endswith("s") and not lowered.endswith("ss") and len(lowered) > 3:
        return lowered[:-1]
    return lowered
