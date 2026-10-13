"""Frozen, in-repo text corpora.

Everything here is committed as Python data so nothing depends on a live download
and the notebooks are reproducible on a fresh Codespace. ``python -m
src.datasets.corpora`` caches the corpora to ``data/corpora/`` as JSON for
inspection; the loaders themselves return the in-memory frozen copies.

The sets are deliberately small (tens to low hundreds of documents) and chosen so
the cleaning effects are measurable on a 2-CPU box: sentiment (where negation and
casing carry the label), topic (where content words dominate), and a small
retrieval set (where stemming's split effect on BM25 vs dense retrieval appears).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

from src.config import CORPORA_DIR

# --- Sentiment -------------------------------------------------------------

# The sentiment set is generated from balanced pools with a fixed seed (so it is
# frozen and reproducible). Subjects are drawn uniformly for BOTH classes, so no
# subject noun correlates with the label — only the sentiment words do, which is
# the signal a classifier should learn. Negations live only in the negatives, and
# several negatives negate a positive word ("not great"), so stopword removal
# deletes the negation and flips the sentence to read positive.

_SUBJECTS: tuple[str, ...] = (
    "film",
    "book",
    "restaurant",
    "hotel",
    "app",
    "concert",
    "course",
    "phone",
    "laptop",
    "cafe",
    "novel",
    "game",
    "meal",
    "headphones",
    "museum",
    "show",
    "service",
    "tour",
    "gadget",
    "camera",
)

_POSITIVE_WORDS: tuple[str, ...] = (
    "great",
    "excellent",
    "wonderful",
    "fantastic",
    "brilliant",
    "superb",
    "delightful",
    "amazing",
    "impressive",
    "beautiful",
    "charming",
    "enjoyable",
    "outstanding",
    "lovely",
    "stellar",
    "gorgeous",
)

_NEGATIVE_WORDS: tuple[str, ...] = (
    "terrible",
    "awful",
    "disappointing",
    "boring",
    "dull",
    "dreadful",
    "poor",
    "mediocre",
    "forgettable",
    "clumsy",
    "frustrating",
    "bland",
    "horrible",
    "tedious",
    "useless",
    "dreary",
)

_POSITIVE_TEMPLATES: tuple[str, ...] = (
    "The {subject} was {a} and genuinely {b}.",
    "I loved the {subject}, it was absolutely {a}.",
    "A {a} {subject} that I would happily recommend.",
    "What a {a} {subject}, {b} from start to finish.",
    "The {subject} was {a}; easily the best I have seen.",
)

_NEGATIVE_TEMPLATES: tuple[str, ...] = (
    "The {subject} was {a} and painfully {b}.",
    "I did not enjoy the {subject}, it was {a}.",
    "A {a} {subject} that I would never recommend.",
    "The {subject} was not {p} at all, just {a}.",
    "What a {a} {subject}, never once {p}.",
)


def _generate(
    templates: tuple[str, ...],
    words: tuple[str, ...],
    rng: random.Random,
    per_class: int,
    seen: set[str],
) -> list[str]:
    out: list[str] = []
    while len(out) < per_class:
        template = rng.choice(templates)
        sentence = template.format(
            subject=rng.choice(_SUBJECTS),
            a=rng.choice(words),
            b=rng.choice(words),
            p=rng.choice(_POSITIVE_WORDS),
        )
        if sentence not in seen:
            seen.add(sentence)
            out.append(sentence)
    return out


def _generate_sentiment(seed: int = 42, per_class: int = 40) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    positives = _generate(_POSITIVE_TEMPLATES, _POSITIVE_WORDS, rng, per_class, seen)
    negatives = _generate(_NEGATIVE_TEMPLATES, _NEGATIVE_WORDS, rng, per_class, seen)
    return positives, negatives


_POSITIVE, _NEGATIVE = _generate_sentiment()


def load_sentiment() -> tuple[list[str], list[int]]:
    """Binary sentiment: ``1`` positive, ``0`` negative. Negation-heavy by design."""
    texts = [*_POSITIVE, *_NEGATIVE]
    labels = [1] * len(_POSITIVE) + [0] * len(_NEGATIVE)
    return texts, labels


# --- Topic -----------------------------------------------------------------

TOPIC_LABELS: tuple[str, ...] = ("technology", "sports", "cooking")

_TOPIC: tuple[tuple[str, int], ...] = (
    ("The new smartphone ships with a faster chip and a brighter screen.", 0),
    ("Researchers released an open source library for training neural networks.", 0),
    ("The laptop battery now lasts twice as long after the firmware update.", 0),
    ("Engineers deployed the server cluster to handle the extra web traffic.", 0),
    ("The startup raised funding to build its cloud database platform.", 0),
    ("A security patch fixed the bug in the operating system kernel.", 0),
    ("The graphics card renders the game at a much higher frame rate.", 0),
    ("Developers debated whether the API should return JSON or XML.", 0),
    ("The compiler flagged a type error deep inside the module.", 0),
    ("Cloud costs dropped after the team optimized the database queries.", 0),
    ("The robot used a camera and sensors to map the warehouse floor.", 0),
    ("A new framework makes building web apps faster for developers.", 0),
    ("The striker scored twice in the second half to win the match.", 1),
    ("Their team clinched the championship after a tense penalty shootout.", 1),
    ("The marathon runner set a new personal best despite the heat.", 1),
    ("The coach substituted the goalkeeper before the final whistle.", 1),
    ("Fans cheered as the home side rallied from two goals down.", 1),
    ("The tennis player saved three match points before winning the set.", 1),
    ("The cyclist attacked on the climb and held the lead to the finish.", 1),
    ("The rookie pitcher struck out the side in the ninth inning.", 1),
    ("The boxer landed a jab and dodged the counter in the final round.", 1),
    ("Spectators roared as the sprinter edged ahead at the finish line.", 1),
    ("The defender blocked the shot and cleared the ball from the goal.", 1),
    ("The swimmer broke the record in the final lap of the relay.", 1),
    ("Simmer the onions in butter until they turn soft and golden.", 2),
    ("Whisk the eggs and sugar together before folding in the flour.", 2),
    ("Roast the vegetables with olive oil, salt, and fresh rosemary.", 2),
    ("Let the dough rise for an hour, then knead it on a floured board.", 2),
    ("Season the soup with a pinch of cumin and a squeeze of lime.", 2),
    ("Grill the salmon skin side down until the flesh flakes easily.", 2),
    ("Blend the tomatoes, garlic, and basil into a smooth sauce.", 2),
    ("Caramelize the sugar slowly, then stir in the warm cream.", 2),
    ("Marinate the chicken in yogurt and spices before you grill it.", 2),
    ("Fold the whipped cream gently into the chilled chocolate mousse.", 2),
    ("Toast the spices in a dry pan to release their aroma.", 2),
    ("Simmer the stock for hours, then strain it through a fine sieve.", 2),
)


def load_topic() -> tuple[list[str], list[int]]:
    """Topic classification across technology / sports / cooking (labels 0/1/2)."""
    texts = [text for text, _ in _TOPIC]
    labels = [label for _, label in _TOPIC]
    return texts, labels


# --- Retrieval -------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalCorpus:
    """A tiny retrieval set: queries, documents, and relevance judgments."""

    queries: list[str]
    documents: list[str]
    #: Maps a query index to the set of relevant document indices.
    qrels: dict[int, set[int]]


# Stemming-sensitive by design. Each query uses an inflected form; the relevant
# documents use OTHER inflections of the same root (so raw BM25, matching on
# surface tokens, misses them) while a distractor contains the exact query word
# but is irrelevant. Stemming merges the word forms and rescues BM25; a dense
# encoder already knows the forms are related, so it needs no such help.
_DOCUMENTS: tuple[str, ...] = (
    "A dedicated runner runs a full marathon every single spring.",  # 0  q0 rel
    "Her marathon pace improved after she chose to run at dawn.",  # 1  q0 rel
    "I keep running late for the morning bus to the office.",  # 2  distractor
    "The baker baked a crusty sourdough loaf of bread overnight.",  # 3  q1 rel
    "A simple recipe to bake fluffy flatbread at home.",  # 4  q1 rel
    "He was baking in the summer heat on the crowded beach.",  # 5  distractor
    "A patient investor who invested early saw her stock grow.",  # 6  q2 rel
    "To invest wisely, diversify and hold each stock for years.",  # 7  q2 rel
    "Investing your energy into office gossip is a total waste.",  # 8  distractor
    "The painter painted a striking portrait in bold oils.",  # 9  q3 rel
    "She loves to paint quiet landscapes and the odd portrait.",  # 10 q3 rel
    "The painting on the wall was crooked and badly lit.",  # 11 distractor
    "The racer raced his bike down the steep mountain trail.",  # 12 q4 rel
    "A fast bike and a smooth road make for a great race.",  # 13 q4 rel
    "Thoughts kept racing through her mind late at night.",  # 14 distractor
)

_QUERIES: tuple[str, ...] = (
    "running marathons",
    "baking breads",
    "investing stocks",
    "painting portraits",
    "racing bikes",
)

_QRELS: dict[int, set[int]] = {
    0: {0, 1},
    1: {3, 4},
    2: {6, 7},
    3: {9, 10},
    4: {12, 13},
}


def load_retrieval() -> RetrievalCorpus:
    """Frozen retrieval corpus with stemming-sensitive word forms (run/runs/running)."""
    return RetrievalCorpus(
        queries=list(_QUERIES),
        documents=list(_DOCUMENTS),
        qrels={q: set(docs) for q, docs in _QRELS.items()},
    )


# --- Adversarial showcase --------------------------------------------------


@dataclass(frozen=True)
class AdversarialExample:
    """A hand-built sentence that cleaning visibly damages."""

    text: str
    label: int
    phenomenon: str
    note: str


def load_adversarial() -> list[AdversarialExample]:
    """Sentences where a cleaning step changes the meaning, not just the form."""
    return [
        AdversarialExample(
            "The movie was not good.",
            0,
            "negation",
            "Stopword removal deletes 'not', turning a negative into 'good'.",
        ),
        AdversarialExample(
            "I would never recommend this.",
            0,
            "negation",
            "Dropping 'never' leaves 'would recommend', flipping the label.",
        ),
        AdversarialExample(
            "This is no masterpiece.",
            0,
            "negation",
            "Removing 'no' leaves the praise word 'masterpiece' standing alone.",
        ),
        AdversarialExample(
            "Apple fell as the US dollar rose.",
            1,
            "casing",
            "Lowercasing collapses 'Apple'->'apple' and 'US'->'us'.",
        ),
        AdversarialExample(
            "Let's eat, Grandma.",
            1,
            "punctuation",
            "Stripping the comma erases the difference from 'Let's eat Grandma'.",
        ),
        AdversarialExample(
            "The studies were inconclusive :(",
            0,
            "emoticon",
            "Punctuation stripping deletes the ':(' sentiment cue.",
        ),
    ]


def _cache() -> None:
    """Write the corpora to data/corpora/ as JSON (invoked by ``make data``)."""
    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    s_texts, s_labels = load_sentiment()
    t_texts, t_labels = load_topic()
    retrieval = load_retrieval()
    payloads = {
        "sentiment.json": {"texts": s_texts, "labels": s_labels},
        "topic.json": {"texts": t_texts, "labels": t_labels, "label_names": list(TOPIC_LABELS)},
        "retrieval.json": {
            "queries": retrieval.queries,
            "documents": retrieval.documents,
            "qrels": {str(q): sorted(docs) for q, docs in retrieval.qrels.items()},
        },
        "adversarial.json": [asdict(ex) for ex in load_adversarial()],
    }
    for name, payload in payloads.items():
        (CORPORA_DIR / name).write_text(json.dumps(payload, indent=2))
    print(f"✅ Cached {len(payloads)} corpora to {CORPORA_DIR}")


if __name__ == "__main__":
    _cache()
