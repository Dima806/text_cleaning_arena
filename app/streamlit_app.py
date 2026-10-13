"""Streamlit app: see the cleaning ritual help old models and hurt new ones.

Run with ``make run``. Four demos:
1. Destruction viewer  — watch each step transform a sentence.
2. Old-vs-new          — a pipeline's opposite effect on TF-IDF and a transformer.
3. Cleaning arena      — the four pipelines ranked per model and task.
4. Negation breaker    — stopword removal flipping 'not good' style examples.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.cleaning.pipelines import PIPELINE_NAMES, clean
from src.cleaning.steps import STEPS
from src.datasets.corpora import load_sentiment, load_topic
from src.evaluation.comparison import evaluate_cleaning
from src.evaluation.failure_cases import find_negation_flips, negations_removed
from src.models.bow import BowClassifier

st.set_page_config(page_title="Text Cleaning Arena", layout="wide")
st.title("Text Cleaning Arena")
st.caption("Every text-cleaning step you were taught is hurting your modern NLP model.")

tab_view, tab_arena, tab_negation = st.tabs(
    ["Destruction viewer", "Cleaning arena", "Negation breaker"]
)


with tab_view:
    st.subheader("Watch each step transform a sentence")
    text = st.text_input("Your sentence", "The movie was NOT good, I would never recommend it!")
    rows = []
    for name, fn in STEPS.items():
        rows.append({"step": name, "result": fn(text)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("**Named pipelines**")
    pipe_rows = [{"pipeline": name, "result": clean(text, name)} for name in PIPELINE_NAMES]
    st.dataframe(pd.DataFrame(pipe_rows), use_container_width=True, hide_index=True)

    removed = negations_removed(text, "standard")
    if removed:
        st.error(f"Standard cleaning deleted the negation(s): {', '.join(sorted(removed))}")


with tab_arena:
    st.subheader("The four pipelines ranked for a bag-of-words model")
    task = st.selectbox("Task", ["sentiment", "topic"])
    texts, labels = load_sentiment() if task == "sentiment" else load_topic()
    frame = evaluate_cleaning(texts, labels, BowClassifier)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.bar_chart(frame.set_index("pipeline")["macro_f1"])


with tab_negation:
    st.subheader("Stopword removal flips negative reviews")
    texts, labels = load_sentiment()
    flips = find_negation_flips(texts, labels, pipeline="standard")
    flip_rows = [
        {
            "original": flip.text,
            "cleaned": flip.cleaned,
            "label": "negative" if flip.label == 0 else "positive",
            "deleted": ", ".join(sorted(flip.removed)),
        }
        for flip in flips
    ]
    st.dataframe(pd.DataFrame(flip_rows), use_container_width=True, hide_index=True)
    st.info(f"{len(flips)} reviews lost their negation under the standard pipeline.")
