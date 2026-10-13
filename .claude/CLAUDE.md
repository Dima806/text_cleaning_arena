# Project: text_cleaning_arena

> A head-to-head benchmark of the standard text-preprocessing pipeline against doing
> almost nothing, across classic bag-of-words models and modern transformer embeddings,
> on classification and retrieval. The reflex that opens every NLP notebook is quietly
> capping the performance of every modern model it touches — this repo measures it.

## Identity
Text-preprocessing benchmark. 6 executed notebooks, 1 Streamlit app, a `src/` library.
**4 cleaning pipelines** (`none` / `minimal` / `standard` / `aggressive`)
× **3 models** (TF-IDF+LogReg, BM25, transformer embeddings)
× **2 task families** (classification, retrieval).

**Thesis:** the cleaning ritual was built for sparse bag-of-words models; on modern
transformers it is at best neutral and at worst destructive. Removing "not" as a
stopword flips sentiment. The right amount of cleaning depends on the model and task,
and is often the opposite of what tutorials teach.

## Claims this repo must defend (asserted in tests — keep them true)
Do not let a refactor soften these; they are what the project proves.
- Stopword removal (and the `standard`/`aggressive` pipelines) delete `not`/`no`/`never`,
  flipping labeled negative reviews → `tests/test_negation.py` (33 flips on the sentiment set).
- Aggressive stemming **lifts BM25 nDCG** over raw text by merging word forms →
  `tests/test_evaluation.py::test_stemming_helps_bm25_retrieval`.
- A cross-validated classifier clears chance on the sentiment corpus (not leaky) →
  `test_evaluate_classification_learns_sentiment`.
- Every cleaning step's/pipeline's vocabulary reduction is **quantified** (`cleaning/analysis.py`, nb01).
- Pipelines compose correctly: `none` is identity, `aggressive` stems, unknown names raise.

## Results as actually measured (demonstrated in notebooks — honest, data-dependent)
- Transformer peaks on `none`/`minimal`; `aggressive` is its worst pipeline (nb03, nb04).
- Retrieval splits cleanly: **BM25 nDCG climbs ~0.19 → 0.98** as cleaning normalizes word
  forms, while **dense stays ~1.0** (needs no cleaning) — nb05.
- Cleaning does the least damage on **topic** (content words survive) and the most on
  **sentiment** (negation/casing are the label).
- **Honest caveat — do not fabricate around it:** on these small, cleanly separable
  *synthetic* corpora, classification sits near the ceiling (transformer hits 1.0 on raw
  sentiment), so the textbook "cleaning *helps* TF-IDF while *hurting* the transformer"
  opposite-directions effect does **not** strongly appear. Cleaning hurts both, the
  transformer most under `aggressive`. The negation flip and the BM25 retrieval split are
  the clean, guaranteed demonstrations. To strengthen the classification story, swap in a
  real dataset (SST/IMDB/AG News) rather than tuning the synthetic data to force a result.

## Stack & constraints
- Python **3.11+** (pinned to **3.12** via `.python-version`; system 3.14 has no torch wheels).
- Package manager **`uv` only** — never `pip`, never a bare `python`; always `uv run ...`.
- Must run inside a **2-CPU / 8 GB GitHub Codespace, no GPU**. Keep everything CPU-viable.
- Key deps: `sentence-transformers` (CPU `torch`), `scikit-learn`, `nltk`, `bm25s`, `numpy`,
  `scipy`, `pandas`, `matplotlib`, `plotly`, `streamlit`, `pydantic`, `pydantic-settings`, `pyyaml`.
- Dev: `pytest`, `ruff`, `ty`, `jupyter`, `ipykernel`, `nbconvert`, `nbclient`.
- Model: **`all-MiniLM-L6-v2`** (22M params), pinned by revision in `config/settings.yaml`
  (`1110a243…`). Imported lazily in `models/transformer.py` so `src` import stays cheap and
  tests never download it.

## Structure
```
src/cleaning/   steps.py (normalize/lowercase/stopwords/stem/lemmatize/punct/numbers)
                pipelines.py (none/minimal/standard/aggressive; clean(text, pipeline))
                analysis.py (vocab size + info loss per step/pipeline)
src/models/     bow.py (TF-IDF + LogReg), bm25.py (pure-Python Okapi), transformer.py (cached, lazy)
src/datasets/   corpora.py (sentiment GENERATED from balanced pools, seeded+frozen+non-leaky;
                topic hand-written; retrieval stemming-sensitive by design; adversarial set)
src/evaluation/ classification.py (k-fold CV, model FACTORY), retrieval.py (nDCG/recall),
                failure_cases.py (negation flips), comparison.py (evaluate_cleaning arena)
src/config.py   pydantic settings from config/settings.yaml; path constants
src/visualisation.py  plot_* figures + save_figure(PNG) + save_json(outputs/data)
app/streamlit_app.py  destruction viewer / arena / negation breaker
notebooks/01..06  ritual → destroys → old-vs-new → class arena → retrieval arena → framework
config/settings.yaml  seed, model+revision, pipelines, tasks, eval knobs
data/corpora/   JSON cache (committed); data/emb/  embeddings cache (gitignored)
outputs/figures/  *.png  (committed)     outputs/data/  *.json results (committed)
tests/          test_steps, test_pipelines, test_negation, test_evaluation  (35 tests)
```

## Commands (always via Makefile / uv)
```
make setup      # first time: install uv, uv sync --all-extras, nltk data, ipykernel
make data       # cache corpora to data/corpora/  -> python -m src.datasets.corpora
make embed      # pre-compute + cache transformer embeddings for all pipelines
make lint       # ruff format + ruff check --fix + ty check
make test       # pytest  (test-cov for coverage)
make notebooks  # execute all notebooks (runs data + embed first)
make run        # streamlit app on :8501
make dev        # fast loop: lint + test
make ci         # sync + lint + test   (the green bar — keep it at exit 0)
```
Run a single test: `uv run pytest tests/test_negation.py -k negation`.

## Architecture rules
- Public surface the reader relies on: `clean(text, pipeline)` and
  `evaluate_cleaning(texts, labels, model_factory)` — keep these stable and simple.
  Note `evaluate_cleaning` / `evaluate_classification` take a **model factory** (a zero-arg
  callable like the `BowClassifier` class), building a fresh model per CV fold.
- Classification is **cross-validated** (StratifiedKFold, `shuffle=True`, seeded). A single
  train/test split is too noisy on these tiny corpora — do not revert to one split.
- **Steps compose into pipelines; pipelines are data, not scattered logic.** A new cleaning
  behavior is a toggleable step in `steps.py` plus an entry wired into `pipelines.py`.
- Every step reports *what it does to vocabulary and what it destroys* (`cleaning/analysis.py`).
- Models share a thin fit/predict (classifier) or index/retrieve (retriever) interface so the
  harnesses run any (pipeline × model) cell uniformly.
- Config (seed, model, pipelines, tasks) comes from `config/settings.yaml` via `src/config.py`.
  Don't hardcode seeds or model names in notebooks.

## Notebook & output conventions
- Notebooks are orchestration, not libraries: logic lives in `src/`; notebooks narrate + call it.
- **Every figure → `save_figure(fig, "nbNN_*.png")`** into `outputs/figures/`.
- **Every table/number/text → `save_json(obj, "nbNN_*.json")`** into `outputs/data/` (a
  DataFrame is saved as row records). Prefix artifacts with the notebook number.
- Notebooks are built/executed via `nbclient`; `make notebooks` re-executes them with nbconvert.

## Caching & reproducibility (non-negotiable)
- Transformer embeddings persist to `data/emb/`, keyed by (model, pipeline, cleaned text).
  First notebook run < 8 min; cached reruns < 2 min. Never recompute cached embeddings.
- Corpora are frozen in `corpora.py` (sentiment seeded-generated; topic/retrieval literal) and
  cached to `data/corpora/` — nothing depends on a live download.
- nltk stemming/lemmatization have pure-Python fallbacks (CI does not download nltk data).
- Seeded everywhere. The negation-flip and BM25-stemming results are deterministic + asserted.

## Conventions
- `ruff` line length 99, lint set `E,F,W,I,UP,N,B,A,SIM,PTH` (ignore `E501`). Use `pathlib`.
- Type-checked with `ty` (`make typecheck`). Keep `src/` clean under it.

## Token rules
- Caveman mode: ON. Code-first, minimal prose. `/compact` at notebook boundaries.
- Prefer editing `src/` over inlining logic in notebooks.

## Definition of done
`make ci` exits 0 (sync + lint + ty + 35 tests). Notebooks execute end-to-end and refresh
their PNG/JSON artifacts. The asserted claims above still hold and the negation-flip test
still fails loudly if someone "fixes" the stopword list.
