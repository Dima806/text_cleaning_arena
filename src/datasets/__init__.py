"""Small, frozen text corpora for the arena.

Import the loaders directly from :mod:`src.datasets.corpora`. That module is also a
CLI (``python -m src.datasets.corpora`` caches the corpora), so it is deliberately
not re-exported here — re-exporting it would make the package import it before
``runpy`` runs it as ``__main__``.
"""
