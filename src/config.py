"""Project configuration and path constants.

Settings are loaded from ``config/settings.yaml`` via pydantic. Everything the
arena needs to be reproducible — the seed, the pinned model revision, the list of
pipelines and tasks — lives here rather than being hardcoded in notebooks.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
DATA_DIR = PROJECT_ROOT / "data"
CORPORA_DIR = DATA_DIR / "corpora"
EMB_DIR = DATA_DIR / "emb"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "data"


# --- Settings models -------------------------------------------------------


class ModelSettings(BaseModel):
    """Transformer encoder settings, pinned for reproducibility."""

    name: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str | None = None
    batch_size: int = 32


class EvaluationSettings(BaseModel):
    """Shared evaluation knobs."""

    test_size: float = 0.3
    retrieval_k: int = 10


class Settings(BaseModel):
    """Top-level configuration for the whole project."""

    seed: int = 42
    model: ModelSettings = Field(default_factory=ModelSettings)
    pipelines: list[str] = Field(
        default_factory=lambda: ["none", "minimal", "standard", "aggressive"]
    )
    tasks: list[str] = Field(default_factory=lambda: ["sentiment", "topic", "retrieval"])
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)


@lru_cache(maxsize=1)
def get_settings(path: Path | None = None) -> Settings:
    """Load settings from YAML, falling back to defaults when the file is absent."""
    config_path = path or CONFIG_PATH
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text()) or {}
        return Settings.model_validate(raw)
    return Settings()
