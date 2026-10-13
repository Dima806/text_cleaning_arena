"""Evaluation harnesses: classification, retrieval, failure cases, and the arena."""

from src.evaluation.classification import ClassificationResult, evaluate_classification
from src.evaluation.comparison import evaluate_cleaning
from src.evaluation.failure_cases import NegationFlip, find_negation_flips, negations_removed
from src.evaluation.retrieval import ndcg_at_k, recall_at_k

__all__ = [
    "ClassificationResult",
    "NegationFlip",
    "evaluate_classification",
    "evaluate_cleaning",
    "find_negation_flips",
    "ndcg_at_k",
    "negations_removed",
    "recall_at_k",
]
