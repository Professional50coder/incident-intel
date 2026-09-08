from __future__ import annotations

from typing import Sequence

from sklearn.metrics import roc_auc_score


def compute_auc_roc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUC-ROC for anomaly scores against binary frame-level ground truth (1 = anomalous)."""
    if len(set(labels)) < 2:
        raise ValueError("AUC-ROC is undefined when all labels are the same class")
    return float(roc_auc_score(labels, scores))
