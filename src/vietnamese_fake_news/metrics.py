from __future__ import annotations

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def classification_metrics(labels, preds) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(
            precision_score(labels, preds, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(labels, preds, average="macro", zero_division=0)
        ),
        "f1": float(
            f1_score(labels, preds, average="macro", zero_division=0)
        ),
    }
