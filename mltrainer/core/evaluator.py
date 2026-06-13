from __future__ import annotations

from math import sqrt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def evaluate_classification(y_true, y_pred) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def sort_results(results: list[dict], ranking_metric: str) -> list[dict]:
    reverse = ranking_metric != "rmse"
    return sorted(results, key=lambda item: item["metrics"][ranking_metric], reverse=reverse)
