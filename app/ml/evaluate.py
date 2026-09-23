"""Evaluation harness.

Rules:
- Split by participant_id, never by row (trajectory leakage otherwise).
- Every report includes a breakdown by schema.FAIRNESS_GROUPS.
- Mock-data metrics are pipeline validation only; risk_flag is a threshold
  on its own inputs and scores ~0.97 with a depth-3 tree.
"""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from app.ml import schema


def _group_key(series):
    """Group labels as strings; missing demographics form their own group."""
    return series.astype(object).where(series.notna(), "missing").astype(str)


def participant_split(frame, test_size=0.2, seed=42):
    """Return (train_idx, test_idx); no participant appears in both."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(frame, groups=frame["participant_id"]))


def classification_metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def regression_metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        # r2 is undefined for a constant y_true (tiny fairness cells)
        "r2": round(float(r2_score(y_true, y_pred)), 4) if len(set(map(float, y_true))) > 1 else None,
    }


def metrics_report(test_frame, y_true, y_pred, metric=classification_metrics):
    """Overall metrics plus one entry per value of each fairness group."""
    report = {"overall": metric(y_true, y_pred)}
    y_true = pd.Series(list(y_true), index=test_frame.index)
    y_pred = pd.Series(list(y_pred), index=test_frame.index)
    for group in schema.FAIRNESS_GROUPS:
        report[group] = {
            str(val): metric(y_true[idx], y_pred[idx])
            for val, idx in test_frame.groupby(_group_key(test_frame[group])).groups.items()
        }
    return report
