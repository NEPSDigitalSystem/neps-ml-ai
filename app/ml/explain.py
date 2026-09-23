"""Explainability for WP6: SHAP contributions in plain language.

explain_risk(model, meta, X) -> one list per row of
    {"feature", "label", "value", "contribution", "share", "direction", "sentence"}
sorted by |contribution|, for the class the model predicted. SHAP values are
in log-odds space; `share` is the feature's fraction of total |SHAP| for that
row and class, a relative attribution, not a probability. Sentences read
"Suicidality contributed 32% towards HIGH risk".

Response shape is a proposal; confirm with the backend intern before the
/explain endpoint (Phase 5).
"""
import re

import numpy as np
import shap

from app.ml import schema

TOP_K = 5

_SCORE_LABELS = {
    "anxiety": "anxiety", "depression": "depression", "stress": "stress",
    "social_isolation": "social isolation", "self_esteem": "self-esteem",
    "loneliness": "loneliness", "daily_functioning": "daily functioning",
    "mood_score": "mood", "sleep_quality_score": "sleep quality",
    "fatigue_score": "fatigue", "attendance_score": "attendance",
    "coping_score": "coping", "substance_abuse_score": "substance use",
    "suicidality_score": "suicidality",
}
_SUFFIX_LABELS = {"_lag1": " last month", "_lag2": " two months ago",
                  "_lag3": " three months ago", "_delta1": " change since last month"}


def feature_label(name):
    """anxiety_lag1 -> 'anxiety last month'; country_Ghana -> 'country: Ghana'."""
    for suf, text in _SUFFIX_LABELS.items():
        if name.endswith(suf):
            return _SCORE_LABELS.get(name[: -len(suf)], name[: -len(suf)]) + text
    if name in _SCORE_LABELS:
        return _SCORE_LABELS[name]
    for demo in schema.DEMOGRAPHICS:
        if name.startswith(f"{demo}_"):
            return f"{demo.replace('_', ' ')}: {name[len(demo) + 1:].replace('_', ' ')}"
    return name.replace("_", " ")


def explain_risk(model, meta, X, top_k=TOP_K):
    """SHAP contributions for the predicted class of each row of X."""
    X = X[meta["features"]]
    values = np.asarray(shap.TreeExplainer(model).shap_values(X))
    if values.ndim == 2:  # binary: shap gives log-odds of classes_[1] only
        values = np.stack([-values, values], axis=-1)
    classes = list(model.classes_)  # SHAP's class axis follows the model, not meta
    preds = model.predict(X)
    out = []
    for i, pred in enumerate(preds):
        v = values[i, :, classes.index(pred)]
        total = float(np.abs(v).sum()) or 1.0
        order = np.argsort(-np.abs(v))[:top_k]
        items = []
        for j in order:
            name = meta["features"][j]
            share = float(abs(v[j]) / total)
            direction = "towards" if v[j] > 0 else "away from"
            items.append({
                "feature": name,
                "label": feature_label(name),
                "value": None if np.isnan(X.iloc[i, j]) else float(X.iloc[i, j]),
                "contribution": round(float(v[j]), 4),
                "share": round(share, 4),
                "direction": direction,
                "sentence": f"{feature_label(name).capitalize()} contributed {share:.0%} {direction} {pred} risk",
            })
        out.append({"predicted": str(pred), "contributions": items})
    return out
