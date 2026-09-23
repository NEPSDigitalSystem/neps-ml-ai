"""Fairness audit for the risk classifier.

fairness_report(test_frame, y_true, y_pred) -> per group in FAIRNESS_GROUPS:
  - per value: n, positives, selection_rate P(pred positive),
    tpr P(pred positive | true positive), fpr P(pred positive | true negative)
  - demographic_parity_gap: max - min selection_rate across values
  - equal_opportunity_gap:  max - min tpr across values (values with no true
    positives are skipped; tpr is None there)

"Positive" means flagged for follow-up: HIGH or CRITICAL. Gaps are reported,
not judged; the tolerance is a governance decision (ponytail: a flat
GAP_WARN=0.1 marks groups worth a look until WP6 sets one).
"""
import pandas as pd

from app.ml import schema

POSITIVE = ("HIGH", "CRITICAL")
GAP_WARN = 0.1


def _group_key(series):
    """Group labels as strings; missing demographics form their own group."""
    return series.astype(object).where(series.notna(), "missing").astype(str)


def _rates(y_true, y_pred):
    n = int(len(y_true))
    pos = y_true.isin(POSITIVE)
    flagged = y_pred.isin(POSITIVE)
    tp = int((flagged & pos).sum())
    fp = int((flagged & ~pos).sum())
    return {
        "n": n,
        "positives": int(pos.sum()),
        "selection_rate": round(float(flagged.mean()), 4) if n else None,
        "tpr": round(tp / int(pos.sum()), 4) if pos.any() else None,
        "fpr": round(fp / int((~pos).sum()), 4) if (~pos).any() else None,
    }


def _gap(values):
    values = [v for v in values if v is not None]
    return round(max(values) - min(values), 4) if len(values) > 1 else None


def fairness_report(test_frame, y_true, y_pred, groups=schema.FAIRNESS_GROUPS):
    y_true = pd.Series(list(y_true), index=test_frame.index).astype(str)
    y_pred = pd.Series(list(y_pred), index=test_frame.index).astype(str)
    report = {"positive_classes": list(POSITIVE), "overall": _rates(y_true, y_pred)}
    for group in groups:
        per = {str(val): _rates(y_true[idx], y_pred[idx])
               for val, idx in test_frame.groupby(_group_key(test_frame[group])).groups.items()}
        dp = _gap(v["selection_rate"] for v in per.values())
        eo = _gap(v["tpr"] for v in per.values())
        report[group] = {
            "values": per,
            "demographic_parity_gap": dp,
            "equal_opportunity_gap": eo,
            "warn": bool((dp or 0) > GAP_WARN or (eo or 0) > GAP_WARN),
        }
    return report
