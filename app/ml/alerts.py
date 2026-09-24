"""Early-warning alerts: hard safety rules first, model second, trend third.

generate_alerts(frame, risk, trajectory=None) -> DataFrame, one row per input
row, with priority (P0 crisis, P1 high, P2 watch, none), the reasons that
fired, and suggested intervention categories from the WP6 proposal
(counselling, peer support, follow-up assessment) plus safeguarding referral.

Rules are deliberately simple and readable by a field lead. Human-in-the-loop:
an alert is a prompt for review, never an automatic action.

ponytail: thresholds below are placeholders read off the mock data
(suicidality >= 7 is exactly where the mock flips to HIGH). Calibrate against
the study's safeguarding protocol before real use.

A NaN suicidality_score never trips the score half of the crisis rule; the
screening half (a required contract column) still does.
"""
import numpy as np
import pandas as pd

from app.ml import features
from app.ml.schema import POSITIVE_TIERS as POSITIVE

SUICIDALITY_CRISIS = 7.0
CRISIS_SCREENING = ("Active plan", "Previous attempt")
P_HIGH_ALERT = 0.5          # P(HIGH or CRITICAL) from the classifier

INTERVENTIONS = {
    "P0": ["safeguarding referral", "counselling"],
    "P1": ["counselling", "follow-up assessment"],
    "P2": ["peer support", "follow-up assessment"],
    "none": [],
}


def _p_positive(model, X):
    proba = model.predict_proba(X)
    idx = [i for i, c in enumerate(model.classes_) if str(c) in POSITIVE]
    return proba[:, idx].sum(axis=1) if idx else np.zeros(len(X))


def generate_alerts(frame, risk, trajectory=None):
    """frame: rows from build_training_frame. risk: (model, meta).
    trajectory: optional (model, meta) for depression, used for the trend."""
    model, meta = risk
    X = features.align(frame, meta)
    pred = model.predict(X).astype(str)
    p_pos = _p_positive(model, X)

    out = pd.DataFrame({
        "participant_id": frame["participant_id"].to_numpy(),
        "month": frame["month"].to_numpy(),
        "risk_pred": pred,
        "p_high": np.round(p_pos, 4),
        "suicidality_score": frame["suicidality_score"].to_numpy(),
    })

    crisis = (out["suicidality_score"] >= SUICIDALITY_CRISIS).to_numpy()
    crisis |= frame["suicidality_screening"].isin(CRISIS_SCREENING).to_numpy()
    high = np.isin(pred, POSITIVE) | (p_pos >= P_HIGH_ALERT)

    out["trend"] = "unknown"
    if trajectory is not None:
        t_model, t_meta = trajectory
        target = t_meta["target"].removesuffix("_next")
        nxt = t_model.predict(features.align(frame, t_meta))
        out["trend"] = features.trend_label(frame[target], nxt, t_meta["trend_threshold"])
    watch = (out["trend"] == "Deteriorating").to_numpy() | (pred == "MEDIUM")

    out["priority"] = np.select([crisis, high, watch], ["P0", "P1", "P2"], default="none")  # first match wins
    out["reasons"] = [
        [r for r, hit in (("suicidality crisis rule", c), ("model high risk", h), ("deteriorating or medium", w)) if hit]
        for c, h, w in zip(crisis, high, watch)
    ]
    out["interventions"] = out["priority"].map(INTERVENTIONS)
    return out
