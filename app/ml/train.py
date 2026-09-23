"""Baseline models and model registry.

    python -m app.ml.train        # train all baselines on data/raw, save, print metrics

Models:
- risk_classifier: monthly scores + lags + demographics -> risk_flag
- trajectory_<score>: same features -> that score one month later, plus a
  trend label (Improving / Stable / Deteriorating) from the predicted change

Registry layout: models/<name>/<version>/{model.pkl, meta.json}. Version is
the UTC training timestamp. meta.json records data_version, feature list,
class labels, metrics (overall and per fairness group) and library versions
so a retrain on real data can be compared with the mock baseline.

Deferred until real data: text features, sequence models, fusion, dropout.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from app.ml import evaluate, fairness, features
from app.ml.loader import load_datasets

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DATA_VERSION = "mock-redcap-v0.3.0"

# Only scores where higher = worse; trend_label assumes that direction.
# Inverted scores (self_esteem, mood_score, coping_score, ...) need a sign flip.
TRAJECTORY_TARGETS = ["depression", "anxiety"]
HIGHER_IS_WORSE = {"depression", "anxiety", "stress", "social_isolation", "loneliness",
                   "fatigue_score", "substance_abuse_score", "suicidality_score"}
assert set(TRAJECTORY_TARGETS) <= HIGHER_IS_WORSE
# ponytail: fixed change threshold for trend labels; calibrate per score
# against clinical minimal-important-difference when real data arrives.
TREND_THRESHOLD = 2.0
TREND_LABELS = ("Improving", "Stable", "Deteriorating")


def _versions():
    return {"sklearn": sklearn.__version__, "pandas": pd.__version__}


def train_risk_classifier(frame, seed=42):
    """Fit on a participant-level train split; return (model, meta)."""
    cols = features.feature_columns(frame)
    train_idx, test_idx = evaluate.participant_split(frame, seed=seed)
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]

    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
    model.fit(train[cols], train[features.TARGET])

    pred = model.predict(test[cols])
    meta = {
        "name": "risk_classifier",
        "target": features.TARGET,
        "classes": [str(c) for c in model.classes_],
        "features": cols,
        "data_version": DATA_VERSION,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "seed": seed,
        "metrics": evaluate.metrics_report(test, test[features.TARGET], pred),
        "fairness": fairness.fairness_report(test, test[features.TARGET], pred),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **_versions(),
    }
    return model, meta


def trend_label(current, predicted, threshold=TREND_THRESHOLD):
    """Higher score = worse: change >= threshold is Deteriorating, <= -threshold Improving."""
    change = np.asarray(predicted, dtype=float) - np.asarray(current, dtype=float)
    if np.isnan(change).any():
        raise ValueError("trend_label: NaN in current or predicted")
    return np.where(change >= threshold, "Deteriorating",
                    np.where(change <= -threshold, "Improving", "Stable"))


def train_trajectory(frame, target, seed=42):
    """Predict next month's `target` score. Returns (model, meta)."""
    data = features.add_next_month_targets(frame, [target])
    cols = features.feature_columns(data)
    y_col = f"{target}_next"
    train_idx, test_idx = evaluate.participant_split(data, seed=seed)
    train, test = data.iloc[train_idx], data.iloc[test_idx]

    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(train[cols], train[y_col])

    pred = model.predict(test[cols])
    trend_true = trend_label(test[target], test[y_col])
    trend_pred = trend_label(test[target], pred)
    meta = {
        "name": f"trajectory_{target}",
        "target": y_col,
        "features": cols,
        "data_version": DATA_VERSION,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "seed": seed,
        "trend_threshold": TREND_THRESHOLD,
        "metrics": evaluate.metrics_report(test, test[y_col], pred, metric=evaluate.regression_metrics),
        "trend_metrics": evaluate.metrics_report(test, trend_true, trend_pred),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **_versions(),
    }
    return model, meta


def save_model(model, meta, models_dir=None):
    models_dir = models_dir or MODELS_DIR
    version = meta["trained_at"].replace(":", "").replace("+0000", "Z")
    out = Path(models_dir) / meta["name"] / version
    out.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, out / "model.pkl")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def load_model(name, version=None, models_dir=None):
    """Load a saved model; latest version by default. Returns (model, meta)."""
    root = Path(models_dir or MODELS_DIR) / name
    versions = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    if not versions:
        raise FileNotFoundError(f"no saved versions of {name!r} under {root}")
    version = version or versions[-1]
    return joblib.load(root / version / "model.pkl"), json.loads((root / version / "meta.json").read_text())


def main():
    frame = features.build_training_frame(load_datasets())
    trained = [train_risk_classifier(frame)] + [train_trajectory(frame, t) for t in TRAJECTORY_TARGETS]
    for model, meta in trained:
        out = save_model(model, meta)
        print(meta["name"], json.dumps(meta["metrics"]["overall"]),
              json.dumps(meta.get("trend_metrics", {}).get("overall", "")))
        print(f"saved {out}")


if __name__ == "__main__":
    main()
