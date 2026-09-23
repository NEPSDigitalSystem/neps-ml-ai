"""Baseline risk classifier and model registry.

    python -m app.ml.train        # train on data/raw, print metrics, save

Registry layout: models/<name>/<version>/{model.pkl, meta.json}. Version is
the UTC training timestamp. meta.json records data_version, feature list,
class labels, metrics (overall and per fairness group) and library versions
so a retrain on real data can be compared with the mock baseline.

Deferred until real data: trajectory regression (Phase 3), text features,
sequence models, fusion, dropout prediction.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from app.ml import evaluate, features
from app.ml.loader import load_datasets

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DATA_VERSION = "mock-redcap-v0.3.0"


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
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
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
    model, meta = train_risk_classifier(frame)
    out = save_model(model, meta)
    print(json.dumps(meta["metrics"], indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
