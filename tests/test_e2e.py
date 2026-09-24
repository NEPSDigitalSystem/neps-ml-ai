"""
NEPS ML/AI — end-to-end: CSV -> validate -> features -> train -> registry ->
reload from disk -> explain + alerts for ONE participant -> JSON-safe output.
This is the path a Phase 5 endpoint will take.
"""

import json

import pytest

from app.ml import alerts, explain, features, train
from app.ml.loader import load_datasets


@pytest.fixture(scope="module")
def registry(tmp_path_factory, monkeypatch_module):
    out = tmp_path_factory.mktemp("registry")
    monkeypatch_module.setattr(train, "MODELS_DIR", out)
    train.main()
    return out


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_pipeline_end_to_end(registry, frame):
    risk = train.load_model("risk_classifier", models_dir=registry)
    traj = train.load_model("trajectory_depression", models_dir=registry)
    model, meta = risk

    # registry feature list matches what the feature code produces today
    assert meta["features"] == features.feature_columns(frame)
    assert set(meta["metrics"]) == set(meta["fairness"]) - {"positive_classes"}

    # one participant only: one-hot columns for other countries/sites are absent
    pid = frame.participant_id.iloc[0]
    one = features.build_training_frame({
        "participants": load_datasets()["participants"].query("participant_id == @pid"),
        "monthly_reports": load_datasets()["monthly_reports"].query("participant_id == @pid"),
    })
    assert len(one) == 24
    assert len(features.feature_columns(one)) < len(meta["features"])  # missing dummies

    latest = one[one.month == one.month.max()]
    alert = alerts.generate_alerts(latest, risk, traj)
    assert len(alert) == 1 and alert.priority[0] in {"P0", "P1", "P2", "none"}
    assert alert.trend[0] in features.TREND_LABELS

    ex = explain.explain_risk(model, meta, latest, top_k=5)
    assert len(ex) == 1 and len(ex[0]["contributions"]) == 5
    assert ex[0]["predicted"] == alert.risk_pred[0]

    # everything an endpoint would return must serialise
    payload = {"alert": alert.to_dict("records")[0], "explanation": ex[0], "meta": meta}
    json.dumps(payload, default=_reject)


def _reject(o):
    raise TypeError(f"not JSON-safe: {type(o).__name__}")


def test_single_participant_matches_full_frame_prediction(registry, frame):
    """align() must not change predictions versus the full training frame."""
    risk = train.load_model("risk_classifier", models_dir=registry)
    model, meta = risk
    pid = frame.participant_id.iloc[5]
    full_rows = frame[frame.participant_id == pid]
    d = load_datasets()
    one = features.build_training_frame({
        "participants": d["participants"].query("participant_id == @pid"),
        "monthly_reports": d["monthly_reports"].query("participant_id == @pid"),
    })
    assert list(model.predict(features.align(one, meta))) == list(model.predict(full_rows[meta["features"]]))
