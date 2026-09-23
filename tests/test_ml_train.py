"""
NEPS ML/AI — Phase 2 tests: features, participant split, evaluation, registry.
"""

import json

import numpy as np
import pandas as pd
import pytest

from app.ml import evaluate, features, schema, train


# ─── FEATURES ──────────────────────────────────────────────────────────────────

class TestMonthlyFeatures:
    def test_row_count_and_order_preserved(self, data):
        out = features.monthly_features(data["monthly_reports"])
        assert len(out) == 3600
        assert out.groupby("participant_id")["month"].apply(lambda s: s.is_monotonic_increasing).all()

    def test_lag_is_previous_month_same_participant(self, data):
        out = features.monthly_features(data["monthly_reports"]).set_index(["participant_id", "month"])
        pid = out.index[0][0]
        for m in (2, 5, 24):
            assert out.loc[(pid, m), "anxiety_lag1"] == out.loc[(pid, m - 1), "anxiety"]
        assert out.loc[(pid, 5), "anxiety_lag3"] == out.loc[(pid, 2), "anxiety"]

    def test_lags_are_nan_in_early_months_only(self, data):
        out = features.monthly_features(data["monthly_reports"])
        assert out.loc[out.month == 1, "anxiety_lag1"].isna().all()
        assert out.loc[out.month <= 3, "anxiety_lag3"].isna().all()
        assert out.loc[out.month > 3, "anxiety_lag3"].notna().all()

    def test_input_order_does_not_matter(self, data):
        shuffled = data["monthly_reports"].sample(frac=1, random_state=0)
        assert features.monthly_features(shuffled).equals(features.monthly_features(data["monthly_reports"]))

    def test_delta_sign(self, data):
        out = features.monthly_features(data["monthly_reports"])
        pid = out.participant_id.iloc[0]
        p = out[out.participant_id == pid].set_index("month")
        assert p.loc[2, "stress_delta1"] == pytest.approx(p.loc[2, "stress"] - p.loc[1, "stress"])
        assert pd.isna(p.loc[1, "stress_delta1"])

    def test_every_score_gets_four_features(self, data):
        out = features.monthly_features(data["monthly_reports"])
        for col in schema.MONTHLY_SCORES:
            for suf in features.TEMPORAL_SUFFIXES:
                assert f"{col}{suf}" in out.columns


class TestDemographicFeatures:
    def test_one_row_per_participant(self, data):
        out = features.demographic_features(data["participants"])
        assert len(out) == 150 and out.participant_id.is_unique

    def test_one_hot_sums_to_one_per_category(self, data):
        out = features.demographic_features(data["participants"])
        for cat in ("country", "gender", "socioeconomic_status"):
            cols = [c for c in out.columns if c.startswith(f"{cat}_")]
            assert (out[cols].sum(axis=1) == 1).all(), cat

    def test_raw_fairness_groups_kept(self, data):
        out = features.demographic_features(data["participants"])
        assert set(schema.FAIRNESS_GROUPS) <= set(out.columns)
        assert "age" in out.columns

    def test_no_spaces_in_column_names(self, data):
        out = features.demographic_features(data["participants"])
        assert not any(" " in c for c in out.columns)


class TestTrainingFrame:
    def test_shape(self, frame):
        assert len(frame) == 3600
        assert frame.duplicated(["participant_id", "month"]).sum() == 0

    def test_no_missing_demographics_after_merge(self, frame):
        assert frame["country"].isna().sum() == 0
        assert frame["age"].isna().sum() == 0

    def test_feature_columns_exclude_ids_target_and_groups(self, frame):
        cols = features.feature_columns(frame)
        assert not set(features.NON_FEATURE) & set(cols)
        assert features.TARGET not in cols
        assert all(pd.api.types.is_numeric_dtype(frame[c]) for c in cols)

    def test_no_leakage_columns(self, frame):
        # requires_follow_up is derived from risk_flag in the mock: must not be a feature
        assert "requires_follow_up" not in features.feature_columns(frame)
        assert "survey_date" not in features.feature_columns(frame)


# ─── EVALUATE ──────────────────────────────────────────────────────────────────

class TestSplit:
    def test_participants_do_not_overlap(self, frame):
        tr, te = evaluate.participant_split(frame)
        assert not set(frame.iloc[tr].participant_id) & set(frame.iloc[te].participant_id)
        assert len(tr) + len(te) == len(frame)

    def test_split_size_and_determinism(self, frame):
        tr, te = evaluate.participant_split(frame, test_size=0.2, seed=1)
        assert frame.iloc[te].participant_id.nunique() == 30
        tr2, te2 = evaluate.participant_split(frame, test_size=0.2, seed=1)
        assert np.array_equal(te, te2)
        _, te3 = evaluate.participant_split(frame, test_size=0.2, seed=2)
        assert not np.array_equal(te, te3)


class TestMetrics:
    def test_perfect_and_wrong(self):
        y = ["LOW", "HIGH", "LOW", "HIGH"]
        assert evaluate.classification_metrics(y, y) == {"n": 4, "accuracy": 1.0, "macro_f1": 1.0}
        m = evaluate.classification_metrics(y, ["HIGH", "LOW", "HIGH", "LOW"])
        assert m["accuracy"] == 0.0 and m["macro_f1"] == 0.0

    def test_report_has_every_group_value_and_sums_to_n(self, frame):
        test = frame.iloc[:100]
        y = test[features.TARGET]
        rep = evaluate.metrics_report(test, y, y)
        assert rep["overall"]["n"] == 100
        for g in schema.FAIRNESS_GROUPS:
            assert set(rep[g]) == set(test[g].astype(str).unique())
            assert sum(v["n"] for v in rep[g].values()) == 100

    def test_report_ignores_input_index(self, frame):
        test = frame.iloc[50:80]
        y_series = test[features.TARGET].reset_index(drop=True)   # index 0..29, not test.index
        rep = evaluate.metrics_report(test, y_series, test[features.TARGET].to_numpy())
        assert rep["overall"]["accuracy"] == 1.0

    def test_group_metrics_use_the_right_rows(self, frame):
        test = frame.iloc[:300]
        y_true = test[features.TARGET].to_numpy()
        y_pred = y_true.copy()
        ghana = (test["country"] == "Ghana").to_numpy()
        y_pred[ghana] = "CRITICAL"   # corrupt one country only
        rep = evaluate.metrics_report(test, y_true, y_pred)
        assert rep["country"]["Ghana"]["accuracy"] < 1.0
        assert all(v["accuracy"] == 1.0 for k, v in rep["country"].items() if k != "Ghana")


# ─── TRAIN + REGISTRY ──────────────────────────────────────────────────────────

class TestTrain:
    def test_model_predicts_known_classes(self, classifier, frame):
        model, meta = classifier
        assert set(meta["classes"]) <= set(schema.RISK_FLAG_TIERS)
        pred = model.predict(frame[meta["features"]].head(5))
        assert set(pred) <= set(schema.RISK_FLAG_TIERS)

    def test_meta_is_complete(self, classifier):
        _, meta = classifier
        for k in ("name", "target", "classes", "features", "data_version", "n_train",
                  "n_test", "seed", "metrics", "trained_at", "sklearn", "pandas"):
            assert k in meta, k
        assert meta["n_train"] + meta["n_test"] == 3600
        assert meta["metrics"]["overall"]["n"] == meta["n_test"]
        assert set(meta["metrics"]) == {"overall", *schema.FAIRNESS_GROUPS}

    def test_beats_majority_baseline(self, classifier):
        # mock risk_flag is a threshold on its inputs; anything below ~0.92 (LOW share) is broken
        _, meta = classifier
        assert meta["metrics"]["overall"]["accuracy"] > 0.92
        assert meta["metrics"]["overall"]["macro_f1"] > 0.5

    def test_deterministic(self, frame, classifier):
        _, meta2 = train.train_risk_classifier(frame)
        assert meta2["metrics"] == classifier[1]["metrics"]


class TestRegistry:
    def test_save_and_load_roundtrip(self, classifier, frame, tmp_path):
        model, meta = classifier
        out = train.save_model(model, meta, models_dir=tmp_path)
        assert (out / "model.pkl").exists() and (out / "meta.json").exists()
        assert out.parent.name == "risk_classifier"
        loaded, loaded_meta = train.load_model("risk_classifier", models_dir=tmp_path)
        assert loaded_meta == json.loads(json.dumps(meta))
        X = frame[meta["features"]].head(20)
        assert list(loaded.predict(X)) == list(model.predict(X))

    def test_load_latest_version(self, classifier, tmp_path):
        model, meta = classifier
        old = dict(meta, trained_at="2020-01-01T00:00:00+00:00")
        train.save_model(model, old, models_dir=tmp_path)
        train.save_model(model, meta, models_dir=tmp_path)
        _, latest = train.load_model("risk_classifier", models_dir=tmp_path)
        assert latest["trained_at"] == meta["trained_at"]
        _, pinned = train.load_model("risk_classifier", version="2020-01-01T000000Z", models_dir=tmp_path)
        assert pinned["trained_at"].startswith("2020")

    def test_load_from_empty_registry(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no saved versions"):
            train.load_model("risk_classifier", models_dir=tmp_path)

    def test_never_overwrites_a_version(self, classifier, tmp_path):
        model, meta = classifier
        train.save_model(model, meta, models_dir=tmp_path)
        with pytest.raises(FileExistsError):
            train.save_model(model, meta, models_dir=tmp_path)

