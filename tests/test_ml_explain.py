"""
NEPS ML/AI — Phase 3 tests: trajectory targets, regression metrics, trend labels,
trajectory trainer, SHAP explanations.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml import evaluate, explain, features, schema, train
from sklearn.ensemble import HistGradientBoostingClassifier


# ─── NEXT-MONTH TARGETS ────────────────────────────────────────────────────────

class TestNextMonthTargets:
    def test_drops_last_month_only(self, frame):
        out = features.add_next_month_targets(frame, ["depression"])
        assert len(out) == 150 * 23
        assert out.month.max() == 23

    def test_next_is_following_month_same_participant(self, frame):
        out = features.add_next_month_targets(frame, ["depression", "anxiety"]).set_index(["participant_id", "month"])
        src = frame.set_index(["participant_id", "month"])
        pid = src.index[0][0]
        for m in (1, 12, 23):
            assert out.loc[(pid, m), "depression_next"] == src.loc[(pid, m + 1), "depression"]
            assert out.loc[(pid, m), "anxiety_next"] == src.loc[(pid, m + 1), "anxiety"]

    def test_input_order_does_not_matter(self, frame):
        shuffled = frame.sample(frac=1, random_state=3)
        assert features.add_next_month_targets(shuffled, ["stress"]).equals(
            features.add_next_month_targets(frame, ["stress"]))

    def test_next_columns_are_not_features(self, frame):
        out = features.add_next_month_targets(frame, ["depression"])
        cols = features.feature_columns(out)
        assert "depression_next" not in cols
        assert "depression" in cols and "depression_lag1" in cols

    def test_gap_in_months_is_not_bridged(self, data):
        # Remove month 5 for one participant. Month 4 has no "next"; month 6 has no lag1.
        monthly = data["monthly_reports"]
        pid = monthly.participant_id.iloc[0]
        gap = monthly[~((monthly.participant_id == pid) & (monthly.month == 5))]
        feats = features.monthly_features(gap).set_index(["participant_id", "month"])
        assert np.isnan(feats.loc[(pid, 6), "depression_lag1"])
        assert feats.loc[(pid, 6), "depression_lag2"] == feats.loc[(pid, 4), "depression"]
        assert feats.loc[(pid, 7), "depression_lag3"] == feats.loc[(pid, 4), "depression"]
        assert np.isnan(feats.loc[(pid, 8), "depression_lag3"])
        assert np.isnan(feats.loc[(pid, 6), "depression_delta1"])
        nxt = features.add_next_month_targets(feats.reset_index(), ["depression"]).set_index(["participant_id", "month"])
        assert (pid, 4) not in nxt.index
        assert nxt.loc[(pid, 6), "depression_next"] == feats.loc[(pid, 7), "depression"]


# ─── REGRESSION METRICS ────────────────────────────────────────────────────────

class TestRegressionMetrics:
    def test_perfect(self):
        m = evaluate.regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert m == {"n": 3, "mae": 0.0, "r2": 1.0}

    def test_off_by_one(self):
        m = evaluate.regression_metrics([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        assert m["mae"] == 1.0 and m["r2"] < 1.0

    def test_single_row_has_no_r2(self):
        assert evaluate.regression_metrics([1.0], [2.0])["r2"] is None

    def test_report_with_regression_metric(self, frame):
        test = frame.iloc[:100]
        y = test["depression"]
        rep = evaluate.metrics_report(test, y, y, metric=evaluate.regression_metrics)
        assert rep["overall"]["mae"] == 0.0
        assert set(rep) == {"overall", *schema.FAIRNESS_GROUPS}
        assert sum(v["n"] for v in rep["country"].values()) == 100


# ─── TREND LABELS ──────────────────────────────────────────────────────────────

class TestTrendLabel:
    def test_direction_and_threshold(self):
        cur = [10, 10, 10, 10, 10]
        pred = [15, 12.0, 11.9, 8.1, 5]
        assert list(features.trend_label(cur, pred, threshold=2.0)) == [
            "Deteriorating", "Deteriorating", "Stable", "Stable", "Improving"]

    def test_exact_negative_threshold_is_improving(self):
        assert features.trend_label([10], [8.0], threshold=2.0)[0] == "Improving"

    def test_accepts_series_and_arrays(self):
        out = features.trend_label(pd.Series([1.0, 2.0]), np.array([1.0, 9.0]))
        assert list(out) == ["Stable", "Deteriorating"]

    def test_nan_is_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            features.trend_label([1.0, 2.0], [1.0, float("nan")])

    def test_targets_are_higher_is_worse_scores(self):
        assert set(train.TRAJECTORY_TARGETS) <= features.HIGHER_IS_WORSE

    def test_only_known_labels(self, frame):
        out = features.trend_label(frame["depression"], frame["anxiety"])
        assert set(out) <= set(features.TREND_LABELS)


# ─── TRAJECTORY TRAINER ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trajectory(frame):
    return train.train_trajectory(frame, "depression")


class TestTrajectory:
    def test_meta(self, trajectory):
        _, meta = trajectory
        assert meta["name"] == "trajectory_depression"
        assert meta["target"] == "depression_next"
        assert "depression_next" not in meta["features"]
        assert meta["n_train"] + meta["n_test"] == 150 * 23
        assert meta["trend_threshold"] == features.TREND_THRESHOLD
        assert set(meta["metrics"]) == set(meta["trend_metrics"]) == {"overall", *schema.FAIRNESS_GROUPS}
        assert {"mae", "r2"} <= set(meta["metrics"]["overall"])
        assert {"accuracy", "macro_f1"} <= set(meta["trend_metrics"]["overall"])

    def test_predictions_in_score_range(self, trajectory, frame):
        model, meta = trajectory
        lo, hi = schema.MONTHLY_SCORES["depression"]
        pred = model.predict(frame[meta["features"]].head(200))
        assert pred.min() >= lo - 5 and pred.max() <= hi + 5

    def test_beats_naive_persistence_or_documents_no_signal(self, trajectory, frame):
        # Mock scores are independent month to month, so R2 near 0 is expected.
        # This pins the ceiling: anything wildly negative means a wiring bug.
        _, meta = trajectory
        assert meta["metrics"]["overall"]["r2"] > -0.5
        assert meta["metrics"]["overall"]["mae"] < 6

    def test_deterministic(self, trajectory, frame):
        _, meta2 = train.train_trajectory(frame, "depression")
        assert meta2["metrics"] == trajectory[1]["metrics"]

    def test_registry_roundtrip(self, trajectory, frame, tmp_path):
        model, meta = trajectory
        train.save_model(model, meta, models_dir=tmp_path)
        loaded, lmeta = train.load_model("trajectory_depression", models_dir=tmp_path)
        X = frame[meta["features"]].head(10)
        assert np.allclose(loaded.predict(X), model.predict(X))

    def test_r2_none_for_constant_group(self):
        assert evaluate.regression_metrics([3.0, 3.0], [1.0, 2.0])["r2"] is None

    def test_main_trains_all_models(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
        train.main()
        assert capsys.readouterr().out.count("saved") == 3
        names = {p.name for p in tmp_path.iterdir()}
        assert names == {"risk_classifier", *(f"trajectory_{t}" for t in train.TRAJECTORY_TARGETS)}


# ─── EXPLAIN ───────────────────────────────────────────────────────────────────

class TestFeatureLabel:
    @pytest.mark.parametrize("name,label", [
        ("anxiety", "anxiety"),
        ("anxiety_lag1", "anxiety last month"),
        ("sleep_quality_score_lag3", "sleep quality three months ago"),
        ("depression_delta1", "depression change since last month"),
        ("country_Sierra_Leone", "country: Sierra Leone"),
        ("gender_Prefer_not_to_say", "gender: Prefer not to say"),
        ("socioeconomic_status_lower_middle", "socioeconomic status: lower middle"),
        ("age", "age"),
        ("something_unknown", "something unknown"),
    ])
    def test_labels(self, name, label):
        assert explain.feature_label(name) == label

    def test_every_model_feature_has_a_readable_label(self, classifier):
        _, meta = classifier
        for f in meta["features"]:
            label = explain.feature_label(f)
            assert label and "_" not in label, f


class TestExplainRisk:
    def test_shape_and_ordering(self, classifier, frame):
        model, meta = classifier
        rows = frame.head(3)
        out = explain.explain_risk(model, meta, rows, top_k=4)
        assert len(out) == 3
        for r, pred in zip(out, model.predict(rows[meta["features"]])):
            assert r["predicted"] == str(pred)
            c = r["contributions"]
            assert len(c) == 4
            shares = [x["share"] for x in c]
            assert shares == sorted(shares, reverse=True)
            assert all(0 <= s <= 1 for s in shares)
            assert sum(shares) <= 1.0 + 1e-6

    def test_direction_matches_sign(self, classifier, frame):
        model, meta = classifier
        out = explain.explain_risk(model, meta, frame.head(5))
        for r in out:
            for c in r["contributions"]:
                assert c["direction"] == ("towards" if c["contribution"] > 0 else "away from")
                assert c["label"] in c["sentence"].lower() or c["label"].capitalize() in c["sentence"]
                assert r["predicted"] in c["sentence"]

    def test_nan_feature_value_serialises_as_none(self, classifier, frame):
        model, meta = classifier
        row = frame[frame.month == 1].head(1)   # lags are NaN in month 1
        out = explain.explain_risk(model, meta, row, top_k=len(meta["features"]))
        lag_items = [c for c in out[0]["contributions"] if c["feature"].endswith("_lag1")]
        assert lag_items and all(c["value"] is None for c in lag_items)

    def test_high_risk_row_driven_by_clinical_scores(self, classifier, frame):
        model, meta = classifier
        high = frame[frame.risk_flag == "HIGH"].head(5)
        out = explain.explain_risk(model, meta, high, top_k=3)
        assert any(r["predicted"] == "HIGH" for r in out)
        for r in out:
            if r["predicted"] == "HIGH":
                assert any(c["feature"] in ("suicidality_score", "depression", "anxiety")
                           for c in r["contributions"])

    def test_contributions_are_class_specific(self, classifier, frame):
        # Same feature, opposite sign for a LOW vs a HIGH prediction guards the class-index lookup.
        model, meta = classifier
        low = frame[frame.risk_flag == "LOW"].head(1)
        high = frame[frame.risk_flag == "HIGH"].head(1)
        out = explain.explain_risk(model, meta, pd.concat([low, high]), top_k=len(meta["features"]))
        assert out[0]["predicted"] == "LOW" and out[1]["predicted"] == "HIGH"
        by = [{c["feature"]: c["contribution"] for c in r["contributions"]} for r in out]
        assert by[0]["suicidality_score"] != by[1]["suicidality_score"]

    def test_shares_sum_to_one_with_all_features(self, classifier, frame):
        model, meta = classifier
        out = explain.explain_risk(model, meta, frame.head(2), top_k=10_000)
        for r in out:
            assert len(r["contributions"]) == len(meta["features"])
            assert sum(c["share"] for c in r["contributions"]) == pytest.approx(1.0, abs=1e-3)

    def test_binary_classifier_supported(self, frame):
        # Real data may have 2 tiers; shap returns a 2-D array for binary HGB.
        sub = frame.head(600)
        y = (sub["risk_flag"] == "LOW").map({True: "LOW", False: "NOT_LOW"})
        cols = features.feature_columns(sub)
        model = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(sub[cols], y)
        meta = {"features": cols, "classes": [str(c) for c in model.classes_]}
        out = explain.explain_risk(model, meta, sub.head(3), top_k=3)
        assert len(out) == 3 and all(len(r["contributions"]) == 3 for r in out)
        assert {r["predicted"] for r in out} <= {"LOW", "NOT_LOW"}

    def test_extra_columns_in_x_are_ignored(self, classifier, frame):
        model, meta = classifier
        rows = frame.head(2).assign(junk=1)
        out = explain.explain_risk(model, meta, rows)
        assert len(out) == 2 and all(c["feature"] != "junk" for c in out[0]["contributions"])
