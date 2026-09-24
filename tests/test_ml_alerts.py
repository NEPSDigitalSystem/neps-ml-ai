"""
NEPS ML/AI — Phase 4 tests: fairness audit and early-warning alerts.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml import alerts, fairness, features, schema, train


class StubModel:
    """Deterministic classifier stand-in: fixed predictions and probabilities."""
    def __init__(self, classes, preds, p_pos):
        self.classes_ = np.array(classes)
        self._preds, self._p = np.array(preds), np.array(p_pos, dtype=float)

    def predict(self, X):
        return self._preds[: len(X)]

    def predict_proba(self, X):
        # positive mass on the last positive class, remainder on the first negative class
        n = len(X)
        proba = np.zeros((n, len(self.classes_)))
        pos = [i for i, c in enumerate(self.classes_) if c in ("HIGH", "CRITICAL")]
        neg = [i for i in range(len(self.classes_)) if i not in pos]
        if pos:
            proba[:, pos[-1]] = self._p[:n]
        proba[:, neg[0]] += 1 - (self._p[:n] if pos else 0)
        return proba


def _quiet(frame, n):
    rows = frame.head(n).copy()
    rows["suicidality_score"] = 0.0
    rows["suicidality_screening"] = "No"
    return rows


# ─── FAIRNESS ──────────────────────────────────────────────────────────────────

def _frame(groups, y_true, y_pred):
    f = pd.DataFrame({"country": groups, "gender": "x", "site": "s"})
    return f, pd.Series(y_true), pd.Series(y_pred)


class TestRates:
    def test_hand_computed(self):
        y_true = pd.Series(["HIGH", "HIGH", "LOW", "LOW", "CRITICAL"])
        y_pred = pd.Series(["HIGH", "LOW", "HIGH", "LOW", "CRITICAL"])
        r = fairness._rates(y_true, y_pred)
        assert r == {"n": 5, "positives": 3, "selection_rate": 0.6, "tpr": round(2 / 3, 4), "fpr": 0.5}

    def test_no_positives_gives_none_tpr(self):
        r = fairness._rates(pd.Series(["LOW", "LOW"]), pd.Series(["LOW", "HIGH"]))
        assert r["tpr"] is None and r["fpr"] == 0.5

    def test_all_positives_gives_none_fpr(self):
        r = fairness._rates(pd.Series(["HIGH"]), pd.Series(["HIGH"]))
        assert r["fpr"] is None and r["tpr"] == 1.0


class TestGap:
    def test_gap_ignores_none(self):
        assert fairness._gap([0.2, None, 0.5]) == 0.3

    def test_single_value_is_none(self):
        assert fairness._gap([0.2, None]) is None

    def test_all_none_or_empty_is_none(self):
        assert fairness._gap([None, None]) is None
        assert fairness._gap([]) is None


class TestFairnessReport:
    def test_perfect_predictor_has_zero_gaps_where_defined(self):
        f, yt, yp = _frame(["A"] * 4 + ["B"] * 4, ["HIGH", "LOW"] * 4, ["HIGH", "LOW"] * 4)
        rep = fairness.fairness_report(f, yt, yp, groups=["country"])
        assert rep["country"]["demographic_parity_gap"] == 0.0
        assert rep["country"]["equal_opportunity_gap"] == 0.0
        assert rep["country"]["warn"] is False

    def test_biased_predictor_flagged(self):
        # B is never flagged even when truly HIGH
        f, yt, yp = _frame(["A"] * 4 + ["B"] * 4, ["HIGH", "LOW"] * 4, ["HIGH", "LOW"] * 2 + ["LOW"] * 4)
        rep = fairness.fairness_report(f, yt, yp, groups=["country"])
        assert rep["country"]["values"]["A"]["tpr"] == 1.0
        assert rep["country"]["values"]["B"]["tpr"] == 0.0
        assert rep["country"]["equal_opportunity_gap"] == 1.0
        assert rep["country"]["demographic_parity_gap"] == 0.5
        assert rep["country"]["warn"] is True

    def test_warn_threshold_boundary(self):
        # selection rates 0.5 vs 0.4 -> gap exactly 0.1, not > GAP_WARN
        f, yt, yp = _frame(["A"] * 10 + ["B"] * 10, ["LOW"] * 20,
                           ["HIGH"] * 5 + ["LOW"] * 5 + ["HIGH"] * 4 + ["LOW"] * 6)
        rep = fairness.fairness_report(f, yt, yp, groups=["country"])
        assert rep["country"]["demographic_parity_gap"] == pytest.approx(0.1)
        assert rep["country"]["warn"] is False

    def test_missing_group_value_is_audited_not_dropped(self):
        f, yt, yp = _frame(["A", "A", None, None], ["HIGH", "LOW", "HIGH", "LOW"], ["HIGH", "LOW", "LOW", "LOW"])
        rep = fairness.fairness_report(f, yt, yp, groups=["country"])
        assert sum(v["n"] for v in rep["country"]["values"].values()) == 4
        assert "missing" in rep["country"]["values"]
        assert rep["country"]["equal_opportunity_gap"] == 1.0

    def test_missing_group_column_raises(self):
        f, yt, yp = _frame(["A"], ["LOW"], ["LOW"])
        with pytest.raises(KeyError):
            fairness.fairness_report(f, yt, yp, groups=["nope"])

    def test_ignores_input_index(self):
        f, yt, yp = _frame(["A", "B"], ["HIGH", "LOW"], ["HIGH", "LOW"])
        f.index = [10, 20]
        rep = fairness.fairness_report(f, yt.to_numpy(), list(yp), groups=["country"])
        assert rep["overall"]["n"] == 2

    def test_default_groups_and_positive_classes(self, frame, classifier):
        model, meta = classifier
        rep = meta["fairness"]
        assert rep["positive_classes"] == ["HIGH", "CRITICAL"]
        assert set(rep) == {"positive_classes", "overall", *schema.FAIRNESS_GROUPS}
        for g in schema.FAIRNESS_GROUPS:
            assert set(rep[g]["values"]) <= set(frame[g].astype(str).unique())
            assert sum(v["n"] for v in rep[g]["values"].values()) == meta["n_test"]
            assert {"demographic_parity_gap", "equal_opportunity_gap", "warn"} <= set(rep[g])

    def test_site_is_a_fairness_group_and_not_a_raw_feature(self, frame):
        assert "site" in schema.FAIRNESS_GROUPS
        cols = features.feature_columns(frame)
        assert "site" not in cols and any(c.startswith("site_") for c in cols)


# ─── ALERTS ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trajectory(frame):
    return train.train_trajectory(frame, "depression")


@pytest.fixture(scope="module")
def alert_table(frame, classifier, trajectory):
    return alerts.generate_alerts(frame, classifier, trajectory)


class TestGenerateAlerts:
    def test_one_row_per_input_with_expected_columns(self, alert_table, frame):
        assert len(alert_table) == len(frame)
        assert {"participant_id", "month", "risk_pred", "p_high", "suicidality_score",
                "trend", "priority", "reasons", "interventions"} <= set(alert_table.columns)
        assert set(alert_table.priority) <= {"P0", "P1", "P2", "none"}

    def test_crisis_rule_fires_regardless_of_model(self, frame, classifier):
        rows = frame.head(3).copy()
        rows["suicidality_score"] = [alerts.SUICIDALITY_CRISIS, alerts.SUICIDALITY_CRISIS - 0.1, 0.0]
        rows["suicidality_screening"] = ["No", "No", "Active plan"]
        out = alerts.generate_alerts(rows, classifier)
        assert list(out.priority[[0, 2]]) == ["P0", "P0"]
        assert out.priority[1] != "P0"
        assert "suicidality crisis rule" in out.reasons[0]
        assert out.interventions[0] == ["safeguarding referral", "counselling"]

    def test_missing_screening_column_is_an_error(self, frame, classifier):
        rows = frame.head(2).drop(columns="suicidality_screening")
        with pytest.raises(KeyError):
            alerts.generate_alerts(rows, classifier)

    def test_screening_is_in_the_schema_contract(self):
        assert "suicidality_screening" in schema.CATEGORICALS["monthly_reports"]
        assert set(alerts.CRISIS_SCREENING) <= set(schema.SUICIDALITY_SCREENING)

    def test_p1_from_predicted_class_or_probability(self, frame, classifier):
        _, meta = classifier
        rows = _quiet(frame, 4)
        stub = StubModel(["CRITICAL", "HIGH", "LOW", "MEDIUM"],
                         preds=["HIGH", "LOW", "LOW", "LOW"],
                         p_pos=[0.9, 0.5, 0.49, 0.0])        # class, threshold, just under, nothing
        out = alerts.generate_alerts(rows, (stub, meta))
        assert list(out.priority) == ["P1", "P1", "none", "none"]
        assert out.reasons[0] == ["model high risk"] and out.reasons[1] == ["model high risk"]
        assert out.interventions[0] == ["counselling", "follow-up assessment"]
        assert list(out.p_high) == [0.9, 0.5, 0.49, 0.0]

    def test_medium_gives_p2_with_interventions(self, frame, classifier):
        _, meta = classifier
        stub = StubModel(["CRITICAL", "HIGH", "LOW", "MEDIUM"], preds=["MEDIUM"], p_pos=[0.0])
        out = alerts.generate_alerts(_quiet(frame, 1), (stub, meta))
        assert out.priority[0] == "P2"
        assert out.reasons[0] == ["deteriorating or medium"]
        assert out.interventions[0] == ["peer support", "follow-up assessment"]

    def test_reasons_accumulate_in_priority_order(self, frame, classifier, trajectory):
        _, meta = classifier
        rows = _quiet(frame, 1)
        rows["suicidality_score"] = 9.0
        stub = StubModel(["CRITICAL", "HIGH", "LOW", "MEDIUM"], preds=["MEDIUM"], p_pos=[0.7])
        out = alerts.generate_alerts(rows, (stub, meta))
        assert out.priority[0] == "P0"
        assert out.reasons[0] == ["suicidality crisis rule", "model high risk", "deteriorating or medium"]

    def test_crisis_with_quiet_model_has_single_reason(self, frame, classifier):
        _, meta = classifier
        rows = _quiet(frame, 1)
        rows["suicidality_screening"] = "Previous attempt"
        stub = StubModel(["CRITICAL", "HIGH", "LOW", "MEDIUM"], preds=["LOW"], p_pos=[0.0])
        out = alerts.generate_alerts(rows, (stub, meta))
        assert out.priority[0] == "P0" and out.reasons[0] == ["suicidality crisis rule"]

    def test_p_positive_handles_missing_positive_classes(self, frame, classifier):
        _, meta = classifier
        X = frame[meta["features"]].head(2)
        two = StubModel(["HIGH", "LOW"], preds=["LOW", "LOW"], p_pos=[0.3, 0.8])
        assert list(alerts._p_positive(two, X)) == [0.3, 0.8]
        none = StubModel(["LOW", "MEDIUM"], preds=["LOW", "LOW"], p_pos=[0.3, 0.8])
        assert list(alerts._p_positive(none, X)) == [0.0, 0.0]

    def test_non_default_index_is_safe(self, frame, classifier, trajectory):
        rows = frame.head(3).copy()
        rows.index = [10, 20, 30]
        out = alerts.generate_alerts(rows, classifier, trajectory)
        assert list(out.index) == [0, 1, 2]
        assert out.priority.notna().all() and out.trend.notna().all()

    def test_deteriorating_trend_gives_p2(self, frame, classifier, trajectory):
        rows = frame[frame.risk_flag == "LOW"].head(50).copy()
        rows["suicidality_score"] = 0.0
        rows["suicidality_screening"] = "No"
        out = alerts.generate_alerts(rows, classifier, trajectory)
        det = out[out.trend == "Deteriorating"]
        assert len(det) > 0, "need at least one deteriorating row for this test to mean anything"
        assert (det.priority.isin(["P2", "P1"])).all()
        assert all("deteriorating or medium" in r for r in det.reasons)

    def test_medium_gives_p2(self, frame, classifier):
        rows = frame[frame.risk_flag == "MEDIUM"].head(20).copy()
        rows["suicidality_score"] = 0.0
        rows["suicidality_screening"] = "No"
        out = alerts.generate_alerts(rows, classifier)
        med = out[out.risk_pred == "MEDIUM"]
        assert len(med) > 0
        assert (med.priority == "P2").all()

    def test_no_trajectory_means_unknown_trend(self, frame, classifier):
        out = alerts.generate_alerts(frame.head(3), classifier)
        assert (out.trend == "unknown").all()

    def test_trend_uses_trajectory_threshold(self, frame, classifier, trajectory):
        t_model, t_meta = trajectory
        rows = frame.head(30)
        out = alerts.generate_alerts(rows, classifier, trajectory)
        nxt = t_model.predict(rows[t_meta["features"]])
        expected = features.trend_label(rows["depression"], nxt, t_meta["trend_threshold"])
        assert list(out.trend) == list(expected)

    def test_priority_precedence_p0_over_p1_over_p2(self, frame, classifier):
        rows = frame[frame.risk_flag == "MEDIUM"].head(1).copy()
        rows["suicidality_score"] = 9.0
        out = alerts.generate_alerts(rows, classifier)
        assert out.priority[0] == "P0"
        assert out.reasons[0][0] == "suicidality crisis rule"

    def test_interventions_never_missing(self, alert_table):
        assert alert_table.interventions.map(lambda x: isinstance(x, list)).all()
        assert set(alerts.INTERVENTIONS) == {"P0", "P1", "P2", "none"}

    def test_none_rows_have_no_reasons_or_interventions(self, alert_table):
        quiet = alert_table[alert_table.priority == "none"]
        assert len(quiet) > 0
        assert all(r == [] for r in quiet.reasons)
        assert all(i == [] for i in quiet.interventions)

    def test_every_alert_has_a_reason(self, alert_table):
        loud = alert_table[alert_table.priority != "none"]
        assert all(len(r) > 0 for r in loud.reasons)

    def test_mock_p0_matches_mock_high_critical(self, alert_table, frame):
        # Mock risk_flag is a threshold on suicidality, so the crisis rule should
        # recover HIGH/CRITICAL almost exactly. Pins the rule threshold to the data.
        actual = frame.risk_flag.isin(["HIGH", "CRITICAL"]).to_numpy()
        agree = ((alert_table.priority == "P0").to_numpy() == actual).mean()
        assert agree > 0.99
