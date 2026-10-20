"""
Tests for the V3.1 clinical feature/score contract.
Verifies the derived-score keys match the Data Platform scoring engine and that
both stored `nds_v31_derived.result` shapes normalise correctly.
"""

from src.neps_v31_clinical_contract import (
    CORE73_FIELD_NAMES,
    DERIVED_SCORE_KEYS,
    DERIVED_SCORE_LABELS,
    normalize_derived_result,
)


class TestCore73Contract:
    def test_core73_has_73_unique_items(self):
        assert len(CORE73_FIELD_NAMES) == 73
        assert len(set(CORE73_FIELD_NAMES)) == 73

    def test_derived_score_keys_and_labels_are_consistent(self):
        assert set(DERIVED_SCORE_KEYS) == set(DERIVED_SCORE_LABELS)
        assert "NSES" in DERIVED_SCORE_KEYS


class TestNormalizeDerivedResult:
    def test_flat_shape(self):
        result = {"EPI": 62.5, "ESI": None, "NSES": 48.0, "K6_total": 12}
        scores, metadata = normalize_derived_result(result)
        assert scores["EPI"] == 62.5
        assert scores["ESI"] is None
        assert scores["NSES"] == 48.0
        assert scores["K6_total"] == 12.0
        assert metadata == {}
        assert set(scores) == set(DERIVED_SCORE_KEYS)

    def test_nested_fraction_shape(self):
        result = {
            "algorithm_version": "NEPS-V3.1-...",
            "provisional": True,
            "scores": {
                "NSES": {"value": {"numerator": 5, "denominator": 8}, "issues": []},
                "EPI": {"value": None, "issues": [{"field": "EPI", "state": "not_applicable"}]},
            },
        }
        scores, metadata = normalize_derived_result(result)
        assert scores["NSES"] == 0.625
        assert scores["EPI"] is None
        assert scores["ESI"] is None
        assert metadata["provisional"] is True
        assert metadata["algorithm_version"] == "NEPS-V3.1-..."

    def test_json_string_shape(self):
        scores, _ = normalize_derived_result('{"NSES": 33.0}')
        assert scores["NSES"] == 33.0

    def test_null_and_empty_inputs(self):
        for value in (None, "", {}, "not-json"):
            scores, metadata = normalize_derived_result(value)
            assert set(scores) == set(DERIVED_SCORE_KEYS)
            assert all(v is None for v in scores.values())
            assert metadata == {}

    def test_numeric_string_is_coerced(self):
        scores, _ = normalize_derived_result({"ESI": "70.0"})
        assert scores["ESI"] == 70.0
