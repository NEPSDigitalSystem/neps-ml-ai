"""
Tests for V31Repository dataset assembly.

`_assemble` is a pure function so it can be exercised without a live database.
It must accept both stored shapes of `nds_v31_derived.result` (flat floats and
nested fraction objects) and produce a single ML-ready DataFrame.
"""

from app.data.v31_repository import METADATA_COLUMNS, _assemble
from src.neps_v31_clinical_contract import DERIVED_SCORE_KEYS

import pandas as pd

FLAT_ROW = {
    "source_id": "v31_NEPS-GHA-0001",
    "payload": {
        "study_id": "NEPS-GHA-0001",
        "country": "Ghana",
        "k6_1": "3",
        "epi_expectations_1": "4",
    },
    "country": "Ghana",
    "observed_at": "2026-10-20T00:00:00+00:00",
    "source_version": "v31-mock-api",
    "contract_version": "3.1",
    "nds_participant_id": "NDS-GH-NEPS-GHA-0001",
    "derived": {"EPI": 62.5, "NSES": 50.0, "K6_total": 12.0},
    "calculation_version": "v1",
}

NESTED_ROW = {
    "source_id": "v31_NEPS-TAN-0101",
    "payload": '{"study_id": "NEPS-TAN-0101", "country": "Tanzania"}',
    "country": "Tanzania",
    "observed_at": "2026-10-20T00:00:00+00:00",
    "source_version": "mock-redcap-v3.1",
    "contract_version": "3.1",
    "nds_participant_id": "NDS-TA-NEPS-TAN-0101",
    "derived": {
        "algorithm_version": "NEPS-V3.1-provisional",
        "provisional": True,
        "scores": {"NSES": {"value": {"numerator": 5, "denominator": 8}, "issues": []}},
    },
    "calculation_version": "v1",
}


class TestAssemble:
    def test_empty_rows_returns_empty_frame(self):
        assert _assemble([]).empty

    def test_columns_include_metadata_features_and_scores(self):
        dataset = _assemble([FLAT_ROW, NESTED_ROW])

        assert len(dataset) == 2
        for col in METADATA_COLUMNS:
            assert col in dataset.columns, f"missing metadata column {col}"
        for key in DERIVED_SCORE_KEYS:
            assert f"score_{key}" in dataset.columns
        assert "k6_1" in dataset.columns
        assert "epi_expectations_1" in dataset.columns

    def test_country_is_not_duplicated(self):
        dataset = _assemble([FLAT_ROW, NESTED_ROW])
        assert list(dataset.columns).count("country") == 1

    def test_study_id_extracted_from_dict_and_json_payload(self):
        dataset = _assemble([FLAT_ROW, NESTED_ROW])
        assert list(dataset["study_id"]) == ["NEPS-GHA-0001", "NEPS-TAN-0101"]

    def test_flat_scores_are_floats(self):
        dataset = _assemble([FLAT_ROW])
        assert dataset.loc[0, "score_NSES"] == 50.0
        assert dataset.loc[0, "score_EPI"] == 62.5
        assert dataset.loc[0, "score_K6_total"] == 12.0

    def test_nested_fraction_scores_are_normalised(self):
        dataset = _assemble([NESTED_ROW])
        assert dataset.loc[0, "score_NSES"] == 0.625
        # EPI absent from the nested payload -> null, not a crash.
        assert pd.isna(dataset.loc[0, "score_EPI"])
