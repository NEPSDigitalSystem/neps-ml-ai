"""
NEPS ML/AI — Phase 1 tests: schema.validate() and the CSV loader.
"""

import pandas as pd
import pytest

from app.ml import schema
from app.ml.loader import load_datasets, SUBSTANCE_USE_NONE, DEFAULT_DATA_DIR
from app.ml.schema import SchemaError

# Pinned to mock REDCap v0.3.0. A mock regeneration is expected to break these.
EXPECTED_ROWS = {
    "participants": 150,
    "monthly_reports": 3600,
    "comprehensive_waves": 600,
    "nlp_responses": 2000,
}


@pytest.fixture(scope="module")
def data():
    return load_datasets()


@pytest.fixture
def monthly(data):
    return data["monthly_reports"].copy()


@pytest.fixture
def nlp(data):
    return data["nlp_responses"].copy()


@pytest.fixture
def ids(data):
    return set(data["participants"]["participant_id"])


# ─── LOADER ────────────────────────────────────────────────────────────────────

class TestLoader:
    def test_loads_all_four_with_expected_rows(self, data):
        assert set(data) == set(EXPECTED_ROWS)
        for name, n in EXPECTED_ROWS.items():
            assert len(data[name]) == n, name

    def test_substance_use_filled(self, data):
        col = data["monthly_reports"]["substance_use"]
        assert col.isna().sum() == 0
        assert (col == SUBSTANCE_USE_NONE).sum() == 1215
        assert set(col.unique()) == {"none", "Alcohol", "Cannabis", "Other"}

    def test_fill_survives_csv_round_trip(self, data, tmp_path):
        # "None" is a pandas NA sentinel; "none" is not
        data["monthly_reports"].to_csv(tmp_path / "m.csv", index=False)
        assert pd.read_csv(tmp_path / "m.csv")["substance_use"].isna().sum() == 0

    def test_fill_does_not_touch_other_columns(self, data):
        raw = pd.read_csv(DEFAULT_DATA_DIR / "monthly_reports.csv")
        assert raw.drop(columns="substance_use").equals(
            data["monthly_reports"].drop(columns="substance_use"))

    def test_unknown_source_rejected(self):
        with pytest.raises(NotImplementedError):
            load_datasets(source="redcap")

    def test_missing_file_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_datasets(data_dir=tmp_path)

    def test_loader_rejects_bad_data(self, tmp_path, data):
        for name, df in data.items():
            df.to_csv(tmp_path / f"{name}.csv", index=False)
        bad = data["monthly_reports"].copy()
        bad.loc[0, "anxiety"] = 999
        bad.to_csv(tmp_path / "monthly_reports.csv", index=False)
        with pytest.raises(SchemaError, match="monthly_reports.anxiety"):
            load_datasets(data_dir=tmp_path)

    def test_loader_rejects_orphan_in_any_dataset(self, tmp_path, data):
        for name, df in data.items():
            df.to_csv(tmp_path / f"{name}.csv", index=False)
        bad = data["comprehensive_waves"].copy()
        bad.loc[0, "participant_id"] = "NEPS-GHA-9999"
        bad.to_csv(tmp_path / "comprehensive_waves.csv", index=False)
        with pytest.raises(SchemaError, match="comprehensive_waves.participant_id: 1 ids"):
            load_datasets(data_dir=tmp_path)


# ─── VALIDATE: VALID INPUT ─────────────────────────────────────────────────────

class TestValidateValid:
    def test_all_mock_datasets_pass(self, data, ids):
        for name, df in data.items():
            assert schema.validate(name, df, participant_ids=ids) is df

    def test_unknown_dataset_name(self):
        with pytest.raises(SchemaError, match="unknown dataset"):
            schema.validate("nope", pd.DataFrame())

    def test_empty_frame_fails(self, monthly, ids):
        with pytest.raises(SchemaError, match="monthly_reports.\\*: no rows"):
            schema.validate("monthly_reports", monthly.iloc[0:0], ids)

    def test_first_violation_wins(self, monthly, ids):
        monthly.loc[0, "participant_id"] = None   # rule 1
        monthly.loc[1, "anxiety"] = 999           # rule 2
        with pytest.raises(SchemaError, match="participant_id: 1 null keys"):
            schema.validate("monthly_reports", monthly, ids)


# ─── VALIDATE: ONE FAILURE PER RULE ────────────────────────────────────────────

class TestRule1Keys:
    def test_null_key(self, monthly, ids):
        monthly.loc[0, "participant_id"] = None
        with pytest.raises(SchemaError, match="participant_id: 1 null keys"):
            schema.validate("monthly_reports", monthly, ids)

    def test_duplicate_key(self, monthly, ids):
        dup = pd.concat([monthly, monthly.iloc[[0]]], ignore_index=True)
        with pytest.raises(SchemaError, match="participant_id\\+month: 1 duplicate keys"):
            schema.validate("monthly_reports", dup, ids)

    def test_missing_key_column(self, monthly, ids):
        with pytest.raises(SchemaError, match="month: key column missing"):
            schema.validate("monthly_reports", monthly.drop(columns="month"), ids)

    def test_nlp_key_is_response_id(self, nlp, ids):
        # (participant_id, month) is legitimately non-unique in nlp_responses
        assert nlp.duplicated(["participant_id", "month"]).any()
        schema.validate("nlp_responses", nlp, ids)
        nlp.loc[1, "response_id"] = nlp.loc[0, "response_id"]
        with pytest.raises(SchemaError, match="response_id: 1 duplicate"):
            schema.validate("nlp_responses", nlp, ids)


class TestRule2Ranges:
    @pytest.mark.parametrize("col", list(schema.MONTHLY_SCORES))
    def test_above_max(self, monthly, ids, col):
        monthly.loc[0, col] = schema.MONTHLY_SCORES[col][1] + 0.1
        with pytest.raises(SchemaError, match=f"monthly_reports.{col}: 1 values outside"):
            schema.validate("monthly_reports", monthly, ids)

    def test_below_min(self, monthly, ids):
        monthly.loc[0, "anxiety"] = -0.1
        with pytest.raises(SchemaError, match="anxiety: 1 values outside"):
            schema.validate("monthly_reports", monthly, ids)

    def test_boundary_values_pass(self, monthly, ids):
        monthly.loc[0, "anxiety"] = 0
        monthly.loc[1, "anxiety"] = 30
        schema.validate("monthly_reports", monthly, ids)

    def test_null_score(self, monthly, ids):
        monthly.loc[0, "depression"] = None
        with pytest.raises(SchemaError, match="depression: 1 null or non-numeric"):
            schema.validate("monthly_reports", monthly, ids)

    def test_string_score_is_schema_error_not_typeerror(self, monthly, ids):
        monthly["depression"] = monthly["depression"].astype(object)
        monthly.loc[0, "depression"] = "N/A"
        with pytest.raises(SchemaError, match="depression: 1 null or non-numeric"):
            schema.validate("monthly_reports", monthly, ids)

    def test_numeric_strings_are_accepted(self, monthly, ids):
        monthly["anxiety"] = monthly["anxiety"].astype(str)
        schema.validate("monthly_reports", monthly, ids)

    def test_missing_score_column(self, monthly, ids):
        with pytest.raises(SchemaError, match="stress: score column missing"):
            schema.validate("monthly_reports", monthly.drop(columns="stress"), ids)


class TestRule3Categoricals:
    def test_unknown_risk_flag(self, monthly, ids):
        monthly.loc[0, "risk_flag"] = "Elevated"
        with pytest.raises(SchemaError, match="risk_flag: unknown values \\['Elevated'\\]"):
            schema.validate("monthly_reports", monthly, ids)

    def test_null_risk_flag_fails(self, monthly, ids):
        monthly.loc[0, "risk_flag"] = None
        with pytest.raises(SchemaError, match="risk_flag: 1 nulls"):
            schema.validate("monthly_reports", monthly, ids)

    def test_case_matters(self, monthly, ids):
        monthly.loc[0, "risk_flag"] = "low"
        with pytest.raises(SchemaError, match="risk_flag"):
            schema.validate("monthly_reports", monthly, ids)

    @pytest.mark.parametrize("col", list(schema.NLP_TARGETS))
    def test_unknown_nlp_target(self, nlp, ids, col):
        nlp.loc[0, col] = "???"
        with pytest.raises(SchemaError, match=f"nlp_responses.{col}: unknown"):
            schema.validate("nlp_responses", nlp, ids)

    def test_missing_categorical_column(self, nlp, ids):
        with pytest.raises(SchemaError, match="severity_level: categorical column missing"):
            schema.validate("nlp_responses", nlp.drop(columns="severity_level"), ids)


class TestRule4ForeignKeys:
    def test_orphan_participant(self, monthly, ids):
        monthly.loc[0, "participant_id"] = "NEPS-GHA-9999"
        with pytest.raises(SchemaError, match="participant_id: 1 ids not in participants"):
            schema.validate("monthly_reports", monthly, ids)

    def test_orphan_in_nlp(self, nlp, ids):
        nlp.loc[0, "participant_id"] = "NEPS-GHA-9999"
        with pytest.raises(SchemaError, match="nlp_responses.participant_id: 1 ids"):
            schema.validate("nlp_responses", nlp, ids)

    def test_skipped_when_no_ids_given(self, monthly):
        monthly.loc[0, "participant_id"] = "NEPS-GHA-9999"
        schema.validate("monthly_reports", monthly)  # no participant_ids -> rule 4 not applied

    def test_null_participant_in_nlp(self, nlp, ids):
        nlp.loc[0, "participant_id"] = None
        with pytest.raises(SchemaError, match="nlp_responses.participant_id: 1 nulls"):
            schema.validate("nlp_responses", nlp, ids)


class TestRule5TimeIndex:
    def test_month_25(self, monthly, ids):
        monthly.loc[0, "month"] = 25
        with pytest.raises(SchemaError, match="month: values outside"):
            schema.validate("monthly_reports", monthly, ids)

    def test_month_0(self, monthly, ids):
        monthly.loc[0, "month"] = 0
        with pytest.raises(SchemaError, match="month: values outside 1..24: \\[0\\]"):
            schema.validate("monthly_reports", monthly, ids)

    def test_null_nlp_month(self, nlp, ids):
        nlp.loc[0, "month"] = None
        with pytest.raises(SchemaError, match="nlp_responses.month: 1 null or non-numeric"):
            schema.validate("nlp_responses", nlp, ids)

    def test_text_month_fails_cleanly(self, nlp, ids):
        nlp["month"] = nlp["month"].astype(object)
        nlp.loc[0, "month"] = "March"
        with pytest.raises(SchemaError, match="nlp_responses.month: 1 null or non-numeric"):
            schema.validate("nlp_responses", nlp, ids)

    def test_wave_month_not_in_set(self, data, ids):
        waves = data["comprehensive_waves"].copy()
        waves.loc[0, "wave_month"] = 7
        with pytest.raises(SchemaError, match="wave_month: values outside 6..24: \\[7\\]"):
            schema.validate("comprehensive_waves", waves, ids)

    def test_nlp_month_out_of_range(self, nlp, ids):
        nlp.loc[0, "month"] = 99
        with pytest.raises(SchemaError, match="nlp_responses.month"):
            schema.validate("nlp_responses", nlp, ids)
