"""
V3.1 Data Repository for ML/AI Training.

Reads the verified V3.1 baseline from PostgreSQL so that models train on the
same single source of truth used by the backend and frontend:

    Render/mock REDCap -> Data Platform -> PostgreSQL (nds_v31_source,
    nds_v31_validated, nds_v31_derived) -> ML/AI (this repository)

Raw CRF answers live in ``nds_v31_source.payload`` (the 223-field baseline
record). Authoritative NSES/subscale scores live in ``nds_v31_derived.result``
and are normalised via the clinical contract so both stored shapes are handled.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sqlalchemy import text

from app.db.connection import engine
from src.neps_v31_clinical_contract import (
    DERIVED_SCORE_KEYS,
    normalize_derived_result,
)

# Columns present in the raw payload that are promoted to metadata; dropped from
# the feature block to avoid duplicated/ambiguous columns.
_FEATURE_EXCLUSIONS = {"study_id", "country"}

# Metadata columns describing the source record (not model features). Exported
# so consumers (e.g. the API summary) can distinguish features from metadata.
METADATA_COLUMNS = [
    "source_id",
    "study_id",
    "country",
    "observed_at",
    "source_version",
    "contract_version",
    "nds_participant_id",
    "calculation_version",
]

_DATASET_SQL = """
    SELECT
        s.id                AS source_id,
        s.payload           AS payload,
        s.country           AS country,
        s.observed_at       AS observed_at,
        s.source_version    AS source_version,
        s.contract_version  AS contract_version,
        s.nds_participant_id AS nds_participant_id,
        d.result            AS derived,
        d.calculation_version AS calculation_version
    FROM nds_v31_source s
    LEFT JOIN nds_v31_derived d ON s.id = d.source_id
"""


def _as_dict(value: Any) -> dict:
    """Return a dict from a JSON column value (dict already, or a JSON string)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def _assemble(rows: list[dict]) -> pd.DataFrame:
    """Assemble ML-ready rows into a single DataFrame.

    Pure function (no I/O) so it can be unit-tested. Each row is expected to
    expose the columns selected by ``_DATASET_SQL``.
    """
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)

    meta_values = {
        "source_id": frame["source_id"],
        "study_id": frame["payload"].map(lambda p: _as_dict(p).get("study_id")),
        "country": frame["country"],
        "observed_at": frame["observed_at"],
        "source_version": frame["source_version"],
        "contract_version": frame["contract_version"],
        "nds_participant_id": frame.get("nds_participant_id"),
        "calculation_version": frame.get("calculation_version"),
    }
    meta = pd.DataFrame({col: meta_values[col] for col in METADATA_COLUMNS})

    payloads = [_as_dict(p) for p in frame["payload"]]
    features = pd.json_normalize(payloads)
    features = features.drop(columns=[c for c in _FEATURE_EXCLUSIONS if c in features.columns])

    score_rows = [normalize_derived_result(result)[0] for result in frame["derived"]]
    scores = pd.DataFrame(score_rows)
    scores.columns = [f"score_{key}" for key in DERIVED_SCORE_KEYS]
    scores = scores.apply(pd.to_numeric, errors="coerce")

    return pd.concat([meta.reset_index(drop=True), features.reset_index(drop=True),
                      scores.reset_index(drop=True)], axis=1)


class V31Repository:
    """Repository for V3.1 baseline data from PostgreSQL."""

    def __init__(self, db_engine=None):
        self.engine = db_engine or engine

    def _fetch(self, country: str | None = None) -> pd.DataFrame:
        if country is None:
            query, params = text(_DATASET_SQL), {}
        else:
            query = text(_DATASET_SQL + " WHERE s.country = :country")
            params = {"country": country}
        with self.engine.connect() as conn:
            frame = pd.read_sql(query, conn, params=params)
        return _assemble(frame.to_dict(orient="records"))

    def get_v31_baseline_dataset(self) -> pd.DataFrame:
        """Raw CRF features (223 fields) + authoritative scores + metadata."""
        return self._fetch()

    def get_v31_raw_features(self) -> pd.DataFrame:
        """Only the raw CRF answer columns (metadata excluded)."""
        dataset = self._fetch()
        if dataset.empty:
            return dataset
        score_cols = [c for c in dataset.columns if c.startswith("score_")]
        return dataset.drop(columns=["source_id", *score_cols])

    def get_v31_scores(self) -> pd.DataFrame:
        """Only the authoritative derived scores plus identifying metadata."""
        dataset = self._fetch()
        if dataset.empty:
            return dataset
        score_cols = [c for c in dataset.columns if c.startswith("score_")]
        return dataset[["source_id", "study_id", "country", "observed_at", *score_cols]]

    def get_v31_by_country(self, country: str) -> pd.DataFrame:
        """V3.1 baseline for a single country (e.g. "Ghana")."""
        return self._fetch(country=country)

    def get_record_count(self) -> dict:
        """Summary statistics for the V3.1 baseline tables."""
        query = text("""
            SELECT
                COUNT(*) FILTER (WHERE d.result IS NOT NULL) AS scored_records
            FROM nds_v31_source s
            LEFT JOIN nds_v31_derived d ON s.id = d.source_id
        """)
        with self.engine.connect() as conn:
            scored = conn.execute(query).scalar_one()
            total = conn.execute(text("SELECT COUNT(*) FROM nds_v31_source")).scalar_one()
            countries = conn.execute(
                text("SELECT COUNT(DISTINCT country) FROM nds_v31_source")
            ).scalar_one()
        return {
            "total_records": total,
            "countries": countries,
            "scored_records": scored,
        }
