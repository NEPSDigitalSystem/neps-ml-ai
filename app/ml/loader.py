"""Load the four core datasets through one door.

    load_datasets()                  -> data/raw/*.csv (mock, default)
    load_datasets(source="redcap")   -> REDCap API (Phase 5, not implemented)

Returns dict keyed by dataset name. Every frame passes schema.validate()
before being returned, so downstream code never checks columns itself.

Derived files (consolidated_ml_dataset, master_nlp_clinical_ml_dataset) are
not read; features.py rebuilds them from the four core files.
"""
from pathlib import Path

import pandas as pd

from app.ml import schema

DATASETS = ["participants", "monthly_reports", "comprehensive_waves", "nlp_responses"]
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Mock quirk: monthly_reports.substance_use is null when the participant
# reported no substance use. Fill so the column is a clean categorical.
# Lowercase on purpose: "None" is in pandas' default na_values and would
# turn back into NaN on any CSV round-trip.
SUBSTANCE_USE_NONE = "none"


def _clean(name, df):
    if name == "monthly_reports":
        df["substance_use"] = df["substance_use"].fillna(SUBSTANCE_USE_NONE)
    return df


def load_datasets(source="csv", data_dir=DEFAULT_DATA_DIR):
    if source != "csv":
        raise NotImplementedError(f"source={source!r}; only 'csv' exists until Phase 5")
    data_dir = Path(data_dir)
    out = {}
    for name in DATASETS:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        out[name] = _clean(name, pd.read_csv(path))
    ids = set(out["participants"]["participant_id"])
    for name, df in out.items():
        schema.validate(name, df, participant_ids=ids)
    return out
