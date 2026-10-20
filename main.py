from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import pandas as pd

from app.data.v31_repository import METADATA_COLUMNS, V31Repository
from src.neps_v31_clinical_contract import (
    CORE73_FIELD_NAMES,
    DERIVED_SCORE_KEYS,
    DERIVED_SCORE_LABELS,
)

app = FastAPI(title="NEPS ML/AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


@app.get("/")
def read_root():
    return {"message": "NEPS ML/AI API", "status": "ok"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/api/v31/contract")
def get_v31_contract():
    """Official V3.1 feature/score contract used for ML feature engineering."""
    return {
        "core73_item_count": len(CORE73_FIELD_NAMES),
        "core73_fields": CORE73_FIELD_NAMES,
        "derived_score_keys": DERIVED_SCORE_KEYS,
        "derived_score_labels": DERIVED_SCORE_LABELS,
    }


@app.get("/api/v31/stats")
def get_v31_stats():
    """Get V3.1 data statistics from PostgreSQL."""
    repo = V31Repository()
    return repo.get_record_count()


@app.get("/api/v31/dataset")
def get_v31_dataset():
    """Summarise the V3.1 baseline dataset available for ML training."""
    repo = V31Repository()
    dataset = repo.get_v31_baseline_dataset()
    if dataset.empty:
        return {"records": 0, "feature_columns": 0, "score_coverage": {}, "sample": None}

    score_cols = [c for c in dataset.columns if c.startswith("score_")]
    score_coverage = {
        col: int(dataset[col].notna().sum()) for col in score_cols
    }
    feature_cols = [
        c for c in dataset.columns
        if c not in METADATA_COLUMNS and c not in score_cols
    ]
    sample_records = [
        {key: (None if pd.isna(value) else value) for key, value in record.items()}
        for record in dataset.head(1).to_dict(orient="records")
    ]
    return {
        "records": len(dataset),
        "feature_columns": len(feature_cols),
        "score_coverage": score_coverage,
        "sample": sample_records[0] if sample_records else None,
    }

