"""Data contract: expected columns, dtypes and ranges for each dataset.

Single source of truth. `validate()` is called on every dataset by the loader
so real REDCap data fails loudly at ingestion, not silently inside a model.
Human-readable version: docs/ml/DATA_CONTRACT.md.
"""
import pandas as pd

# Rule 1: unique, non-null keys. nlp_responses is NOT unique on
# (participant_id, month): a participant can answer several prompts a month.
KEYS = {
    "participants": ["participant_id"],
    "monthly_reports": ["participant_id", "month"],
    "comprehensive_waves": ["participant_id", "wave_month"],
    "nlp_responses": ["response_id"],
}

# Rule 2: tabular model inputs (monthly_reports), inclusive ranges observed in
# mock v0.3.0. DATASET_DESCRIPTION.md ranges were narrower than the data.
MONTHLY_SCORES = {
    "anxiety": (0, 30),
    "depression": (0, 30),
    "stress": (0, 40),
    "social_isolation": (0, 15),
    "self_esteem": (0, 40),
    "loneliness": (0, 25),
    "daily_functioning": (0, 100),
    "mood_score": (0, 100),
    "sleep_quality_score": (0, 30),
    "fatigue_score": (0, 30),
    "attendance_score": (0, 25),
    "coping_score": (0, 30),
    "substance_abuse_score": (0, 15),
    "suicidality_score": (0, 15),
}

# Rule 3: categorical targets.
RISK_FLAG_TIERS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]  # mock has 4 tiers, not 2

NLP_TARGETS = {
    "sentiment_manual": ["positive", "mildly_positive", "neutral", "mildly_negative", "negative"],
    "emotional_label": ["joy", "hope", "fear", "sadness", "disappointment", "confusion"],
    "severity_level": ["low", "moderate", "high"],
    "clinical_status": ["normal", "anxious", "stressed", "depressed", "suicidal_ideation"],
    "alert_priority": ["p0", "p1", "p2"],
}

# Safety rule input for alerts; a rename or re-spelling must fail at ingestion.
SUICIDALITY_SCREENING = ["No", "Ideation", "Active plan", "Previous attempt"]

CATEGORICALS = {
    "monthly_reports": {"risk_flag": RISK_FLAG_TIERS, "suicidality_screening": SUICIDALITY_SCREENING},
    "nlp_responses": NLP_TARGETS,
}

# Rule 5: time indices.
MONTHS = range(1, 25)
WAVE_MONTHS = [6, 12, 18, 24]
TIME_INDEX = {
    "monthly_reports": ("month", set(MONTHS)),
    "comprehensive_waves": ("wave_month", set(WAVE_MONTHS)),
    "nlp_responses": ("month", set(MONTHS)),
}

DEMOGRAPHICS = ["country", "site", "gender", "age", "socioeconomic_status",
                "food_security", "healthcare_access", "employment_status"]

# Columns used for fairness breakdowns in every evaluation report.
FAIRNESS_GROUPS = ["country", "gender", "site"]


class SchemaError(ValueError):
    """Raised with the dataset and column that broke the contract."""


def _fail(name, col, msg):
    raise SchemaError(f"{name}.{col}: {msg}")


def _numeric(name, col, series):
    """Coerce to numeric; strings like 'N/A' become NaN and fail as nulls."""
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().any():
        _fail(name, col, f"{int(s.isna().sum())} null or non-numeric values")
    return s


def validate(name, df, participant_ids=None):
    """Apply the five contract rules to one dataset. Returns df unchanged.

    participant_ids: set of valid ids from participants.csv, for rule 4.
    """
    if name not in KEYS:
        raise SchemaError(f"unknown dataset {name!r}")
    if df.empty:
        _fail(name, "*", "no rows")

    # Rule 1: keys present, non-null, unique.
    keys = KEYS[name]
    for k in keys:
        if k not in df.columns:
            _fail(name, k, "key column missing")
        if df[k].isna().any():
            _fail(name, k, f"{int(df[k].isna().sum())} null keys")
    if df.duplicated(keys).any():
        _fail(name, "+".join(keys), f"{int(df.duplicated(keys).sum())} duplicate keys")

    # Rule 2: score columns present and in range (monthly only).
    if name == "monthly_reports":
        for col, (lo, hi) in MONTHLY_SCORES.items():
            if col not in df.columns:
                _fail(name, col, "score column missing")
            s = _numeric(name, col, df[col])
            bad = int(((s < lo) | (s > hi)).sum())
            if bad:
                _fail(name, col, f"{bad} values outside [{lo}, {hi}]")

    # Rule 3: categoricals only take known values.
    for col, allowed in CATEGORICALS.get(name, {}).items():
        if col not in df.columns:
            _fail(name, col, "categorical column missing")
        if df[col].isna().any():
            _fail(name, col, f"{int(df[col].isna().sum())} nulls")
        unknown = set(df[col].unique()) - set(allowed)
        if unknown:
            _fail(name, col, f"unknown values {sorted(map(str, unknown))}")

    # Rule 4: every participant_id exists in participants.
    if name != "participants":
        if "participant_id" not in df.columns:
            _fail(name, "participant_id", "column missing")
        if df["participant_id"].isna().any():
            _fail(name, "participant_id", f"{int(df['participant_id'].isna().sum())} nulls")
        if participant_ids is not None:
            orphans = set(df["participant_id"]) - set(participant_ids)
            if orphans:
                _fail(name, "participant_id", f"{len(orphans)} ids not in participants")

    # Rule 5: time index in the expected set.
    if name in TIME_INDEX:
        col, allowed = TIME_INDEX[name]
        if col not in df.columns:
            _fail(name, col, "time column missing")
        s = _numeric(name, col, df[col])
        bad = sorted(set(s.tolist()) - allowed)
        if bad:
            _fail(name, col, f"values outside {min(allowed)}..{max(allowed)}: {bad}")

    return df
