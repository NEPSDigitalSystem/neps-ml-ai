"""Feature engineering. Pure functions: DataFrame in, DataFrame out.

build_training_frame(datasets) is the one entry point. It returns one row
per (participant_id, month) with:
  - ID/target/group columns (see NON_FEATURE) kept as-is
  - the 14 MONTHLY_SCORES
  - temporal features per score: lag1..3, delta1
  - one-hot DEMOGRAPHICS
Early months have NaN lags; HistGradientBoosting handles NaN natively.
Shifts are computed on a full participant x month grid, so a missing month
(real data has ~85% completion) yields NaN rather than the wrong month.
"""
import pandas as pd

from app.ml import schema

LAGS = (1, 2, 3)
TEMPORAL_SUFFIXES = tuple(f"_lag{k}" for k in LAGS) + ("_delta1",)
TARGET = "risk_flag"
KEYS = ["participant_id", "month"]
# Kept in the frame but never fed to a model.
NON_FEATURE = KEYS + [TARGET, "survey_date", "requires_follow_up"] + schema.FAIRNESS_GROUPS


def _shifted(df, cols, periods):
    """Values of cols shifted by `periods` calendar months within participant.

    Works on the full participant x month grid so gaps are not bridged.
    Returns a frame indexed like the sorted input.
    """
    grid = pd.MultiIndex.from_product(
        [df["participant_id"].unique(), list(schema.MONTHS)], names=KEYS)
    full = df.set_index(KEYS)[cols].reindex(grid)
    shifted = full.groupby(level="participant_id", sort=False).shift(periods)
    return shifted.reindex(pd.MultiIndex.from_frame(df[KEYS])).reset_index(drop=True)


def monthly_features(monthly):
    """Add lag1..3 and month-over-month delta per score, per participant.

    No rolling mean: with the lags present it is a linear combination.
    """
    df = monthly.sort_values(KEYS).reset_index(drop=True)
    cols = list(schema.MONTHLY_SCORES)
    new = {}
    for k in LAGS:
        lag = _shifted(df, cols, k)
        for col in cols:
            new[f"{col}_lag{k}"] = lag[col]
    for col in cols:
        new[f"{col}_delta1"] = df[col] - new[f"{col}_lag1"]
    return pd.concat([df, pd.DataFrame(new)], axis=1)


def demographic_features(participants):
    """participant_id + numeric age + one-hot of categorical DEMOGRAPHICS.

    Raw country and gender are kept too, for fairness breakdowns.
    """
    cats = [c for c in schema.DEMOGRAPHICS if c != "age"]
    dummies = pd.get_dummies(participants[cats], prefix=cats, dtype="int8")
    dummies.columns = [c.replace(" ", "_") for c in dummies.columns]
    return pd.concat(
        [participants[["participant_id", "age"] + schema.FAIRNESS_GROUPS], dummies], axis=1
    )


def build_training_frame(datasets):
    """Merge monthly temporal features with demographics on participant_id."""
    monthly = monthly_features(datasets["monthly_reports"])
    demo = demographic_features(datasets["participants"])
    keep = [c for c in monthly.columns if c in NON_FEATURE or c in schema.MONTHLY_SCORES
            or c.endswith(TEMPORAL_SUFFIXES)]
    return monthly[keep].merge(demo, on="participant_id", how="left", validate="m:1")


def add_next_month_targets(frame, cols):
    """Add {col}_next = the participant's value of col one month later.

    Rows with no following month (month 24) are dropped. Used by the
    trajectory regressors; the resulting columns are targets, not features.
    """
    df = frame.sort_values(KEYS).reset_index(drop=True)
    nxt = _shifted(df, list(cols), -1)
    for col in cols:
        df[f"{col}_next"] = nxt[col]
    return df.dropna(subset=[f"{c}_next" for c in cols]).reset_index(drop=True)


def feature_columns(frame):
    return [c for c in frame.columns if c not in NON_FEATURE and not c.endswith("_next")]
