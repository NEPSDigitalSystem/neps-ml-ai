"""Feature engineering. Pure functions: DataFrame in, DataFrame out.

Planned:
- monthly_features(monthly): lag-1..3, rolling mean/std, month-over-month
  delta for each MONTHLY_SCORES column; months since enrollment.
- demographic_features(participants): one-hot of DEMOGRAPHICS.
- text_features(nlp): TF-IDF (reuse src/nlp-preprocessing.py) plus the
  existing sentiment/emotion model outputs as numeric columns.
- build_training_frame(datasets): merge on KEYS, never positional concat.

Status: stub.
"""
