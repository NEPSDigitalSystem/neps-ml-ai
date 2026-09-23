# Data Contract

Machine-readable version: `app/ml/schema.py`. This file explains it.
Baseline: mock REDCap v0.3.0, exported to `data/raw/`.

## Datasets and keys

| File | Rows | Key | Used for |
|---|---|---|---|
| participants.csv | 150 | participant_id | demographics, fairness groups |
| monthly_reports.csv | 3,600 | participant_id, month | tabular risk, trajectory |
| comprehensive_waves.csv | 600 | participant_id, wave_month | clustering, stigma |
| nlp_responses.csv | 2,000 | response_id (participant_id + month is not unique) | text models |

Always merge on keys. Never positional concat.

`consolidated_ml_dataset.csv` and `master_nlp_clinical_ml_dataset.csv` are
derived files. The loader does not read them; features.py (Phase 2) rebuilds
them from the four core files.

## Corrections to DATASET_DESCRIPTION.md (observed in the actual CSVs)

| Documented | Actual |
|---|---|
| risk_flag is LOW or HIGH | LOW 3301, MEDIUM 125, HIGH 165, CRITICAL 9 |
| waves at months 1, 6, 12, 18, 24 | 6, 12, 18, 24 only |
| anxiety 0 to 21, depression 0 to 27 | anxiety up to 30, depression up to 27.5, suicidality up to 11.8 |
| sentiment has 5 tiers | only negative, mildly_negative, positive appear |
| emotion has 6 labels | only sadness, fear, hope, joy appear |
| master file 62 columns | 64: two empty unnamed trailing columns, duplicated country/site from merge |

## Known nulls

| File | Column | Nulls | Handling |
|---|---|---|---|
| monthly_reports | substance_use | 1,215 | null means no use; loader fills with "none" |
| monthly_reports | fatigue_level | 40 | impute or drop column (fatigue_score exists) |
| comprehensive_waves | bullying_exposure, food_insecurity, employment_pressure, digital_access, help_seeking_behaviour, awareness_of_services | 116 to 199 each | impute |

## Signal caveats

- risk_flag is a threshold on anxiety, depression and suicidality. A depth-3
  tree recovers it at 0.97. Mock metrics validate the pipeline only.
- Monthly scores are drawn independently each month. No per-person trend.
- 112 unique response texts across 2,000 rows from 10 prompts.
- 1,365 responses labelled high severity fall in LOW risk months.

## Validation rules (implemented in schema.validate(), Phase 1)

0. Dataset has at least one row.
1. Key columns present, non-null, unique per dataset.
2. All MONTHLY_SCORES columns present, numeric, non-null, within range. Non-numeric strings such as "N/A" fail as nulls.
3. Categorical targets non-null and only take values listed in NLP_TARGETS and RISK_FLAG_TIERS. Case-sensitive.
4. Every participant_id in the other three files exists in participants.
5. month numeric and in 1..24, wave_month in {6, 12, 18, 24}.

Every failure raises SchemaError naming `dataset.column`.
