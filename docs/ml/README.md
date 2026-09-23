# NEPS ML Pipeline: Architecture and Roadmap

Owner: Kelvin Ackah, AI/ML intern, longitudinal track (listed as "Daniel" on the
WP4 roster by mistake). Topic: Longitudinal Machine Learning Models for Youth
Psychosocial Risk Prediction, 30 hours. Yasmine's NLP track (sentiment, emotion,
notebooks in src/ and notebooks/) is complete and is consumed here as input.
Status: setup phase, no model code yet.

## Kelvin's tasks from the docs

Core (named in the roster abstract and assessment brief):
1. Predictive models for stress and risk trajectories over time
2. Explainable AI on those models (SHAP, plain-language contributions)
3. Bias evaluation by country, gender and site
4. Combine numeric data with Yasmine's text outputs as features

Secondary (from the proposal's use cases):
5. Early warning alerts and intervention category recommendation
6. Psychosocial pattern clustering
7. Cross-country comparative analytics
8. Dropout and retention prediction (blocked: no withdrawal dates in mock)
9. Stigma and help-seeking modelling from wave surveys
10. Dashboard and backend integration of risk trajectories and explanations

## Goal

Build the pipeline on the mock REDCap data so that when real WP4 data arrives
only the ingestion step changes. Models retrain from the same feature code and
the same evaluation harness.

## Architecture

```
data source (csv | redcap)
        |
   app/ml/loader.py      one function, two backends, chosen by REDCAP_MOCK_ENABLED
        |
   app/ml/schema.py      validate columns, types, ranges; fail loudly
        |
   app/ml/features.py    pure functions, merge on participant_id + month
        |
   app/ml/train.py       three simple baselines, saved with metadata
        |
   app/ml/evaluate.py    participant-level splits, fairness breakdown
        |
   app/ml/explain.py     SHAP -> plain-language contributions for WP6
        |
   main.py               /predict/risk  /predict/trajectory  /analyse/text  /explain
```

## Deliverables and what the mock data supports

| Deliverable | Mock data | Notes |
|---|---|---|
| Joins and pipeline plumbing | Good | Clean keys, 150 x 24 months, zero null keys |
| Risk flag classifier | Good for plumbing | 4 tiers, CRITICAL has 9 rows; flag is a threshold on inputs so scores are not meaningful |
| Explainability (SHAP) | Good | Flag driven by anxiety, depression, suicidality |
| Fairness audit | Good | Country balanced 50/50/50, four gender values |
| Clustering, cross-country | Good | Waves need imputation on six columns |
| Trajectory forecasting | Weak | Per-person trend centred on zero; lag-1 autocorr 0.46 |
| Text models | Weak | 112 unique sentences in 2,000 rows, 91% negative |
| Multimodal fusion | Blocked | Text severity contradicts clinical risk flag |
| Dropout prediction | Blocked | No withdrawal dates |

## Roadmap

1. Schema contract and validate() (app/ml/schema.py) DONE
2. Loader with csv backend (app/ml/loader.py) DONE; redcap backend in step 8
3. Feature functions and merge (app/ml/features.py) DONE
4. Baselines: risk classifier DONE; trajectory regression DONE; text features later (app/ml/train.py)
5. Evaluation harness with participant splits and fairness table (app/ml/evaluate.py) DONE
6. SHAP explanations in WP6 format (app/ml/explain.py) DONE, wording to confirm with WP6
7. Model registry: joblib + metadata JSON per model DONE
8. API endpoints in main.py

Deferred until real data: transformer fine-tuning, LSTM or temporal
transformers, multimodal fusion, dropout prediction.

## Model registry

Each trained model is saved to `models/<name>/<version>/` with `model.pkl` and
`meta.json` holding data_version, feature list, metrics (overall and per
fairness group), and trained_at. Retraining on real data produces a new
version compared against the mock baseline in the same table.

## Open questions for the team

- Frank (ETL): exact REDCap export field names; is risk_flag 2 or 4 tiers in production?
- Backend intern: response shape of the /explain endpoint.
- Predict path (Phase 5): reindex incoming one-hot columns to meta["features"] so unseen or missing categories cannot silently shift columns.
- Study leads: are withdrawn or non-consented participants excluded from training?

## Ethics rules (from Yasmine's proposal)

No stigmatising labels, human-in-the-loop, explainable outputs only, bias
monitoring by country and gender, anonymised inputs, aggregate where possible.
