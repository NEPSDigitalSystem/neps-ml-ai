# NEPS ML Pipeline: Architecture and Roadmap

Owner: Kelvin Ackah, AI/ML intern, longitudinal track (listed as "Daniel" on the
WP4 roster by mistake). Topic: Longitudinal Machine Learning Models for Youth
Psychosocial Risk Prediction, 30 hours. Yasmine's NLP track (sentiment, emotion,
notebooks in src/ and notebooks/) is complete and is consumed here as input.
Status: Phases 1 to 4 done (foundation, features and risk classifier,
trajectory and explainability, fairness and alerts). Phase 5 (endpoints,
REDCap backend) next. See SETUP_LOG.md for the per-phase record.

## Kelvin's tasks from the docs

Core (named in the roster abstract and assessment brief):
1. Predictive models for stress and risk trajectories over time DONE as baselines (risk tier classifier, next-month depression/anxiety regressors, trend label). Sequence models deferred to real data
2. Explainable AI on those models (SHAP, plain-language contributions) DONE for the risk classifier; trajectory regressors are not explained yet
3. Bias evaluation by country, gender and site DONE (app/ml/fairness.py)
4. Combine numeric data with Yasmine's text outputs as features NOT STARTED (nlp_responses is loaded and validated only)

Secondary (from the proposal's use cases):
5. Early warning alerts and intervention category recommendation DONE (app/ml/alerts.py)
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
data source (csv today | redcap in Phase 5)
        |
   app/ml/loader.py      load_datasets(source=...); every frame validated
        |
   app/ml/schema.py      contract: keys, ranges, categoricals, FKs, months
        |
   app/ml/features.py    lags/deltas on a full month grid, one-hot demographics,
                         align() for inference, trend_label()
        |
   app/ml/train.py       risk_classifier + trajectory_<score>; registry with meta.json
        |
   app/ml/evaluate.py    participant-level split; metrics overall + by group
   app/ml/fairness.py    selection rate, TPR, FPR, parity and opportunity gaps
        |
   app/ml/explain.py     SHAP -> plain-language contributions for WP6
   app/ml/alerts.py      crisis rule + model + trend -> P0/P1/P2 + interventions
        |
   main.py               Phase 5: /predict/risk /predict/trajectory /explain /alerts
                         (today: / and /health only)
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
2. Loader with csv backend (app/ml/loader.py) DONE; redcap backend in Phase 5
3. Feature functions and merge (app/ml/features.py) DONE
4. Baselines: risk classifier DONE; trajectory regression DONE; text features later (app/ml/train.py)
5. Evaluation harness with participant splits and fairness table (app/ml/evaluate.py) DONE
6. SHAP explanations in WP6 format (app/ml/explain.py) DONE, wording to confirm with WP6
7. Model registry: joblib + metadata JSON per model DONE
8. API endpoints in main.py

Deferred until real data: transformer fine-tuning, LSTM or temporal
transformers, multimodal fusion, dropout prediction.

## Model registry

Each trained model is saved to `models/<name>/<version>/` (gitignored) with
`model.pkl` and `meta.json` holding name, target, classes, feature list,
data_version, split sizes, seed, metrics (overall and per fairness group),
fairness report (classifier), trend_threshold and trend_metrics (trajectory),
trained_at and library versions. Retraining on real data produces a new
version compared against the mock baseline in the same table.
`python -m app.ml.train` trains all models; `train.load_model(name)` loads
the latest version.

## Deferred from the proposal (with reason)

- LSTM / temporal transformers: no trend signal in mock data.
- Transformer text models, fusion: text is templated; fusion target contradicts.
- Counterfactual explanations, attention visualisation: need real data and text models.
- Anomaly detection: rule + model hybrid covers the alert need for now.
- Dropout prediction: no withdrawal dates.
- A single "risk trajectory score": today p_high (alerts) plus trend label
  cover it; a combined score needs a definition agreed with WP6.

## Open questions for the team

- Frank (ETL): exact REDCap export field names; is risk_flag 2 or 4 tiers in production?
- Backend intern: response shape of the /explain endpoint.
- Phase 5 needs: a predict-one-participant function (history -> features -> risk, trend, alert, explanation), a REDCap loader backend that tolerates missing waves/nlp for a single participant, and a cached SHAP explainer per loaded model. align() already guards one-hot drift.
- Study leads: are withdrawn or non-consented participants excluded from training?

## Ethics rules (from Yasmine's proposal)

No stigmatising labels, human-in-the-loop, explainable outputs only, bias
monitoring by country and gender, anonymised inputs, aggregate where possible.
