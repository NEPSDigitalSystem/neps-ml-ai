# Setup Log

Separates what existed before the ML setup from what was added, so the
boundary is clear. Date: 2026-09-22.

## Existed before (untouched)

| Path | What it is |
|---|---|
| README.md, DATASET_DESCRIPTION.md, walkthrough.md | Project and dataset docs from the architecture team |
| docs/*.pdf | Intern assessment brief, Yasmine's proposal, WP4 intern roster |
| main.py | FastAPI skeleton: root, health, Prometheus. No model endpoints |
| app/services/redcap_mock.py | Seeded mock REDCap client |
| data/raw/*.csv | Six exported mock datasets |
| src/nlp-*.py | TF-IDF sentiment, emotion, preprocessing scripts |
| models/*.pkl | Trained sentiment and emotion pickles |
| notebooks/nlp-01..08 | EDA through recommendation notebooks |
| tests/test_redcap_mock_client.py, test_mock.py | Mock client tests only |
| Dockerfile, .github/workflows/ci-cd.yml, .env.example | Infra |

## Added in this setup

| Path | What it is | Code? |
|---|---|---|
| app/ml/__init__.py | Package docstring describing the flow | no |
| app/ml/schema.py | Column lists, ranges, tiers, fairness groups. validate() is a stub | data only |
| app/ml/loader.py | Stub with backend design and mock quirks documented | stub |
| app/ml/features.py | Stub listing planned feature functions | stub |
| app/ml/train.py | Stub listing the three baselines and what is deferred | stub |
| app/ml/evaluate.py | Stub with split and fairness rules | stub |
| app/ml/explain.py | Stub with WP6 output format | stub |
| docs/ml/README.md | Architecture, capability table, roadmap, registry, open questions | |
| docs/ml/DATA_CONTRACT.md | Human-readable contract plus corrections to DATASET_DESCRIPTION.md | |
| docs/ml/SETUP_LOG.md | This file | |

## Modified

| Path | Change |
|---|---|
| requirements.txt | Added pandas, numpy, scikit-learn, joblib, shap. They were already used by src/ and notebooks/ but undeclared, so the Docker image could not run the models |
| README.md | Added a pointer to docs/ml/ |

## Phase 1: Foundation (2026-09-23)

| Path | Change |
|---|---|
| app/ml/schema.py | validate() implements rules 0 to 5. Ranges corrected to observed data. nlp_responses key changed to response_id (participant_id + month is not unique, 443 duplicates). Non-numeric scores and months fail as SchemaError, not TypeError. Null categoricals fail |
| app/ml/loader.py | CSV backend. Loads four core files, fills substance_use nulls with "none" (lowercase: "None" is a pandas NA sentinel), validates each with participant ids for rule 4 |
| tests/test_ml_loader.py | 62 tests: row counts, fill, round-trip, loader wiring, one or more failure per rule, edge cases |
| docs/ml/DATA_CONTRACT.md | nlp key corrected, rules section matches implementation |

Reviewed by two independent subagents (implementation, tests). Findings fixed:
TypeError leak on string scores, null categoricals passing, "None" NA sentinel,
two doc contradictions, weak assertions. Remaining: a missing substance_use
column raises KeyError from the loader rather than SchemaError (accepted).

## Phase 2: Features and first model (2026-09-23)

| Path | Change |
|---|---|
| app/ml/features.py | lag1..3 and delta1 per score per participant, one-hot demographics, merge to training frame, feature_columns(). Rolling mean dropped after review: linear combination of the lags |
| app/ml/evaluate.py | participant-level GroupShuffleSplit, accuracy and macro-F1, breakdown by country and gender |
| app/ml/train.py | HistGradientBoostingClassifier (balanced) on risk_flag; registry models/<name>/<version>/{model.pkl, meta.json}; `python -m app.ml.train` |
| tests/test_ml_train.py | 37 tests: feature correctness, participant isolation, metrics slicing, registry round-trip and collision |
| .gitignore | models/*/ (versioned registry output is reproducible, not committed) |

Baseline on mock v0.3.0, 720 test rows: accuracy 0.98, macro-F1 0.65 (CRITICAL
has 9 rows total). Pipeline validation only.

Reviewed by two subagents. Fixed: default-arg bug that wrote test output into
the real registry, redundant rolling feature, dead argv, empty-registry error,
four weak tests. Open: one-hot columns must be reindexed to meta["features"]
when the predict path lands (Phase 5); per-group macro-F1 averages over the
classes present in that group.

## Not done yet

Phase 3 onward: trajectory regression, SHAP explanations, fairness metrics,
alerts, endpoints.
