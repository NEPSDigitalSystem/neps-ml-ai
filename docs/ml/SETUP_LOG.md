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
| tests/test_ml_loader.py | 55 tests (parametrised): row counts, fill, round-trip, loader wiring, one or more failure per rule, edge cases |
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
| tests/test_ml_train.py | 28 tests: feature correctness, participant isolation, metrics slicing, registry round-trip and collision |
| .gitignore | models/*/ (versioned registry output is reproducible, not committed) |

Baseline on mock v0.3.0, 720 test rows: accuracy 0.98, macro-F1 0.65 (CRITICAL
has 9 rows total). Pipeline validation only.

Reviewed by two subagents. Fixed: default-arg bug that wrote test output into
the real registry, redundant rolling feature, dead argv, empty-registry error,
four weak tests. Open: one-hot columns must be reindexed to meta["features"]
when the predict path lands (Phase 5); per-group macro-F1 averages over the
classes present in that group.

## Phase 3: Trajectory and explainability (2026-09-23)

| Path | Change |
|---|---|
| app/ml/features.py | Lags and next-month targets now computed on a full participant x month grid, so a missing month yields NaN instead of the wrong month (real data has ~85% completion). add_next_month_targets() |
| app/ml/evaluate.py | regression_metrics (MAE, R2, R2 None on constant groups); metrics_report takes a metric function |
| app/ml/train.py | train_trajectory() for next-month depression and anxiety; trend_label() Improving / Stable / Deteriorating at TREND_THRESHOLD=2.0; guard that targets are higher-is-worse scores; main() trains all three models |
| app/ml/explain.py | SHAP TreeExplainer contributions for the predicted class, share of total |SHAP|, plain-language labels and sentences; binary and multiclass models |
| tests/conftest.py | Session-scoped data, frame, classifier fixtures (suite 42s to 12s) |
| tests/test_ml_explain.py | 40 tests |

Baseline on mock v0.3.0: trajectory R2 0.13 (depression) and 0.07 (anxiety),
trend accuracy 0.50. Expected: mock scores are independent month to month.
Pipeline validation only.

Reviewed by two subagents. Fixed: positional shift bridging missing months
(found by a test), binary-class SHAP crash, class index read from meta rather
than the model, NaN predictions silently labelled Stable, R2 NaN on constant
fairness cells, vacuous HIGH-risk test, duplicate end-to-end test. Open:
TREND_THRESHOLD is a placeholder pending clinical minimal important difference;
share is a log-odds attribution, not a probability, and the sentence wording
should be checked with WP6.

## Phase 4: Fairness audit and early-warning alerts (2026-09-23)

| Path | Change |
|---|---|
| app/ml/fairness.py | fairness_report(): per-group selection rate, TPR, FPR, demographic parity gap, equal opportunity gap, warn flag at GAP_WARN=0.1. Positive = HIGH or CRITICAL. Missing demographics form their own group |
| app/ml/alerts.py | generate_alerts(): crisis rule (suicidality >= 7 or screening Active plan / Previous attempt) -> P0, model HIGH/CRITICAL or P(positive) >= 0.5 -> P1, Deteriorating trend or MEDIUM -> P2; reasons list; intervention categories per priority |
| app/ml/schema.py | site added to FAIRNESS_GROUPS; suicidality_screening is now a contract categorical (safety-rule input must fail at ingestion) |
| app/ml/features.py | suicidality_screening kept in the frame as a non-feature |
| app/ml/train.py | risk classifier meta gains "fairness" |
| app/ml/evaluate.py | group breakdowns no longer drop rows with a missing group value |
| tests/test_ml_alerts.py | 33 tests, with a deterministic stub model for P1/P2 paths |

Mock baseline: country parity gap 0.03, gender 0.07, site 0.19 (warn: small
site cells). Crisis rule recovers mock HIGH/CRITICAL at 99.9%, which confirms
the mock flag is a suicidality threshold. Alerts on all 3,600 rows: P0 179,
P2 1,020, none 2,401.

Reviewed by two subagents. Fixed: screening rule was soft (no schema rule,
guarded column), rows with a missing demographic were silently dropped from
fairness and metric breakdowns, POSITIVE duplicated, two vacuous tests replaced
with stub-model tests. Open: SUICIDALITY_CRISIS, P_HIGH_ALERT and GAP_WARN are
placeholders pending the safeguarding protocol and a governance tolerance;
anomaly detection from the proposal skipped (rules plus model cover it for now).

## Pre-Phase-5 review and end-to-end check (2026-09-24)

Fresh registry, `python -m app.ml.train`, reload in a separate process,
explain and alerts on the saved models: all consistent. Two whole-of-work
reviews (code coherence, conformance to the docs). Fixed:

| Path | Change |
|---|---|
| app/ml/features.py | align(frame, meta): reindex to the trained feature list. A single-participant frame lacks most one-hot columns and crashed explain and alerts (Phase 5 blocker). trend_label and its constants moved here from train.py so alerts no longer imports train |
| app/ml/schema.py | POSITIVE_TIERS lives next to RISK_FLAG_TIERS |
| app/ml/evaluate.py | by_group() shared with fairness.py; duplicate group-key helper removed |
| app/ml/features.py | KEYS taken from schema.KEYS |
| app/ml/explain.py | unused import removed; uses align() |
| tests/test_e2e.py | 2 tests: train to a temp registry, reload, single-participant alert + explanation, JSON-safe payload; single-participant predictions equal full-frame predictions |
| docs/ml/README.md | status, architecture (fairness, alerts, align), registry contents, deferred-from-proposal list, Phase 5 needs, task statuses for 1, 2, 4 |
| docs/ml/DATA_CONTRACT.md | features use two of the four files; fatigue_level is dropped |

Test counts in earlier phase entries corrected to collected counts.
Suite: 198 tests (40 mock client, 158 ML) in ~20s.

## Not done yet

Phase 5: FastAPI endpoints (predict risk, trajectory, explain, alerts) with
the one-hot reindex guard; REDCap loader backend; Docker image check.
Phase 6: clustering, cross-country analytics, stigma model.
