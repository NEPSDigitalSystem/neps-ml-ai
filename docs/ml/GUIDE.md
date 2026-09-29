# A Beginner's Guide to the NEPS ML Work

This guide explains, in plain language, what the ML intern on the longitudinal
track was asked to do, what was built, and how the two match up. You do not
need a machine learning background to follow it. Where a technical term is
unavoidable it is explained the first time it appears.

Last updated: 29 September 2026. Covers Phases 1 to 4 plus the review before
Phase 5.

---

## 1. The project in one paragraph

NEPS is a study running in Ghana, Sierra Leone and Tanzania. Every month,
young people answer a survey about how they are feeling: anxiety, depression,
stress, sleep, mood, attendance, and so on. Some also write short free-text
answers. The study wants a digital system that can look at this monthly data
and help field teams (called WP6) spot young people whose mental health is
getting worse, early enough to offer support. The ML module is the part of
that system that does the "spotting".

## 2. What the documents asked for

Three PDFs in the `docs/` folder define the job.

### 2.1 The intern assessment brief

The AI/ML intern must show how they would:

1. Model longitudinal psychosocial risk over time. In plain words: use a
   person's history of monthly scores to say how at-risk they are and which
   way they are heading.
2. Combine numerical and text-based data. Use both the survey numbers and
   the written answers.
3. Explain model outputs to WP6 teams ethically. Field teams are not data
   scientists. Whatever the model says must come with a reason a person can
   read, and must not label anyone in a harmful way.

### 2.2 The WP4 intern roster

Two AI/ML interns are listed. One seat, "Longitudinal Machine Learning Models
for Youth Psychosocial Risk Prediction", is Kelvin's (the roster prints
"Daniel" by mistake). The abstract for that seat says: build predictive
models for stress and risk trajectories, use explainable AI, and evaluate
bias. The other seat, NLP and explainable AI, is Yasmine's, and her work
(sentiment and emotion models, eight notebooks) was already in the repository
when this work began.

### 2.3 Yasmine's proposal

A nine-page plan listing eleven possible uses of ML. The ones that fall on
the longitudinal side are:

- Longitudinal stress risk prediction, with a risk score, early-warning
  flags and a trend category (Stable, Improving, Deteriorating)
- Multimodal fusion (numbers plus text)
- Explainable AI for WP6 (statements like "sleep instability contributed
  32% to elevated risk")
- Early warning and intervention recommendation (counselling, peer support,
  follow-up assessment)
- Clustering of psychosocial patterns
- Cross-country comparison
- Dropout prediction
- Stigma and help-seeking modelling
- Bias and fairness auditing

It also lists ethics rules: no stigmatising labels, a human always makes the
final decision, only explainable outputs, monitor bias across countries and
genders, keep data anonymised, aggregate where possible.

---

## 3. The data we have (and its limits)

The real survey data does not exist yet. A "mock" dataset was generated to
build against. It has:

| File | What it holds | Rows |
|---|---|---|
| participants.csv | 150 young people: country, site, age, gender, socio-economic details | 150 |
| monthly_reports.csv | 24 monthly surveys per person with 14 numeric scores and a risk flag | 3,600 |
| comprehensive_waves.csv | deeper surveys at months 6, 12, 18, 24 | 600 |
| nlp_responses.csv | free-text answers with labels | 2,000 |

Before writing any model code, the data was profiled. Three findings shaped
everything that followed:

1. **The risk flag is a simple threshold.** In the mock, a person is HIGH risk
   almost exactly when their suicidality score is 7 or more. A model will
   look near-perfect on this data, which proves the plumbing works but says
   nothing about real-world accuracy.
2. **There is no trend to learn.** Each month's scores were generated
   independently. Nobody in the mock gets steadily better or worse. So a
   "trajectory" model cannot show its worth until real data arrives.
3. **The text is templated.** Only 112 distinct sentences appear across
   2,000 answers. Text models would just memorise them.

The decision taken: build the whole pipeline on mock data so that when real
data arrives only the data source changes, and be honest that mock numbers
are pipeline checks, not results.

---

## 4. What was built, phase by phase

The work was split into phases. Each phase was implemented, tested, reviewed
by two independent reviewers, fixed, and committed before the next began.

### Phase 1: Foundation (the front door)

**Problem it solves.** If every part of the pipeline read the CSV files its
own way, every part would make its own assumptions. Real data would break
them in different places at different times.

**What was built.**

- A *data contract* (`app/ml/schema.py`). A written list of the columns each
  file must have, the range each score may take, the allowed values of each
  category, and how files link together. Think of it as a checklist the data
  must pass.
- A *validator* that runs the checklist. It enforces five rules: keys are
  present and unique, scores are numeric and in range, categories only take
  known values, every person in a file exists in the participants file, and
  months are in the expected set. If anything fails, it stops with a message
  naming the file and column.
- A *loader* (`app/ml/loader.py`). One function that returns the four tables
  and runs the validator on each. Everything downstream calls this and
  nothing else.

**Why it matters for a beginner.** Most ML failures on real data are data
failures, not model failures. Catching them at the door with a clear message
saves days of confusion later.

**Found along the way.** The dataset description said two risk tiers; the
data has four. It said the text file was unique per person per month; it is
not. Five score ranges were narrower than the data. All corrected in the
contract.

### Phase 2: Features and the first model

**Problem it solves.** A model cannot read "history" directly. You have to
turn each person's past months into columns.

**What was built.**

- *Temporal features* (`app/ml/features.py`). For each of the 14 scores, the
  value one, two and three months ago, and the change since last month. If a
  month is missing (real surveys have gaps), the feature is blank rather than
  silently borrowing the wrong month. This gap handling was itself a bug
  found by a test in Phase 3 and fixed.
- *Demographic features*. Country, site, gender and so on turned into
  yes/no columns, which is how models consume categories.
- *The risk classifier* (`app/ml/train.py`). A gradient boosting model (a
  standard, robust method that builds many small decision trees) that
  predicts the risk tier from the features.
- *A participant-level split* (`app/ml/evaluate.py`). When measuring the
  model, some people are held back entirely so the model is judged on
  people it has never seen. Splitting by row instead would let the model
  peek at a person's other months, which inflates results.
- *A model registry*. Every trained model is saved with a metadata file
  recording which data version, which features, the results overall and by
  country and gender, and when it was trained. When real data arrives, the
  retrained model sits next to the mock baseline for comparison.

**Mock result.** Accuracy 0.98. As explained in section 3, this is
expected and not meaningful.

### Phase 3: Trajectory and explanations

**What was built.**

- *Trajectory models*. Two regressors that predict next month's depression
  and anxiety score. From the predicted change, a *trend label*: Deteriorating
  if the score is predicted to rise by 2 or more points, Improving if it falls
  by 2 or more, Stable otherwise. The 2-point threshold is a placeholder
  awaiting a clinical decision.
- *Explanations* (`app/ml/explain.py`). Uses SHAP, a standard technique that
  attributes a prediction to each input. The output is a ranked list of the
  features that pushed the prediction, each turned into a sentence:
  "Suicidality contributed 79% towards HIGH risk". This is the exact shape the
  proposal asked for.

**Mock result.** Trajectory R² of 0.13 and 0.07, trend accuracy 0.50. R² is
a measure of how much of the variation the model explains; near zero means
almost none. This is the "no trend to learn" limit from section 3, not a bug.

### Phase 4: Fairness and early-warning alerts

**What was built.**

- *Fairness audit* (`app/ml/fairness.py`). For country, gender and site, it
  reports how often each group is flagged, how often truly at-risk people in
  each group are caught, and the gap between the best and worst group. A
  warn flag fires when a gap exceeds 0.1. On mock data country and gender
  are fine; site warns because some sites have very few people.
- *Alerts* (`app/ml/alerts.py`). A three-layer rule, applied in order:
  1. **P0 crisis**: suicidality score at or above 7, or a screening answer of
     "Active plan" or "Previous attempt". This fires regardless of what the
     model says, because safety rules must not depend on a model.
  2. **P1 high**: the model predicts HIGH or CRITICAL, or gives them at least
     a 50% probability.
  3. **P2 watch**: the trend is Deteriorating, or the model says MEDIUM.

  Each alert lists the reasons that fired and suggested intervention
  categories from the proposal. An alert is a prompt for a human to review,
  never an automatic action.

**Safety detail.** The screening answer was made a required column in the
data contract, so a renamed or re-spelt column in real data fails at
ingestion rather than silently switching off a safety rule.

### Pre-Phase-5 review

The full pipeline was run end to end from a clean state: train, save, reload
in a separate process, produce alerts and explanations. A test now guards
this path. The review found one blocker: asking for a single participant
crashed, because yes/no columns for the other countries did not exist for
that person. Fixed with an alignment step that every prediction goes
through.

---

## 5. Matching what was built to what was asked

### 5.1 Against the assessment brief

| Brief said | What exists | Status |
|---|---|---|
| Model longitudinal risk over time | Risk tier classifier, next-month regressors, trend label | Done as baselines. Sequence models wait for real data |
| Combine numerical and text data | Text file is loaded and validated only | Not started. Text is too templated to be useful yet |
| Explain outputs to WP6 ethically | SHAP sentences, neutral alert priorities, reasons on every alert, human-in-the-loop by design | Done for the risk model. Trajectory models not explained yet. Label wording to confirm with WP6 |

### 5.2 Against the roster abstract

| Abstract said | Status |
|---|---|
| Predictive models for stress and risk trajectories | Done as baselines |
| Explainable AI techniques | Done for the risk model |
| Bias evaluation | Done, by country, gender and site |

### 5.3 Against the proposal's use cases

| Proposal item | Status | Notes |
|---|---|---|
| Risk trajectory score | Partial | Alerts carry a probability of high risk and a trend label. A single combined score needs a definition agreed with WP6 |
| Early warning flags | Done | P0, P1, P2 |
| Trend categories Stable, Improving, Deteriorating | Done | Exactly those three words |
| LSTM or temporal transformers | Deferred | No trend signal in mock data |
| Multimodal fusion | Deferred | Text labels contradict the clinical flag in the mock |
| SHAP explanations "X contributed N%" | Done | Same shape as the proposal's example |
| Attention visualisation, counterfactuals | Deferred | Need real data and text models |
| Early warning and intervention recommendation | Done | Counselling, peer support, follow-up assessment, safeguarding referral |
| Anomaly detection | Skipped for now | The rule-plus-model hybrid covers the need |
| Clustering | Not started | Planned Phase 6 |
| Cross-country comparative analytics | Not started | Planned Phase 6. Per-country metric slices exist |
| Dropout prediction | Blocked | No withdrawal dates in the data |
| Stigma and help-seeking modelling | Not started | Planned Phase 6 |
| Bias and fairness auditing | Done | Demographic parity and equal opportunity, as the proposal names them |
| Dashboard integration | Not started | Planned Phase 5 |

### 5.4 Against the ethics rules

| Rule | How it is reflected |
|---|---|
| No stigmatising labels | Alert levels are P0, P1, P2. Explanations name scores, not people. Outputs still carry the mock's tier names such as CRITICAL; wording to be confirmed with WP6 |
| Human-in-the-loop | Every alert carries its reasons and is documented as a review prompt. No automatic action exists |
| Explainable outputs only | Risk model explained. Trajectory models not yet |
| Bias monitoring across countries and genders | Built into every training run |
| Anonymised data, access control | Backend team's responsibility; only participant ids are used here |
| Aggregate where possible | Metrics and fairness are aggregate. Alerts are per person by necessity |
| Feedback loop with WP6 | Not started |

---

## 6. How to run it yourself

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 198 tests, about 20 seconds
python -m app.ml.train              # trains three models into models/
```

Then in Python:

```python
from app.ml import features, train, alerts, explain
from app.ml.loader import load_datasets

frame = features.build_training_frame(load_datasets())
risk = train.load_model("risk_classifier")
traj = train.load_model("trajectory_depression")

table = alerts.generate_alerts(frame, risk, traj)   # one row per person per month
print(table.priority.value_counts())

row = frame[frame.risk_flag == "HIGH"].head(1)
for c in explain.explain_risk(*risk, row)[0]["contributions"]:
    print(c["sentence"])
```

---

## 7. Things to remember

- **Mock numbers are not results.** They prove the code runs. Real
  performance is unknown until real data arrives.
- **Three thresholds are placeholders**: the suicidality crisis level (7),
  the alert probability (0.5) and the fairness warn gap (0.1). Each is marked
  in the code and needs a decision from the safeguarding protocol or WP6.
- **The trend threshold (2 points) is a placeholder** awaiting a clinical
  minimal important difference per score.
- **Everything is retrainable in seconds.** That was deliberate: simple
  models first, so real data can be tried the day it lands.

## 8. Where to look next

- `docs/ml/README.md`: architecture, roadmap, deferred list, open questions
- `docs/ml/DATA_CONTRACT.md`: every column, range and rule
- `docs/ml/SETUP_LOG.md`: what changed in each phase, what reviewers found
- `app/ml/`: the code, each file with a docstring explaining its role
- `tests/`: every behaviour described above has a test
