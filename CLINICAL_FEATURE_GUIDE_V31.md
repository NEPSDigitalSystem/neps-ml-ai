# NEPS V3.1 Clinical Feature Integration Guide for ML/AI Team

**Target Audience:** Lead ML Engineer (Yasmine) & NEPS ML Research Team  
**Author:** DevOps Lead & Project Manager (Damien Nsoh Ayine)  
**Date:** October 2026  
**Status:** Official Reference Architecture (Aligned with Board Handover)

---

## 1. Executive Summary

During the recent architectural review, the NEPS Board (Prof Brew) and the Data Platform team finalized and froze the official **NEPS Master Case Report Form (CRF V3.1)** and the clinical psychometric scoring rules for the **NEPS Stress Experience Scale (NSES)**.

This resolves the long-standing question regarding **what exact survey questions, clinical subscales, and database tables exist in the NEPS Digital System**.

---

## 2. The Official Clinical Feature Inventory (73 Items)

Rather than arbitrary synthetic metrics, the official system measures youth psychosocial wellbeing using **7 validated clinical instruments comprising exactly 73 items**:

```
                       NEPS Master CRF V3.1 (188 Survey Items)
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
               73 Core NSES Items              115 Longitudinal /
          (Strict Clinical Psychometrics)       Qualitative Items
                         │
     ┌───────┬───────┬───┴───┬───────┬───────┬───────┬───────┐
     ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
    EPI     ESI     SBI     K6      BRS    MSPSS    GSE   (Demographics,
   (10)    (10)    (19)     (6)     (6)     (12)    (10)   Free-Text NLP)
```

### Breakdown of the 7 Clinical Domains:

1. **EPI (Educational Pressure Index - 10 items):**
   * Fields: `epi_expectations_1-2`, `epi_failure_1-2`, `epi_caregivers_1-2`, `epi_future_1-2`, `epi_progress_1-2`
   * Measures: Academic burden, caregiver expectations, fear of failure (1–5 Likert).
2. **ESI (Economic Stress Index - 10 items):**
   * Fields: `esi_finstrain_1-2`, `esi_food_1-2`, `esi_educost_1-2`, `esi_materials_1-2`, `esi_transport_1-2`
   * Measures: Household financial strain, food insecurity, schooling costs (1–5 Likert).
3. **SBI (Stigma and Burden Index - 19 items):**
   * Fields: `sbi_loved_1-8`, `sbi_comm_1-4`, `sbi_self_1-4`, `sbi_help_1-3`
   * Measures: Mental health stigma across loved ones, community, self-stigma, and barriers to help-seeking (1–5 Likert).
4. **K6 (Kessler-6 Psychological Distress Scale - 6 items):**
   * Fields: `k6_1` to `k6_6`
   * Measures: Non-specific severe distress (nervous, hopeless, restless, depressed, effort, worthless; 0–4 scale).
5. **BRS (Brief Resilience Scale - 6 items):**
   * Fields: `brs_01` to `brs_06`
   * Measures: Ability to recover and bounce back from adversity (1–5 Likert).
6. **MSPSS (Multidimensional Scale of Perceived Social Support - 12 items):**
   * Fields: `mspss_01` to `mspss_12`
   * Measures: Perceived support from Significant Others, Family, and Friends (1–7 Likert).
7. **GSE (General Self-Efficacy Scale - 10 items):**
   * Fields: `gse_01` to `gse_10`
   * Measures: Coping confidence across difficult life challenges (1–4 Likert).

---

## 3. Mapping: Old Mock Features vs. Official V3.1 Scales

To help you adapt existing notebooks (`nlp-01-eda.ipynb` through `nlp-08-recommendation-system.ipynb`), here is how our earlier synthetic columns map to the real clinical instruments:

| Previous Mock Metric | Official Clinical Scale | Exact V3.1 Database Source |
| :--- | :--- | :--- |
| `stress` | **NSES Composite & EPI** | `nds_v31_derived.result['NSES']` (composite), `nds_v31_derived.result['EPI']`, `epi_*` items |
| `anxiety` / `depression` | **K6 Distress Scale** | `k6_1` to `k6_6` items |
| `resilience` / `coping_score` | **BRS & GSE Scales** | `brs_01-06`, `gse_01-10` |
| `social_isolation` | **MSPSS Social Support** | `mspss_01-12` (inversely correlated with isolation) |
| `food_security` / `SES` | **ESI Economic Stress** | `esi_finstrain_*`, `esi_food_*` |
| `stigma_score` | **SBI Stigma & Burden** | `sbi_loved_*`, `sbi_comm_*`, `sbi_self_*` |

---

## 4. How to Use the New Contract in Python

A helper module has been added to `src/neps_v31_clinical_contract.py`:

```python
from src.neps_v31_clinical_contract import (
    CORE73_FIELD_NAMES,
    SUBSCALES,
    extract_core73_features,
    calculate_subscale_means
)

# Example: Extracting features from a dictionary
raw_survey_dict = {"k6_1": 3, "k6_2": 2, "epi_expectations_1": 4, ...}
features = extract_core73_features(raw_survey_dict)

# Calculate domain-level aggregates for tabular models
subscale_means = calculate_subscale_means(features)
# Returns: {'EPI': 3.5, 'ESI': 2.1, 'SBI': 1.8, 'K6': 2.5, 'BRS': 3.0, 'MSPSS': 5.2, 'GSE': 3.1}
```

### 4.1 Authoritative Derived Scores (`nds_v31_derived.result`)

The Data Platform scoring engine writes the official NSES composite and its subscales
into `nds_v31_derived.result`. **Do not re-derive these in the ML service** — read them
directly. The only valid score keys are (see `DERIVED_SCORE_KEYS`):

| Key | Meaning |
| :--- | :--- |
| `NSES` | NEPS Stress Experience Scale composite (0–100; `null` unless EPI applicability is verified) |
| `EPI` | Educational Pressure Index (0–100; `null` unless applicability is verified) |
| `ESI` | Economic Stress Index (0–100) |
| `SBI` | Stigma & Burden Index (0–100) |
| `MHDI` | Mental Health Distress Index (K6 normalised, 0–100) |
| `K6_total` | Kessler-6 total raw score (0–24) |
| `BRS_100` | Brief Resilience Scale (0–100) |
| `MSPSS_100` | Perceived Social Support (0–100) |
| `GSE_100` | General Self-Efficacy (0–100) |
| `PFI` | Protective Factors Index composite (0–100) |

> **Note:** values may be `null` when the underlying items are incomplete, or (for
> `NSES`/`EPI`) when applicability has not yet been verified — this is expected and
> the missingness should be handled, not imputed blindly.

### 4.2 Reading Verified Data Directly from PostgreSQL

Use the repository — it is the single source of truth and joins `nds_v31_source`
(the 223-field baseline payload) with `nds_v31_derived` (the scores above):

```python
from app.data.v31_repository import V31Repository

repo = V31Repository()

# One ML-ready row per participant: metadata + all payload feature columns + score_* columns
dataset = repo.get_v31_baseline_dataset()

# Feature columns only, or scores only
raw_features = repo.get_v31_raw_features()   # source payload fields
scores = repo.get_v31_scores()               # score_NSES, score_EPI, ...
country_view = repo.get_v31_by_country("Ghana")

# Coverage summary
repo.get_record_count()  # {'total_records': ..., 'countries': ..., 'scored_records': ...}
```

The API also exposes `GET /api/v31/contract`, `GET /api/v31/stats`, and
`GET /api/v31/dataset` for quick inspection without Python.

---

## 5. Guidelines for ML Training & Deployment

1. **Continue Using Synthetic Data for Development:**  
   Do not pause your ongoing model training. Continue developing text sentiment and tabular risk forecasting using `data/raw/` (`nlp_responses.csv`, `consolidated_ml_dataset.csv`).
2. **Align Future Feature Selectors with the 73 Items:**  
   When building tabular feature vectors, ensure your feature transformers accept the 73 core columns or the 7 domain subscale means.
3. **Multimodal NLP + Clinical Risk Joining:**  
   Once live REDCap data is ingested by the Data Platform, each qualitative narrative (`nlp_responses`) will be accompanied by the participant's exact 73 core responses in `nds_v31_source` and their verified composite NSES score in `nds_v31_derived`.
