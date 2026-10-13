# NEPS ML/AI Dataset Description & Integration Guide

**Author**: NEPS System Architecture Team  
**Target Audience**: Lead ML Engineer (Yasmine) & NEPS ML/AI Research Team  
**Version**: 0.3.0  
**Last Updated**: September 2026  

---

## 1. Overview & Architecture

The NEPS Digital Platform generates synthetic, clinical-grade longitudinal mock data modeled directly after the research protocols for Ghana, Sierra Leone, and Tanzania.

When executing the data extraction pipeline in `neps-ml-ai` (e.g. via `notebooks/nlp-01-eda.ipynb`), **5 CSV datasets** are saved to `data/raw/`.

---

## 2. Dataset Inventory & Descriptions

### Summary Table

| CSV Filename | Row Count | Granularity Level | Purpose & Core Content | Key Join Columns |
| :--- | :--- | :--- | :--- | :--- |
| **`participants.csv`** | 150 | Baseline Participant Level | Demographics (age, gender, country, site, school) & Socio-economic indicators (employment, food security, healthcare access, SES). | `participant_id` |
| **`monthly_reports.csv`** | 3,600 | Monthly Longitudinal Level | 24 monthly self-report surveys per participant ($150 \times 24 = 3,600$). Contains all **14 ML numeric scores** (anxiety, depression, stress, mood_score, sleep_quality_score, fatigue_score, attendance_score, coping_score, substance_abuse_score, suicidality_score, etc.) & risk flags. | `['participant_id', 'month']` |
| **`comprehensive_waves.csv`** | 600 | Survey Wave Level | Deep longitudinal wave surveys conducted at months 1, 6, 12, 18, 24 ($150 \times 4 = 600$). Covers academic pressure, stigma (internalized, community, family, school), and resilience scores. | `['participant_id', 'wave_month']` |
| **`nlp_responses.csv`** | 2,000 | Qualitative Response Level | Open-ended interview transcripts & youth narrative responses (~13–14 responses per participant across 24 months). Includes text content, sentiment scores, emotion labels, severity levels, and thematic codes. | `['participant_id', 'month']` |
| **`consolidated_ml_dataset.csv`** | 3,600 | Pre-merged Monthly Level | Pre-joined `monthly_reports.csv` + `participants.csv` for direct time-series and tabular risk prediction model training. | `['participant_id', 'month']` |

---

## 3. Detailed Schema & Field Mapping

### A. `participants.csv` (150 Rows)
- `participant_id` (e.g., `NEPS-GHA-0001` .. `NEPS-TAN-0150`): Unique participant identifier.
- `country`: Country of origin (`Ghana`, `Sierra Leone`, `Tanzania`).
- `site`: Clinical data collection site (e.g. `Tamale`, `Freetown`, `Dar es Salaam`).
- `school`: Educational institution.
- `age`: Age in years (12–24).
- `date_of_birth`: DOB string (`YYYY-MM-DD`).
- `gender`: Gender identity (`Male`, `Female`, `Other`).
- `grade_level`: Current grade or education status.
- `enrollment_date`: Study enrollment date.
- `cohort_status`: `active`, `inactive`, or `withdrawn`.
- `consent_status`: `consented`, `pending`.
- `phone_contact`: Anonymized phone number.
- `employment_status`: `student`, `employed`, `unemployed`, `informal_work`.
- `food_security`: `secure`, `mild_insecurity`, `moderate_insecurity`, `severe_insecurity`.
- `healthcare_access`: `good`, `limited`, `poor`.
- `socioeconomic_status`: `low`, `lower_middle`, `upper_middle`, `high`.

### B. `monthly_reports.csv` (3,600 Rows)
- `participant_id`: Foreign key matching `participants.csv`.
- `month`: Integer (`1` to `24`).
- `survey_date`: Date survey was completed.
- **Psychosocial Scores**:
  - `anxiety`: Continuous score ($0.0 - 21.0$).
  - `depression`: Continuous score ($0.0 - 27.0$).
  - `stress`: Continuous score ($0.0 - 40.0$).
  - `social_isolation`: Continuous score ($0.0 - 10.0$).
  - `self_esteem`: Continuous score ($10.0 - 40.0$).
  - `loneliness`: Continuous score ($0.0 - 20.0$).
  - `daily_functioning`: Functional impairment score ($0.0 - 100.0$).
- **Yasmine's 7 Numeric ML Scores**:
  - `mood_score`: Numeric mood rating ($0.0 - 100.0$).
  - `sleep_quality_score`: Numeric sleep rating ($0.0 - 20.0$).
  - `fatigue_score`: Numeric fatigue rating ($0.0 - 30.0$).
  - `attendance_score`: Numeric attendance score ($0.0 - 25.0$).
  - `coping_score`: Numeric coping mechanism score ($0.0 - 30.0$).
  - `substance_abuse_score`: Risk indicator score ($0.0 - 10.0$).
  - `suicidality_score`: Safeguarding screening score ($0.0 - 10.0$).
- **Risk Indicators**:
  - `risk_flag`: `LOW` or `HIGH`.
  - `requires_follow_up`: `True` / `False`.

### C. `nlp_responses.csv` (2,000 Rows)
- `participant_id`: Foreign key matching `participants.csv`.
- `response_id`: Unique response ID (e.g. `NLP-A1B2C3D4`).
- `month`: Integer (`1` to `24`), foreign key matching `monthly_reports.csv`.
- `month_name`: Human-readable month label (`Month 1`, `Month 2`).
- `collection_date`: Date text was collected.
- `question_prompt`: Prompt question presented to participant.
- `response_text`: Qualitative transcript text / youth narrative.
- `word_count`: Word count of `response_text`.
- **NLP Targets & Annotations**:
  - `sentiment_score`: Continuous sentiment score ($-1.0$ to $+1.0$).
  - `sentiment_manual`: 5-tier classification (`positive`, `mildly_positive`, `neutral`, `mildly_negative`, `negative`).
  - `severity_level`: Distress severity (`low`, `moderate`, `high`).
  - `emotional_label`: Dominant emotion (`joy`, `hope`, `fear`, `sadness`, `disappointment`, `confusion`).
  - `clinical_status`: Clinical status (`normal`, `anxious`, `stressed`, `depressed`, `suicidal_ideation`).
  - `anxiety_level`, `depression_level`, `stress_level`: Binned level string (`low`, `moderate`, `high`).
  - `suicidality_flag`: `yes` / `no`.
  - `requires_referral`: `yes` / `no`.
  - `alert_priority`: `p0` (crisis), `p1` (high priority), `p2` (standard).
  - `thematic_codes`: Python list of string codes (e.g. `['academic_pressure', 'financial_stress']`).

---

## 4. How to Join / Merge Datasets in Python (Pandas)

> [!IMPORTANT]
> **Do NOT use `pd.concat([df1, df2], axis=1)` (Positional Row Concatenation)**.  
> Row 1 of `nlp_responses.csv` does **not** correspond to Row 1 of `monthly_reports.csv` because row counts and granularities differ.  
> **Always use foreign key joins (`pd.merge`) on `participant_id` and `month`.**

### Example 1: Merging NLP Text Responses with Monthly Clinical Scores
```python
import pandas as pd

# Load CSV files
nlp_df = pd.read_csv("data/raw/nlp_responses.csv")
monthly_df = pd.read_csv("data/raw/monthly_reports.csv")
participants_df = pd.read_csv("data/raw/participants.csv")

# 1. Join NLP responses with matching monthly report scores on (participant_id, month)
nlp_clinical_df = nlp_df.merge(
    monthly_df,
    on=["participant_id", "month"],
    how="inner",
    suffixes=("_nlp", "_monthly")
)

# 2. Join with baseline participant demographics
master_df = nlp_clinical_df.merge(
    participants_df,
    on="participant_id",
    how="left"
)

print(f"Master merged dataset shape: {master_df.shape}")
# Yields exactly 2,000 rows, each linking qualitative text to that participant's exact clinical scores for that month!
```

### Example 2: Correlating NLP Sentiment / Anxiety Level with Monthly Anxiety Score
```python
# Verify text sentiment correlates with quantitative anxiety score
correlation = master_df.groupby("anxiety_level")["anxiety"].mean()
print("Mean Monthly Anxiety Score by NLP Anxiety Level:")
print(correlation)
```

---

## 5. Summary Checklist for ML Workflow

1. **Text Models (NLP Sentiment, Emotion & Risk Extraction)**:
   - Use `nlp_responses.csv`. Target columns: `sentiment_score`, `emotional_label`, `severity_level`, `clinical_status`.
2. **Tabular Models (Risk Flag Prediction & Psychosocial Trajectory Forecasting)**:
   - Use `consolidated_ml_dataset.csv` or `monthly_reports.csv` merged with `participants.csv`. Target column: `risk_flag` or `anxiety` / `depression`.
3. **Multimodal Joint Models (Combined Text + Tabular Clinical Scores)**:
   - Merge `nlp_responses.csv` + `monthly_reports.csv` + `participants.csv` using the `pd.merge` pattern above.

---

## 6. Official NEPS V3.1 Clinical Survey Contract (73-Item NSES & Production Database Schema)

> [!IMPORTANT]
> **Update (October 2026)**: The NEPS Board and Data Platform team have frozen the official **NEPS Master CRF V3.1** data contract and psychometric scoring rules.  
> ML models transitioning from synthetic development to production ingestion should align their feature definitions with the **73 verified clinical core items** below.

### 6.1 The 7 Verified Psychometric Sub-Indices (73 Items Total)

The official **NEPS Stress Experience Scale (NSES)** is computed strictly from 73 validated screening questions across 7 clinical indices. All 73 items use standardized integer Likert scales:

| Index Code | Instrument / Domain Name | Item Count | Variable Prefix | Measurement Focus & Scale |
| :--- | :--- | :---: | :--- | :--- |
| **EPI** | Educational Pressure Index | 10 | `epi_expectations_1-2`, `epi_failure_1-2`, `epi_caregivers_1-2`, `epi_future_1-2`, `epi_progress_1-2` | Academic expectations, failure anxiety, caregiver pressure (1–5 Likert). |
| **ESI** | Economic Stress Index | 10 | `esi_finstrain_1-2`, `esi_food_1-2`, `esi_educost_1-2`, `esi_materials_1-2`, `esi_transport_1-2` | Financial strain, food insecurity, schooling costs, transportation (1–5 Likert). |
| **SBI** | Stigma & Burden Index | 19 | `sbi_loved_1-8`, `sbi_comm_1-4`, `sbi_self_1-4`, `sbi_help_1-3` | Loved ones' stigma, community stigma, internalized self-stigma, help-seeking reluctance (1–5 Likert). |
| **K6** | Kessler Psychological Distress Scale | 6 | `k6_1` to `k6_6` | Non-specific distress: nervousness, hopelessness, restlessness, depression, effort, worthlessness (0–4 scale). |
| **BRS** | Brief Resilience Scale | 6 | `brs_01` to `brs_06` | Ability to bounce back from stress and recover from hard times (1–5 Likert). |
| **MSPSS**| Multidimensional Scale of Perceived Social Support | 12 | `mspss_01` to `mspss_12` | Perceived social support from Family, Friends, and Significant Others (1–7 Likert). |
| **GSE** | General Self-Efficacy Scale | 10 | `gse_01` to `gse_10` | Optimistic self-belief in coping with life difficulties (1–4 Likert). |

### 6.2 Production Database Tables (Where ML Features Live)

When the Data Platform ingests live surveys into PostgreSQL, the ML team can query feature sets from these three primary tables:

1. **`nds_v31_source` (Raw Intake)**:
   - Stores all **188 raw survey responses** in a JSON `payload` column.
   - Includes survey timestamps, country (`country`), and links to the NDS participant (`nds_participant_id`).
2. **`nds_v31_derived` (Clinical Psychometric Scores)**:
   - Contains the calculated **NSES Composite Score** and individual dimension sub-scores (`result` JSON column).
   - Key JSON fields:
     ```json
     {
       "nses_composite_score": 58.4,
       "subscales": {
         "epi": 24.0,
         "esi": 18.0,
         "sbi": 42.0,
         "k6": 12.0,
         "brs": 19.0,
         "mspss": 65.0,
         "gse": 28.0
       },
       "calculation_version": "v3.1-provisional"
     }
     ```
3. **`nds_v31_participant_links` (Participant Cross-Walk)**:
   - Maps the REDCap field record ID (`source_id`) to the official NDS research participant ID (`nds_id`).
   - Ensures ML predictions can join on `nds_id` without exposing raw field survey tokens.
