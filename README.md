# NEPS ML/AI Module
NEPS Digital: Machine Learning & Artificial Intelligence module for youth mental health analysis.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Access: V3.1 Source of Truth](#data-access-v31-source-of-truth)
3. [NLP Pipeline & Models](#nlp-pipeline--models)
4. [Jupyter Notebooks](#jupyter-notebooks)
5. [Directory Structure](#directory-structure)
6. [Testing](#testing)
7. [Docker Deployment](#docker-deployment)

## Project Overview
The NEPS ML/AI module provides NLP (Natural Language Processing) capabilities for analyzing qualitative mental health data, including:
- Sentiment Analysis
- Emotion Detection
- Theme Classification
- Risk Explanation
- Recommendation Systems

## Data Access: V3.1 Source of Truth

ML/AI does **not** call REDCap and does **not** generate its own mock data. It reads the
verified V3.1 baseline records and the authoritative NSES/subscale scores directly from
**PostgreSQL**, which is the single source of truth shared by every NEPS team
(frontend, backend, data platform).

```
REDCap (mock or real) → Data Platform → PostgreSQL (nds_v31_source, nds_v31_derived) → ML/AI
```

Swapping mock REDCap for the real REDCap instance is a **configuration change in the
Data Platform** (`REDCAP_API_URL` / `REDCAP_MOCK_ENABLED`); it requires no change to ML/AI.

### Connection

The service does not hardcode the database user or password. In Docker it builds
the URL from connection parts plus mounted secrets, exactly like `neps-backend`:

```bash
DATABASE_USER_FILE=/run/secrets/postgres_user
DATABASE_PASSWORD_FILE=/run/secrets/postgres_password
```

For local development outside Docker, set the connection parts instead
(`DATABASE_HOST`, `DATABASE_USER`, `DATABASE_PASSWORD` — see `.env.example`).
There is deliberately no `DATABASE_URL` override, so a stale `.env` can never
shadow the mounted secrets. TLS is controlled by `DATABASE_SSL_MODE`.

Running the service on the host needs the warehouse port published, which the
infrastructure dev overlay does (from `neps-infrastructure`):

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres
```

Then point the parts at it — `DATABASE_HOST=localhost`, `DATABASE_PORT=15432`.
15432 rather than 5432, because a native PostgreSQL install usually holds 5432.

### Reading data in Python

```python
from app.data.v31_repository import V31Repository

repo = V31Repository()

dataset = repo.get_v31_baseline_dataset()   # metadata + 223-field payload + score_* columns
features = repo.get_v31_raw_features()      # source payload fields only
scores = repo.get_v31_scores()              # score_NSES, score_EPI, ...
ghana = repo.get_v31_by_country("Ghana")
repo.get_record_count()                     # {'total_records': ..., 'countries': ..., 'scored_records': ...}
```

### HTTP endpoints for inspection

| Endpoint | Purpose |
| :--- | :--- |
| `GET /api/v31/contract` | The 73 core items and the 10 derived score keys/labels |
| `GET /api/v31/stats` | Record/country/scored-record counts |
| `GET /api/v31/dataset` | Dataset summary: record count, feature columns, score coverage, one sample row |

The derived scores are produced by the Data Platform scoring engine and must be read, not
recomputed. `NSES` and `EPI` are `null` until EPI applicability is verified.

## NLP Pipeline & Models

### Pre-trained Models
All trained models are stored in the `models/` directory:
- `sentiment_model.pkl` & `sentiment_vectorizer.pkl`: 5-tier sentiment classification (positive, mildly_positive, neutral, mildly_negative, negative)
- `emotion_detection_model.pkl` & `emotion_detection_vectorizer.pkl`: Multi-label emotion detection
- `sentiment_pipeline.pkl` & `emotion_detection_pipeline.pkl`: End-to-end pipelines for inference

### NLP Scripts
Located in the `src/` directory:
- `nlp-preprocessing.py`: Text preprocessing utilities
- `nlp-sentiment.py`: Sentiment analysis functionality
- `nlp-emotion.py`: Emotion detection functionality

## Jupyter Notebooks
Located in the `notebooks/` directory, documenting the full ML/AI development workflow:
1. `nlp-01-eda.ipynb`: Exploratory Data Analysis of the NLP dataset
2. `nlp-02-text-preprocessing.ipynb`: Text cleaning and preprocessing
3. `nlp-03-sentiment-analysis.ipynb`: Sentiment analysis model development
4. `nlp-04-emotion-detection.ipynb`: Emotion detection model development
5. `nlp-05-theme-classification.ipynb`: Thematic analysis and classification
6. `nlp-06-risk-explanation.ipynb`: Risk factor explanation and interpretability
7. `nlp-07-transformer-model.ipynb`: Transformer-based model experiments
8. `nlp-08-recommendation-system.ipynb`: Intervention recommendation system

## Directory Structure
```
neps-ml-ai/
├── app/                     # Application code
│   ├── __init__.py
│   ├── config.py           # Settings (Docker-secret aware)
│   ├── data/
│   │   └── v31_repository.py  # V3.1 PostgreSQL repository
│   └── db/
│       └── connection.py   # SQLAlchemy engine
├── models/                  # Pre-trained ML models
├── notebooks/               # Jupyter notebooks for development
├── src/
│   ├── neps_v31_clinical_contract.py  # 73-item + derived-score contract
│   └── nlp-*.py             # NLP source scripts
├── tests/                   # Test suite
├── .env.example            # Environment variables template
├── .gitignore
├── Dockerfile              # Docker container configuration
├── README.md
├── main.py                 # FastAPI entry point
└── requirements.txt        # Python dependencies
```

## Testing
Run the test suite:
```bash
python -m pytest tests/ -v
```

## Docker Deployment
Build and run the Docker container:
```bash
docker build -t neps-ml-ai:latest .
docker run -p 8001:8000 neps-ml-ai:latest
```

The service exposes:
- Health check: `http://localhost:8001/health`
- V3.1 contract: `http://localhost:8001/api/v31/contract`
- V3.1 stats: `http://localhost:8001/api/v31/stats`
- V3.1 dataset summary: `http://localhost:8001/api/v31/dataset`
- Metrics: `http://localhost:8001/metrics` (Prometheus format)

