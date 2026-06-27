# NEPS ML/AI Module
NEPS Digital: Machine Learning & Artificial Intelligence module for youth mental health analysis.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Quick Start: Using Mock REDCap Data](#quick-start-using-mock-redcap-data)
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

## Quick Start: Using Mock REDCap Data

We have two options for using mock REDCap data:

### Option 1: Use Hosted Mock REDCap API (Recommended)
The mock API is deployed on Render for all teams to use!

1. Create a `.env` file in the project root:
```bash
REDCAP_API_URL=https://mock-redcap-service.onrender.com/api
REDCAP_API_TOKEN=mock_token_neps_2025
```

2. Make HTTP requests to the API (e.g., using `requests`):
```python
import requests
import os

api_url = os.getenv("REDCAP_API_URL")
api_token = os.getenv("REDCAP_API_TOKEN")

# Get all participants
response = requests.get(f"{api_url}/participants")
participants = response.json()["data"]

# Get project stats
stats_response = requests.get(f"{api_url}/stats")
print(stats_response.json())
```

### Option 2: Use Local Embedded Mock (No Network Needed)
We've also included a local mock REDCap dataset!

1. Test the Mock Data:
```bash
python test_mock.py
```

This should show you:
- 150 participants across 3 countries (Ghana, Sierra Leone, Tanzania)
- 24 months of survey data
- Distress screenings
- WP6 intervention sessions

2. Use the Mock in Your Code:
```python
from app.services.redcap_mock import RedCapMockClient

client = RedCapMockClient()

# Get all participants
participants = client.get_participants()

# Filter by country
ghana_participants = client.get_participants(country="Ghana")

# Get survey responses
survey_responses = client.get_survey_responses()

# Export all records
all_records = client.export_records()  # Returns list of dicts
# Or export as CSV
csv_data = client.export_records(format="csv")
```

### 3. Data Available
- **150 participants** (50/year across 3 countries)
- **24 months of monthly self-reports** (85% completion rate)
- **Comprehensive wave data** (6, 12, 18, 24 months)
- **Distress screenings** (10% flagged as high-risk)
- **WP6 intervention data** (20 participants, 8 sessions each)

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
│   └── services/
│       └── redcap_mock.py  # Mock REDCap client
├── models/                  # Pre-trained ML models
├── notebooks/               # Jupyter notebooks for development
├── src/                     # NLP source scripts
├── tests/                   # Test suite
├── .env.example            # Environment variables template
├── .gitignore
├── Dockerfile              # Docker container configuration
├── README.md
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
└── test_mock.py            # Test script for mock data
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
- Metrics: `http://localhost:8001/metrics` (Prometheus format)

