"""Baseline models. Deliberately simple so they retrain in seconds on real data.

- risk_classifier:  monthly scores + demographics -> risk_flag (4 tiers)
- trajectory:       last 3 months -> next month depression/anxiety (regression)
- text:             existing TF-IDF sentiment + emotion models in models/

Each saved with joblib alongside a metadata JSON: data_version, feature
list, metrics, trained_at. See docs/ml/README.md "Model registry".

Deferred until real data: transformer fine-tuning, LSTM/temporal
transformers, multimodal fusion, dropout prediction.

Status: stub.
"""
