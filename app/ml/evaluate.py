"""Evaluation harness.

Rules:
- Split by participant_id, never by row (trajectory leakage otherwise).
- Every report includes a breakdown by schema.FAIRNESS_GROUPS.
- Mock-data metrics are pipeline validation only; risk_flag is a threshold
  on its own inputs and scores ~0.97 with a depth-3 tree.

Status: stub.
"""
